"""Tests of the pore water: buoyancy, drag, porosity and their coupling to the simulation."""

import math
import re
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies import (DEMSimulation, Domain, Geometry, Geometry3D, Heaviside, LoadedPlaten, MaterialProperties,
                        Particle, Particle3D, TimeHistory, Wall, Water, generate_lattice_particles)
from DEMcookies.water import PorePressure, PorosityField, drag_force, porosity_3d, submerged_fraction


def _material(density: float = 2500.0) -> MaterialProperties:
    """Return the reference material, heavily damped so that tests settle quickly."""
    return MaterialProperties(density=density, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.1, friction=0.5)


def _hertz_overlap(force: float, radius: float) -> float:
    """Return the Hertz overlap of a particle pressed against a wall by a force [m]."""
    material = _material()
    e_star = material.youngs_modulus / (2.0 * (1.0 - material.poisson_ratio**2))
    return (3.0 * force / (4.0 * e_star * math.sqrt(radius)))**(2.0 / 3.0)


def _volume(radius: float) -> float:
    """Return the volume of a sphere [m^3]."""
    return 4.0 / 3.0 * math.pi * radius**3


def _solve(function: Callable[[float], float], target: float, high: float = 10.0) -> float:
    """Return the root of an increasing function equal to a target, by bisection."""
    low = 0.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if function(middle) < target else (low, middle)
    return 0.5 * (low + high)


def _vertical_drag(speed: float, radius: float, porosity: float = 1.0) -> float:
    """Return the drag of water rising past a fixed sphere [N]."""
    water = Water(level=0.0)
    return float(drag_force(np.array([[0.0, speed]]), np.array([radius]), np.array([porosity]), water)[0, 1])


class TestWaterSpecification:
    """Tests of the Water input checks."""

    @pytest.mark.parametrize("field, value, message", [
        ("density", 0.0, "Water density must be positive, got 0.0."),
        ("viscosity", -1.0, "Water viscosity must be positive, got -1.0."),
        ("cell_size", 0.0, "Water cell_size must be positive, got 0.0."),
        ("bulk_modulus", -1.0, "Water bulk_modulus must be positive, got -1.0."),
        ("level", float("nan"), "Water level must be a finite number or a function of time, got nan."),
        ("seepage_velocity", "fast", "Water seepage_velocity must be a finite number or a function of time"),
    ])
    def test_invalid_values_are_rejected(self, field: str, value: object, message: str) -> None:
        """Unphysical water properties must be rejected with a specific message."""
        arguments: dict = {"level": 1.0, field: value}
        with pytest.raises(ValueError, match=re.escape(message)):
            Water(**arguments)

    def test_level_may_be_a_function_of_time(self) -> None:
        """A water level given as a function of time must be accepted."""
        Water(level=TimeHistory([0.0, 1.0], [0.0, 0.5]))


class TestBuoyancyAndPorosity:
    """Tests of the submerged volume and of the porosity estimates."""

    def test_submerged_fraction(self) -> None:
        """The submerged fraction must follow the volume of a spherical cap."""
        heights = np.array([2.0, -2.0, 0.0, 0.5])
        # the last sphere is cut half a radius above its bottom, a cap of volume 5/32 of the sphere
        assert_allclose(submerged_fraction(heights, np.ones(4), 0.0), [0.0, 1.0, 0.5, 5.0 / 32.0], atol=1e-15)

    def test_porosity_conversion(self) -> None:
        """Pure water must stay pure water, and hexagonal disc packing must map onto face-centred cubic packing."""
        hexagonal = 1.0 - math.pi / (2.0 * math.sqrt(3.0))
        assert_allclose(porosity_3d(np.array([1.0, hexagonal])), [1.0, 1.0 - math.pi / (3.0 * math.sqrt(2.0))])

    def test_square_lattice_porosity(self) -> None:
        """A square lattice of touching discs must have porosity 1 - pi/4 everywhere, also next to the walls."""
        radius = 0.005
        xs = np.arange(radius, 0.4, 2.0 * radius)
        positions = np.array([(x, y) for x in xs for y in xs])
        field = PorosityField(Geometry.rectangle(0.4, 0.4), 0.04)
        assert_allclose(field.porosity_2d(positions, np.full(len(positions), radius)), 1.0 - math.pi / 4.0)

    def test_isolated_particle_sees_water(self) -> None:
        """A single small particle in a large cell must see a porosity close to one."""
        field = PorosityField(Geometry.rectangle(1.0, 1.0), 0.5)
        assert field.porosity_2d(np.array([[0.5, 0.5]]), np.array([0.005]))[0] > 0.999


class TestDragForce:
    """Tests of the Di Felice drag law."""

    def test_no_relative_velocity_no_drag(self) -> None:
        """A particle moving with the water feels no drag."""
        assert_allclose(_vertical_drag(0.0, 0.005), 0.0, atol=1e-18)

    def test_stokes_limit(self) -> None:
        """At vanishing Reynolds number the drag of an isolated sphere must approach 0.96 of Stokes' law."""
        radius, speed = 0.001, 1.0e-10
        stokes = 6.0 * math.pi * 1.0e-3 * radius * speed
        # (4.8)^2 / 24 = 0.96 is the low-Reynolds limit of the Dallavalle drag coefficient, approached as sqrt(Re)
        assert_allclose(_vertical_drag(speed, radius), 0.96 * stokes, rtol=1e-3)

    @pytest.mark.parametrize("speed", [1.0e-4, 0.01, 0.1, 1.0])
    def test_crowding_increases_drag(self, speed: float) -> None:
        """At the same relative velocity, the drag must grow steadily as the packing around the sphere densifies."""
        drags = [_vertical_drag(speed, 0.005, porosity) for porosity in (1.0, 0.8, 0.6, 0.4, 0.3)]
        assert np.all(np.diff(drags) > 0.0)


class TestSubmergedParticles:
    """Tests of complete simulations with water against closed-form results."""

    def test_submerged_particle_rests_on_buoyant_weight(self) -> None:
        """Under water a resting particle must press on the floor with its buoyant weight only."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            _material(),
                            particles=[Particle(0.05, 0.01, 0.01)],
                            water=Water(level=0.08))
        sim.rest(0.5)
        expected = _hertz_overlap((2500.0 - 1000.0) * _volume(0.01) * 9.81, 0.01)
        assert_allclose(0.01 - sim.pos[0, 1], expected, rtol=1e-3)

    def test_light_particle_floats_half_submerged(self) -> None:
        """A particle of half the density of water must float with its centre at the free surface."""
        # viscous water damps the bobbing of the sphere, released half a millimetre too high; it touches nothing,
        # so the time step only has to resolve the drag
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.2),
                            _material(density=500.0),
                            particles=[Particle(0.05, 0.1005, 0.01)],
                            dt=2.5e-4,
                            water=Water(level=0.1, viscosity=0.05))
        sim.rest(4.0)
        assert_allclose(sim.pos[0, 1], 0.1, atol=1e-5)

    def test_terminal_settling_velocity(self) -> None:
        """A sphere settling in still water must reach the speed at which drag balances its buoyant weight."""
        radius = 0.005
        sim = DEMSimulation(Geometry.rectangle(0.4, 2.0),
                            _material(),
                            particles=[Particle(0.2, 1.9, radius)],
                            water=Water(level=2.0, cell_size=0.4))
        sim.rest(0.8)
        weight = (2500.0 - 1000.0) * _volume(radius) * 9.81
        expected = _solve(lambda speed: _vertical_drag(speed, radius, float(sim.porosity[0])), weight)
        assert_allclose(-sim.vel[0, 1], expected, rtol=1e-4)

    @pytest.mark.parametrize("ratio, lifts", [(0.8, False), (1.2, True)])
    def test_upward_seepage_lifts_particle_beyond_terminal_velocity(self, ratio: float, lifts: bool) -> None:
        """An upward flow must reduce the contact force of a particle, and lift it once it exceeds its fall speed."""
        radius = 0.005
        weight = (2500.0 - 1000.0) * _volume(radius) * 9.81
        terminal = _solve(lambda speed: _vertical_drag(speed, radius), weight)
        sim = DEMSimulation(Geometry.rectangle(0.4, 0.4),
                            _material(),
                            particles=[Particle(0.2, radius, radius)],
                            water=Water(level=0.4, seepage_velocity=ratio * terminal, cell_size=0.4))
        sim.rest(0.5)
        assert (sim.pos[0, 1] > 2.0 * radius) == lifts
        if not lifts:
            porosity = float(sim.porosity[0])
            # in a packing the water pushes with the drag divided by the porosity
            drag = _vertical_drag(ratio * terminal / porosity, radius, porosity) / porosity
            buoyancy = 1000.0 * _volume(radius) * 9.81
            assert_allclose(sim.fluid_force[0, 1], buoyancy + drag, rtol=1e-6)

    def test_rising_water_table_unloads_the_contact(self) -> None:
        """A water table rising over a resting particle must reduce its contact force to the buoyant weight."""
        rising = TimeHistory([0.0, 0.2, 0.4], [0.0, 0.0, 0.05])
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            _material(),
                            particles=[Particle(0.05, 0.01, 0.01)],
                            water=Water(level=rising))
        sim.rest(0.15)
        assert_allclose(0.01 - sim.pos[0, 1], _hertz_overlap(2500.0 * _volume(0.01) * 9.81, 0.01), rtol=1e-3)
        sim.rest(0.6)
        assert_allclose(0.01 - sim.pos[0, 1], _hertz_overlap(1500.0 * _volume(0.01) * 9.81, 0.01), rtol=1e-3)
        level = sim.frames[-1].water_level
        assert level is not None
        assert_allclose(level, 0.05)

    def test_particle_above_water_feels_nothing(self) -> None:
        """A particle entirely above the free surface must feel no force from the water."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            _material(),
                            particles=[Particle(0.05, 0.05, 0.01)],
                            water=Water(level=0.02))
        sim.compute_forces()
        assert_allclose(sim.fluid_force, 0.0, atol=1e-18)

    def test_viscous_drag_is_stable_at_the_default_time_step(self) -> None:
        """Drag is integrated implicitly: very viscous water must not shorten the time step nor destabilise it."""
        radius, viscosity = 0.001, 100.0
        dry = DEMSimulation(Geometry.rectangle(0.1, 0.2), _material(), particles=[Particle(0.05, 0.15, radius)])
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.2),
                            _material(),
                            particles=[Particle(0.05, 0.15, radius)],
                            water=Water(level=0.2, viscosity=viscosity, cell_size=0.1))
        assert_allclose(sim.dt, dry.dt)
        sim.rest(0.05)
        weight = (2500.0 - 1000.0) * _volume(radius) * 9.81
        water = Water(level=0.0, viscosity=viscosity)
        porosity = float(sim.porosity[0])

        def drag(speed: float) -> float:
            return float(drag_force(np.array([[0.0, speed]]), np.array([radius]), np.array([porosity]), water)[0, 1])

        expected = _solve(lambda speed: drag(speed) / porosity, weight, high=1.0)
        assert_allclose(-sim.vel[0, 1], expected, rtol=1e-6)

    def test_coupling_shortens_the_default_time_step(self) -> None:
        """A stiff pore fluid must shorten the default time step so that the pore pressure coupling stays stable."""
        drained = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                                _material(),
                                radius_range=(0.004, 0.006),
                                water=Water(level=0.1))
        coupled = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                                _material(),
                                radius_range=(0.004, 0.006),
                                water=Water(level=0.1, coupled=True, bulk_modulus=1.0e9))
        assert coupled.dt < 0.5 * drained.dt

    def test_too_large_time_step_is_rejected(self) -> None:
        """A user time step that cannot resolve the pore pressure coupling must be rejected."""
        with pytest.raises(ValueError, match=re.escape("is too large for the pore pressure coupling")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1),
                          _material(),
                          radius_range=(0.004, 0.006),
                          dt=1.0e-4,
                          water=Water(level=0.1, coupled=True, bulk_modulus=1.0e9))


class TestWaterStages:
    """Tests of changing the water between stages and of the edge forces."""

    def test_edge_forces_carry_the_weight(self) -> None:
        """The floor must carry the full weight of a dry particle, and the buoyant weight once it is flooded."""
        weight = 2500.0 * _volume(0.01) * 9.81
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[Particle(0.05, 0.01, 0.01)])
        sim.rest(0.3)
        assert_allclose(sim.edge_forces[0], [0.0, -weight], rtol=1e-3, atol=1e-12)
        assert_allclose(sim.edge_forces[1:], 0.0, atol=1e-15)

        sim.set_water(Water(level=0.05))
        sim.rest(0.3, phase="flooded")
        assert_allclose(sim.edge_forces[0, 1], -weight * 1500.0 / 2500.0, rtol=1e-3)
        assert sim.frames[-1].phase == "flooded"

    def test_removing_the_water_restores_dry_conditions(self) -> None:
        """Without water there must be no fluid force and no porosity estimate."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            _material(),
                            particles=[Particle(0.05, 0.01, 0.01)],
                            water=Water(level=0.05))
        sim.compute_forces()
        assert sim.fluid_force[0, 1] > 0.0
        sim.set_water(None)
        sim.compute_forces()
        assert_allclose(sim.fluid_force, 0.0, atol=1e-18)
        assert np.all(np.isnan(sim.porosity))

    def test_coupling_rejects_a_user_time_step(self) -> None:
        """Adding coupled water must not silently change a time step that the user chose."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.004, 0.006), dt=1.0e-4)
        with pytest.raises(ValueError, match=re.escape("is too large for the pore pressure coupling")):
            sim.set_water(Water(level=0.1, coupled=True, bulk_modulus=1.0e9))


def _terzaghi(depth: float, height: float, time_factor: float) -> float:
    """Return Terzaghi's excess pore pressure, as a fraction of its initial value, at a depth below the drained top."""
    total = 0.0
    for m in range(200):
        wave = (2 * m + 1) * math.pi / 2.0
        total += 2.0 / wave * math.sin(wave * depth / height) * math.exp(-wave**2 * time_factor)
    return total


class TestPorePressure:
    """Tests of the excess pore pressure solver and of its coupling to the particles."""

    def test_pressure_diffusion_follows_terzaghi(self) -> None:
        """A uniform excess pressure in a column drained at the top must dissipate as Terzaghi's solution."""
        cell, height = 0.05, 0.5
        geometry = Geometry.rectangle(0.1, 1.0)
        field = PorosityField(geometry, cell)
        pressure = PorePressure(field, geometry)
        nodes = field.node_positions
        porosity, mobility, stiffness = np.full(len(nodes), 0.4), np.full(len(nodes), 1.0e-6), 1.0e6
        diffusivity = mobility[0] * stiffness / porosity[0]
        # the free surface lies just below the node row at the drained height
        level = height - 0.25 * cell
        pressure.pressure[:] = np.where((nodes[:, 1] < level) & (field.node_area > 0.0), 1.0, 0.0)
        centre = np.abs(nodes[:, 0] - 0.05) < 1e-9
        base = centre & (np.abs(nodes[:, 1]) < 1e-9)
        middle = centre & (np.abs(nodes[:, 1] - 0.25) < 1e-9)
        dt, time = 2.0e-5, 0.0
        for target in (0.2, 0.5, 1.0):
            while diffusivity * time / height**2 < target - 1e-12:
                pressure.advance(dt, porosity, mobility, np.zeros((len(nodes), 2)), level, stiffness)
                time += dt
            factor = diffusivity * time / height**2
            assert_allclose(pressure.pressure[base][0], _terzaghi(height, height, factor), atol=0.01)
            assert_allclose(pressure.pressure[middle][0], _terzaghi(0.25, height, factor), atol=0.01)

    def test_coupled_water_carries_a_sudden_load_first(self) -> None:
        """Under a fast load the water must first carry it, and the skeleton once the excess pressure dissipates."""
        load = 5.0
        geometry = Geometry.rectangle(0.08,
                                      0.2,
                                      right=Wall(friction=0.0),
                                      left=Wall(friction=0.0),
                                      top=LoadedPlaten(load=Heaviside(load, rise_time=0.01), mass=0.2))
        particles = generate_lattice_particles(geometry, 60, (0.004, 0.005), seed=1)
        sim = DEMSimulation(geometry, _material(), particles=particles, water=Water(level=0.19))
        sim.rest(0.4)
        # viscous water drains slowly; a soft pore fluid is still about 25 times stiffer than this skeleton
        sim.set_water(Water(level=0.19, coupled=True, viscosity=1.0, bulk_modulus=2.0e6))
        sim.rest(0.02)
        floor_before = -sim.edge_forces[0, 1]
        sim.compress(0.02)
        early_share = (-sim.edge_forces[0, 1] - floor_before) / load
        early_pressure = sim.excess_pore_pressure(np.array([[0.04, 0.003]]))[0]
        sim.compress(0.28)
        late_share = (-sim.edge_forces[0, 1] - floor_before) / load
        late_pressure = sim.excess_pore_pressure(np.array([[0.04, 0.003]]))[0]
        assert early_share < 0.3
        assert early_pressure > 1000.0
        assert late_share > 0.9
        assert late_pressure < 0.1 * early_pressure
        assert_allclose(sim.platens[0].reaction, load, rtol=0.02)

    def test_coupling_keeps_the_settling_of_a_lone_particle(self) -> None:
        """A particle alone in clear water must settle as in drained water: the skeleton carries no water there."""
        speeds = []
        for coupled in (False, True):
            sim = DEMSimulation(Geometry.rectangle(0.4, 2.0),
                                _material(),
                                particles=[Particle(0.2, 1.9, 0.005)],
                                water=Water(level=2.0, cell_size=0.4, coupled=coupled))
            sim.rest(0.8)
            speeds.append(-sim.vel[0, 1])
        assert_allclose(speeds[1], speeds[0], rtol=1e-3)

    def test_drained_water_has_no_excess_pressure(self) -> None:
        """Drained water must report no excess pore pressure."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            _material(),
                            particles=[Particle(0.05, 0.01, 0.01)],
                            water=Water(level=0.08))
        assert_allclose(sim.excess_pore_pressure(np.array([[0.05, 0.01], [0.02, 0.02]])), 0.0, atol=1e-18)


class TestWater3D:
    """Tests of the pore water in a 3D domain."""

    def test_submerged_particle_rests_on_buoyant_weight(self) -> None:
        """Under water a resting particle must press on the floor face with its buoyant weight only."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                            _material(),
                            particles=[Particle3D(0.05, 0.01, 0.05, 0.01)],
                            water=Water(level=0.08))
        sim.rest(0.5)
        expected = _hertz_overlap((2500.0 - 1000.0) * _volume(0.01) * 9.81, 0.01)
        assert_allclose(0.01 - sim.pos[0, 1], expected, rtol=1e-3)

    def test_terminal_settling_velocity(self) -> None:
        """A sphere settling in still water must fall straight at the speed at which drag balances its weight."""
        radius = 0.005
        sim = DEMSimulation(Geometry3D.box(0.4, 2.0, 0.4),
                            _material(),
                            particles=[Particle3D(0.2, 1.9, 0.2, radius)],
                            water=Water(level=2.0, cell_size=0.4))
        sim.rest(0.8)
        weight = (2500.0 - 1000.0) * _volume(radius) * 9.81
        expected = _solve(lambda speed: _vertical_drag(speed, radius, float(sim.porosity[0])), weight)
        assert_allclose(-sim.vel[0, 1], expected, rtol=1e-4)
        assert_allclose(sim.vel[0, [0, 2]], 0.0, atol=1e-15)

    def test_simple_cubic_lattice_porosity(self) -> None:
        """A cubic lattice of touching spheres must have porosity 1 - pi/6 everywhere, also next to the walls."""
        radius = 0.005
        axis = np.arange(radius, 0.2, 2.0 * radius)
        positions = np.array([(x, y, z) for x in axis for y in axis for z in axis])
        field = PorosityField(Geometry3D.box(0.2, 0.2, 0.2), 0.04)
        assert field.shape == (8, 8, 8)
        assert_allclose(field.measured_porosity(positions, np.full(len(positions), radius)), 1.0 - math.pi / 6.0)
        # spheres need no conversion from the porosity of a layer of discs
        assert_allclose(field.volume_porosity(np.array([0.3, 1.2])), [0.3, 1.0])
        with pytest.raises(ValueError, match=re.escape("A 3D domain has no disc layer")):
            field.porosity_2d(positions, np.full(len(positions), radius))

    def test_pressure_diffusion_follows_terzaghi(self) -> None:
        """A uniform excess pressure in a column drained at the top must dissipate as Terzaghi's solution."""
        cell, height = 0.05, 0.5
        geometry = Geometry3D.box(0.1, 1.0, 0.1)
        field = PorosityField(geometry, cell)
        pressure = PorePressure(field, geometry)
        nodes = field.node_positions
        porosity, mobility, stiffness = np.full(len(nodes), 0.4), np.full(len(nodes), 1.0e-6), 1.0e6
        diffusivity = mobility[0] * stiffness / porosity[0]
        level = height - 0.25 * cell
        pressure.pressure[:] = np.where((nodes[:, 1] < level) & (field.node_volume > 0.0), 1.0, 0.0)
        centre = (np.abs(nodes[:, 0] - 0.05) < 1e-9) & (np.abs(nodes[:, 2] - 0.05) < 1e-9)
        base = centre & (np.abs(nodes[:, 1]) < 1e-9)
        middle = centre & (np.abs(nodes[:, 1] - 0.25) < 1e-9)
        dt, time = 2.0e-5, 0.0
        for target in (0.2, 0.5, 1.0):
            while diffusivity * time / height**2 < target - 1e-12:
                pressure.advance(dt, porosity, mobility, np.zeros((len(nodes), 3)), level, stiffness)
                time += dt
            factor = diffusivity * time / height**2
            assert_allclose(pressure.pressure[base][0], _terzaghi(height, height, factor), atol=0.01)
            assert_allclose(pressure.pressure[middle][0], _terzaghi(0.25, height, factor), atol=0.01)

    @pytest.mark.parametrize("geometry", [Geometry.rectangle(0.1, 1.0), Geometry3D.box(0.1, 1.0, 0.1)])
    def test_steady_drainage_of_carried_water(self, geometry: Domain) -> None:
        """Water carried down onto an impermeable floor must drain back up through a linear pressure profile.

        At steady state the Darcy flow returns the carried flux c, so p = c (y_d - y) / (k / mu) below the drained
        node at y_d, whatever the cells cut by the walls.
        """
        cell, carried_speed, mobility = 0.05, 1.0e-3, 2.0e-6
        field = PorosityField(geometry, cell)
        pressure = PorePressure(field, geometry)
        nodes = field.node_positions
        carried = np.zeros((len(nodes), geometry.dimension))
        carried[:, 1] = -carried_speed
        level = 0.5 - 0.25 * cell
        # a step much longer than the diffusion time reaches the steady state at once
        pressure.advance(1.0e6, np.full(len(nodes), 0.4), np.full(len(nodes), mobility), carried, level, 1.0e6)
        free = (field.node_volume > 0.0) & (nodes[:, 1] < level)
        expected = carried_speed * (0.5 - nodes[free, 1]) / mobility
        assert_allclose(pressure.pressure[free], expected, rtol=1e-6)

    def test_coupling_shortens_the_default_time_step(self) -> None:
        """A stiff pore fluid must shorten the default time step of a 3D simulation too."""
        drained = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                                _material(),
                                radius_range=(0.004, 0.006),
                                water=Water(level=0.1))
        coupled = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                                _material(),
                                radius_range=(0.004, 0.006),
                                water=Water(level=0.1, coupled=True, bulk_modulus=1.0e9))
        assert coupled.dt < 0.5 * drained.dt

    def test_coupled_water_resists_a_sudden_load(self) -> None:
        """Under a fast load an excess pore pressure must build up below a 3D packing."""
        geometry = Geometry3D.box(0.06, 0.12, 0.06, top=LoadedPlaten(load=Heaviside(5.0, rise_time=0.002), mass=0.2))
        particles = generate_lattice_particles(geometry, 80, (0.004, 0.005), seed=1)
        sim = DEMSimulation(geometry, _material(), particles=particles, water=Water(level=0.11))
        sim.rest(0.15)
        assert_allclose(sim.excess_pore_pressure(np.array([[0.03, 0.003, 0.03]])), 0.0, atol=1e-18)
        sim.set_water(Water(level=0.11, coupled=True, viscosity=1.0, bulk_modulus=2.0e6))
        sim.compress(0.008)
        assert sim.excess_pore_pressure(np.array([[0.03, 0.003, 0.03]]))[0] > 10.0
        assert sim.fluid_force.shape == (80, 3)
