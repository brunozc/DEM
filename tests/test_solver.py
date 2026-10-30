"""Tests of the integrator, the settings and the DEM simulation engine."""

import math
import pickle
import re
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry, Geometry3D, GlobalSettings, Heaviside, LoadedPlaten,
                        MaterialProperties, Open, Particle, Particle3D, PrescribedPlaten, RainSettings, Sinusoidal,
                        TimeHistory, Wall, generate_lattice_particles)
from DEMcookies.solver import VelocityVerlet


def _material(restitution: float = 0.1, friction: float = 0.5, rolling_friction: float = 0.0) -> MaterialProperties:
    """Return the reference material, heavily damped by default so that tests settle quickly."""
    return MaterialProperties(density=2500.0,
                              youngs_modulus=5.0e6,
                              poisson_ratio=0.25,
                              restitution=restitution,
                              friction=friction,
                              rolling_friction=rolling_friction)


def _sphere_on_incline(slope: float, rolling_friction: float) -> DEMSimulation:
    """Return a 10 mm particle resting on the middle of a 0.4 m long incline of the given gradient tan(theta)."""
    material = _material(friction=0.8, rolling_friction=rolling_friction)
    length, radius = 0.4, 0.01
    rise = slope * length
    geometry = Geometry([(0.0, rise), (length, 0.0), (length, rise + 0.1), (0.0, rise + 0.1)],
                        [Wall(), Wall(), Open(), Wall()])
    normal = np.array([slope, 1.0]) / math.hypot(slope, 1.0)
    weight = _sphere_mass(material, radius) * 9.81
    gap = radius - _static_overlap(material, radius, weight * normal[1])
    centre = np.array([0.5 * length, 0.5 * rise]) + gap * normal
    return DEMSimulation(geometry, material, particles=[Particle(centre[0], centre[1], radius)])


def _static_overlap(material: MaterialProperties, radius: float, force: float) -> float:
    """Return the Hertz overlap of a particle pressed against a wall by a force [m]."""
    e_star = material.youngs_modulus / (2.0 * (1.0 - material.poisson_ratio**2))
    return (3.0 * force / (4.0 * e_star * math.sqrt(radius)))**(2.0 / 3.0)


def _sphere_mass(material: MaterialProperties, radius: float) -> float:
    """Return the mass of a particle [kg]."""
    return material.density * 4.0 / 3.0 * math.pi * radius**3


def _seated_indentation(material: MaterialProperties, radius: float, force: float) -> float:
    """Return how far a platen seated on a particle resting on a floor travels to press it with a force [m].

    The particle already rests on the floor at the overlap of its weight, and the platen adds its own overlap on top
    and pushes the particle further into the floor.
    """
    weight = _sphere_mass(material, radius) * 9.81
    return (_static_overlap(material, radius, force) + _static_overlap(material, radius, force + weight) -
            _static_overlap(material, radius, weight))


class TestVelocityVerlet:
    """Tests of the reference integrator."""

    def test_free_fall_is_exact(self) -> None:
        """A constant acceleration must be integrated exactly."""
        dt, g = 1.0e-3, 9.81
        integrator = VelocityVerlet(dt)

        def gravity(positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.tile([0.0, -g], (positions.shape[0], 1))

        positions, velocities, _ = integrator.step(np.zeros((1, 2)), np.zeros((1, 2)), gravity)
        assert_allclose(positions[0], [0.0, -0.5 * g * dt**2], atol=1e-18)
        assert_allclose(velocities[0], [0.0, -g * dt], atol=1e-15)

    def test_non_positive_time_step_is_rejected(self) -> None:
        """A non-positive time step must be rejected."""
        with pytest.raises(ValueError, match=re.escape("dt must be positive, got 0.0.")):
            VelocityVerlet(0.0)


class TestSettings:
    """Tests of GlobalSettings and RainSettings."""

    @pytest.mark.parametrize("field, value, message", [
        ("gravity", -1.0, "gravity must be non-negative, got -1.0."),
        ("time_step_safety", 1.5, "time_step_safety must be in (0, 1], got 1.5."),
        ("quiet_tolerance", 0.0, "quiet_tolerance must be positive, got 0.0."),
        ("neighbour_skin", -0.1, "neighbour_skin must be non-negative, got -0.1."),
        ("escape_margin", -1.0, "escape_margin must be non-negative, got -1.0."),
    ])
    def test_invalid_global_settings_are_rejected(self, field: str, value: float, message: str) -> None:
        """Invalid numerical settings must be rejected with a specific message."""
        with pytest.raises(ValueError, match=re.escape(message)):
            GlobalSettings(**{field: value})

    def test_invalid_rain_count_is_rejected(self) -> None:
        """A wave must insert at least one particle."""
        with pytest.raises(ValueError, match=re.escape("batch_size must be at least 1, got 0.")):
            RainSettings(batch_size=0)

    def test_negative_step_limit_is_rejected(self) -> None:
        """Step limits cannot be negative."""
        with pytest.raises(ValueError, match=re.escape("settle_steps must be non-negative, got -5.")):
            RainSettings(settle_steps=-5)

    def test_empty_insertion_band_is_rejected(self) -> None:
        """The insertion band must have a positive height."""
        with pytest.raises(ValueError, match=re.escape("insertion_band must be (y_min, y_max) with y_min < y_max")):
            RainSettings(insertion_band=(0.5, 0.4))


class TestDEMSimulationSetup:
    """Tests of the input checks of DEMSimulation."""

    def test_radius_range_or_particles_required(self) -> None:
        """An empty simulation needs a radius range for the particles it will rain in."""
        with pytest.raises(ValueError,
                           match=re.escape("radius_range is required when no initial particles are given.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1), _material())

    def test_invalid_radius_range_is_rejected(self) -> None:
        """The radius range must be ordered and positive."""
        with pytest.raises(ValueError, match=re.escape("radius_range must satisfy 0 < r_min <= r_max")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.02, 0.01))

    def test_particle_outside_domain_is_rejected(self) -> None:
        """Initial particles must lie inside the domain."""
        with pytest.raises(ValueError, match=re.escape("Particle 1 at (0.2, 0.05) lies outside the domain.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1),
                          _material(),
                          particles=[Particle(0.05, 0.05, 0.01), Particle(0.2, 0.05, 0.01)])

    def test_non_positive_radius_is_rejected(self) -> None:
        """Initial particles need a positive radius."""
        with pytest.raises(ValueError, match=re.escape("Particle radii must be positive.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[Particle(0.05, 0.05, 0.0)])

    def test_non_positive_time_step_is_rejected(self) -> None:
        """A user time step must be positive."""
        with pytest.raises(ValueError, match=re.escape("dt must be positive, got -1e-05.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01), dt=-1.0e-5)

    def test_compress_without_platen_is_rejected(self) -> None:
        """Loading needs a LoadedPlaten or PrescribedPlaten edge."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        message = "The geometry has no LoadedPlaten or PrescribedPlaten edge to apply a load with."
        with pytest.raises(ValueError, match=re.escape(message)):
            sim.compress(0.1)

    def test_negative_duration_is_rejected(self) -> None:
        """A stage cannot last a negative time."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("duration must be non-negative, got -1.0.")):
            sim.rest(-1.0)

    def test_invalid_record_interval_is_rejected(self) -> None:
        """Frames must be recorded at a positive interval."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("record_every must be at least 1, got 0.")):
            sim.rest(0.01, record_every=0)


class TestDEMSimulationPhysics:
    """Tests of the simulated physics against closed-form results."""

    def test_particle_rests_at_hertz_overlap(self) -> None:
        """A particle resting on a wall must settle at the static Hertz overlap under its own weight."""
        material = _material()
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), material, particles=[Particle(0.05, 0.01, 0.01)])
        sim.rest(0.5)
        expected = _static_overlap(material, 0.01, _sphere_mass(material, 0.01) * 9.81)
        assert_allclose(0.01 - sim.pos[0, 1], expected, rtol=1e-3)

    @pytest.mark.parametrize("shift", [0.0, 1.0e-4, -1.0e-4])
    def test_shared_vertex_gives_one_contact(self, shift: float) -> None:
        """A particle on or next to the vertex between two collinear floor edges must feel a single contact."""
        material = _material()
        vertices = [(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
        geometry = Geometry(vertices, [Wall(), Wall(), Wall(), Open(), Wall()])
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05 + shift, 0.01, 0.01)])
        sim.rest(0.5)
        expected = _static_overlap(material, 0.01, _sphere_mass(material, 0.01) * 9.81)
        assert_allclose(0.01 - sim.pos[0, 1], expected, rtol=1e-3)
        # a spurious second contact at the vertex would push the particle sideways off the vertex
        assert_allclose(sim.pos[0, 0], 0.05 + shift, atol=1e-9)

    def test_collision_conserves_momentum_and_dissipates_energy(self) -> None:
        """An oblique collision must conserve linear momentum exactly and lose kinetic energy."""
        material = _material(restitution=0.5)
        particles = [Particle(0.47, 0.5, 0.01, vx=0.5), Particle(0.53, 0.505, 0.015, vx=-0.5)]
        sim = DEMSimulation(Geometry.rectangle(1.0, 1.0),
                            material,
                            particles=particles,
                            settings=GlobalSettings(gravity=0.0))
        momentum = (sim.mass[:, None] * sim.vel).sum(axis=0)
        energy = sim.mean_kinetic_energy()
        sim.rest(0.08)
        assert_allclose((sim.mass[:, None] * sim.vel).sum(axis=0), momentum, atol=1e-14)
        assert sim.mean_kinetic_energy() < 0.9 * energy
        assert np.all(np.abs(sim.omega) > 1e-3)

    def test_compute_forces_does_not_advance_history(self) -> None:
        """Re-evaluating the forces of an unchanged state must give the same forces every time."""
        material = _material()
        overlap = _static_overlap(material, 0.01, _sphere_mass(material, 0.01) * 9.81)
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            material,
                            particles=[Particle(0.05, 0.01 - overlap, 0.01, vx=1.0e-3)])
        sim.step()
        sim.compute_forces()
        first = sim.force.copy()
        for _ in range(5):
            sim.compute_forces()
        assert_allclose(sim.force, first, rtol=1e-12, atol=1e-15)

    def test_open_edge_lets_particles_out(self) -> None:
        """A particle falling through an open floor must be removed once it is clear of the domain."""
        geometry = Geometry.rectangle(0.1, 0.1, bottom=Open())
        particles = [Particle(0.05, 0.05, 0.005), Particle(0.02, 0.05, 0.005)]
        sim = DEMSimulation(geometry, _material(), particles=particles, settings=GlobalSettings(escape_margin=1.0))
        sim.rest(0.2)
        assert sim.n == 0
        assert sim.n_removed == 2

    def test_strip_footing_loads_only_particles_under_it(self) -> None:
        """A platen on part of the top edge must load the particle under it, reach equilibrium and spare the other."""
        material = _material()
        load = 2.0
        footing = LoadedPlaten(load=Heaviside(load, rise_time=0.05), mass=0.2)
        vertices = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.06, 0.1), (0.04, 0.1), (0.0, 0.1)]
        geometry = Geometry(vertices, [Wall(), Wall(), Open(), footing, Open(), Wall()])
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05, 0.01, 0.01), Particle(0.02, 0.01, 0.01)])
        sim.rest(0.2)
        sim.compress(0.4)

        weight = _sphere_mass(material, 0.01) * 9.81
        platen = sim.platens[0]
        assert_allclose(platen.reaction, load, rtol=1e-2)
        assert_allclose(0.01 - sim.pos[0, 1], _static_overlap(material, 0.01, load + weight), rtol=1e-2)
        assert_allclose(0.01 - sim.pos[1, 1], _static_overlap(material, 0.01, weight), rtol=1e-3)
        expected = _static_overlap(material, 0.01, load) + _static_overlap(material, 0.01, load + weight) - \
            _static_overlap(material, 0.01, weight)
        assert_allclose(platen.displacement, expected, rtol=2e-2)
        curve = sim.load_curves[0].as_array()
        assert curve.shape[1] == 4
        assert_allclose(curve[0, :3], [0.0, 0.0, 0.0], atol=1e-15)

    def test_seated_platen_touches_bed_and_stays_fixed(self) -> None:
        """Seating must lower the platen onto the top particle and hold it there without starting the load."""
        material = _material()
        footing = LoadedPlaten(load=Heaviside(2.0), mass=0.2)
        geometry = Geometry.rectangle(0.1, 0.1, top=footing)
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05, 0.01, 0.01), Particle(0.05, 0.03, 0.01)])
        sim.rest(0.2)
        sim.seat_platens()
        platen = sim.platens[0]
        top = float(np.max(sim.pos[:, 1] + sim.radius))
        assert_allclose(platen.starts[0, 1], top, atol=1e-12)
        assert not platen.active
        offset = platen.offset
        sim.rest(0.05)
        assert_allclose(platen.offset, offset, atol=1e-15)
        assert_allclose(sim.load_time, 0.0, atol=1e-15)
        # loading starts from the seated position, which compress cannot improve on
        sim.compress(0.0)
        assert platen.active
        assert_allclose(platen.offset, offset, atol=1e-9)

    def test_seat_without_platen_is_rejected(self) -> None:
        """Seating needs a LoadedPlaten or PrescribedPlaten edge."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        with pytest.raises(ValueError,
                           match=re.escape("The geometry has no LoadedPlaten or PrescribedPlaten edge to seat.")):
            sim.seat_platens()

    def test_seat_after_loading_is_rejected(self) -> None:
        """A platen that loading has released cannot be seated again."""
        geometry = Geometry.rectangle(0.1, 0.1, top=LoadedPlaten(load=Heaviside(2.0), mass=0.2))
        sim = DEMSimulation(geometry, _material(), particles=[Particle(0.05, 0.01, 0.01)])
        sim.compress(0.0)
        with pytest.raises(ValueError,
                           match=re.escape("Loading has already started; a released platen cannot be seated again.")):
            sim.seat_platens()

    def test_side_platen_pushes_along_its_normal(self) -> None:
        """A platen on a side edge must move inwards along its normal and balance its load."""
        material = _material()
        side = LoadedPlaten(load=Heaviside(1.0, rise_time=0.05), mass=0.2)
        geometry = Geometry.rectangle(0.1, 0.1, right=side)
        sim = DEMSimulation(geometry,
                            material,
                            particles=[Particle(0.01, 0.05, 0.01)],
                            settings=GlobalSettings(gravity=0.0))
        sim.compress(0.4)
        platen = sim.platens[0]
        assert_allclose(platen.direction, [-1.0, 0.0], atol=1e-15)
        assert_allclose(platen.starts[0, 0], 0.02 - 2.0 * _static_overlap(material, 0.01, 1.0), rtol=1e-3)
        assert_allclose(platen.reaction, 1.0, rtol=1e-2)

    def test_wedge_moves_rigidly_and_balances_its_load(self) -> None:
        """A V-shaped body of two edges must move as one, touch the particle once at its tip and balance its load."""
        material = _material()
        load = 2.0
        wedge = LoadedPlaten(load=Heaviside(load, rise_time=0.05), mass=0.2)
        vertices = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.06, 0.1), (0.05, 0.08), (0.04, 0.1), (0.0, 0.1)]
        geometry = Geometry(vertices, [Wall(), Wall(), Open(), wedge, wedge, Open(), Wall()])
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05, 0.01, 0.01)])
        sim.rest(0.2)
        sim.compress(0.4)

        platen = sim.platens[0]
        weight = _sphere_mass(material, 0.01) * 9.81
        assert platen.edges == (3, 4)
        assert_allclose(platen.direction, [0.0, -1.0], atol=1e-12)
        assert_allclose(platen.ends[0], platen.starts[1], atol=1e-15)
        assert_allclose(platen.reaction, load, rtol=1e-2)
        # a double contact at the tip would halve the overlap of the top contact
        expected = _static_overlap(material, 0.01, load) + _static_overlap(material, 0.01, load + weight) - \
            _static_overlap(material, 0.01, weight)
        assert_allclose(platen.displacement, expected, rtol=2e-2)
        assert_allclose(sim.pos[0, 0], 0.05, atol=1e-9)
        assert sim.frames[-1].platens[0].shape == (2, 2, 2)

    def test_rolling_friction_holds_particle_on_gentle_incline(self) -> None:
        """A particle must stay put on an incline with tan(theta) below the rolling friction coefficient."""
        sim = _sphere_on_incline(slope=0.32, rolling_friction=0.4)
        start = sim.pos[0].copy()
        sim.rest(0.5)
        # only the elastic deformation of the springs, some 40 um here, may move it
        assert np.hypot(*(sim.pos[0] - start)) < 1.0e-4
        assert np.hypot(*sim.vel[0]) < 1.0e-4

    @pytest.mark.parametrize("rolling_friction", [0.0, 0.2, 0.4])
    def test_particle_rolls_down_steep_incline(self, rolling_friction: float) -> None:
        """Above the rolling limit a sphere must roll without slipping at a = g (sin - mu_r cos) / 1.4."""
        slope = 0.48
        sim = _sphere_on_incline(slope=slope, rolling_friction=rolling_friction)
        down = np.array([1.0, -slope]) / math.hypot(1.0, slope)
        sim.rest(0.1)
        early_time, early_speed = sim.time, float(sim.vel[0] @ down)
        sim.rest(0.2)
        acceleration = (float(sim.vel[0] @ down) - early_speed) / (sim.time - early_time)
        theta = math.atan(slope)
        expected = 9.81 * (math.sin(theta) - rolling_friction * math.cos(theta)) / 1.4
        assert_allclose(acceleration, expected, rtol=0.02)
        # rolling without slipping: the contact point is at rest
        assert_allclose(sim.omega[0] * sim.radius[0], -float(sim.vel[0] @ down), rtol=0.02)

    def test_rain_fills_non_rectangular_domain(self) -> None:
        """Raining must fill a trapezoid with non-overlapping particles that all stay inside it."""
        geometry = Geometry([(0.0, 0.0), (0.06, 0.0), (0.08, 0.06), (-0.02, 0.06)], [Wall(), Wall(), Open(), Wall()])
        sim = DEMSimulation(geometry, _material(), radius_range=(0.003, 0.004))
        rain = RainSettings(batch_size=10, steps_between=150, patience=3, settle_steps=2000, seed=1)
        n = sim.rain_until_full(rain)
        assert n == sim.n and n > 40
        assert np.all(geometry.contains(sim.pos))
        assert sim.bed_top() < 0.06
        delta = sim.pos[:, None, :] - sim.pos[None, :, :]
        gap = np.hypot(delta[..., 0], delta[..., 1]) - (sim.radius[:, None] + sim.radius[None, :])
        np.fill_diagonal(gap, np.inf)
        assert gap.min() > -0.05 * sim.radius_range[0]
        assert len(sim.frames) > 0


class TestPrescribedPlaten:
    """Tests of displacement-controlled platens against closed-form results."""

    def test_loaded_and_prescribed_footings_reach_their_equilibrium(self) -> None:
        """A footing pressed in by the Hertz indentation of a load must feel that load, beside a loaded footing."""
        material = _material()
        load = 2.0
        indentation = _seated_indentation(material, 0.01, load)
        loaded = LoadedPlaten(load=Heaviside(load, rise_time=0.05), mass=0.2)
        prescribed = PrescribedPlaten(displacement=Heaviside(indentation, rise_time=0.05))
        vertices = [(0.0, 0.0), (0.12, 0.0), (0.12, 0.1), (0.1, 0.1), (0.08, 0.1), (0.04, 0.1), (0.02, 0.1), (0.0, 0.1)]
        geometry = Geometry(vertices, [Wall(), Wall(), Open(), prescribed, Open(), loaded, Open(), Wall()])
        sim = DEMSimulation(geometry, material, particles=[Particle(0.09, 0.01, 0.01), Particle(0.03, 0.01, 0.01)])
        sim.rest(0.2)
        sim.compress(0.4)

        driven, pushed = sim.platens
        assert driven.prescribed and not pushed.prescribed
        assert_allclose(driven.displacement, indentation, rtol=1e-12)
        # the driven footing does not oscillate about its equilibrium, so it reaches it far more closely
        assert_allclose(driven.reaction, load, rtol=1e-9)
        assert_allclose(0.01 - sim.pos[0, 1],
                        _static_overlap(material, 0.01, load + _sphere_mass(material, 0.01) * 9.81),
                        rtol=1e-9)
        assert_allclose(pushed.reaction, load, rtol=1e-2)
        curve = sim.load_curves[0].as_array()
        assert_allclose(curve[:, 1], curve[:, 3], rtol=0.0, atol=0.0)

    def test_platen_follows_its_displacement_history(self) -> None:
        """A sinusoidally driven platen must be where its history puts it, whatever the particle pushes back with."""
        material = _material()
        amplitude, frequency = 0.5 * _seated_indentation(material, 0.01, 2.0), 10.0
        history = Sinusoidal(amplitude, frequency, mean=amplitude, phase=-0.5 * math.pi)
        geometry = Geometry.rectangle(0.1, 0.1, top=PrescribedPlaten(displacement=history))
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05, 0.01, 0.01)])
        sim.rest(0.2)
        sim.compress(0.25, record_every=10)

        time, applied, displacement, reaction = sim.load_curves[0].as_array().T
        assert_allclose(displacement, [history(t) for t in time], rtol=0.0, atol=1e-15)
        assert_allclose(applied, reaction, rtol=0.0, atol=0.0)
        assert np.all(reaction >= 0.0) and np.max(reaction) > 1.0
        # the velocity of the last step is its mean, which is the velocity half way through it to second order
        platen = sim.platens[0]
        omega = 2.0 * math.pi * frequency
        expected = amplitude * omega * math.sin(omega * (sim.load_time - 0.5 * sim.dt))
        assert_allclose(platen.velocity, expected, rtol=0.0, atol=1e-6 * amplitude * omega)

    def test_platen_drawn_back_loses_contact(self) -> None:
        """A platen pressed in and then drawn back past first contact must leave the particle and feel no reaction."""
        material = _material()
        load = 2.0
        indentation = _seated_indentation(material, 0.01, load)
        history = TimeHistory([0.0, 0.05, 0.15, 0.2], [0.0, indentation, indentation, -indentation])
        geometry = Geometry.rectangle(0.1, 0.1, top=PrescribedPlaten(displacement=history))
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05, 0.01, 0.01)])
        sim.rest(0.2)
        sim.compress(0.4)

        time, _, _, reaction = sim.load_curves[0].as_array().T
        held = (time > 0.14) & (time < 0.15)
        assert_allclose(reaction[held], load, rtol=1e-3)
        assert_allclose(reaction[time > 0.2], 0.0, atol=0.0)
        weight = _sphere_mass(material, 0.01) * 9.81
        assert_allclose(0.01 - sim.pos[0, 1], _static_overlap(material, 0.01, weight), rtol=1e-3)

    def test_lid_reproduces_the_hertz_reaction_in_3d(self) -> None:
        """A lid pressed in by the Hertz indentation of a load must feel that load in 3D as well."""
        material = _material()
        load = 2.0
        indentation = _seated_indentation(material, 0.01, load)
        lid = PrescribedPlaten(displacement=Heaviside(indentation, rise_time=0.05))
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1, top=lid),
                            material,
                            particles=[Particle3D(0.05, 0.01, 0.05, 0.01)])
        sim.rest(0.3)
        sim.compress(0.4)

        platen = sim.platens[0]
        assert_allclose(platen.displacement, indentation, rtol=1e-12)
        assert_allclose(platen.reaction, load, rtol=1e-9)
        assert_allclose(sim.face_forces[2], [0.0, load, 0.0], rtol=1e-9, atol=1e-12)


def _rotation(axis: Tuple[float, float, float], angle: float) -> npt.NDArray[np.float64]:
    """Return the matrix of a rotation by an angle about an axis (Rodrigues' formula)."""
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    skew = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    return np.asarray(np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew)


def _sphere_on_oblique_incline(slope: float, rolling_friction: float) -> Tuple[DEMSimulation, npt.NDArray[np.float64]]:
    """Return a 10 mm particle resting on a floor tilted about the horizontal axis (1, 0, 1), and the normal."""
    material = _material(friction=0.8, rolling_friction=rolling_friction)
    rotation = _rotation((1.0, 0.0, 1.0), math.atan(slope))
    box = Geometry3D.box(0.4, 0.2, 0.4, origin=(-0.2, 0.0, -0.2))
    geometry = Geometry3D((box.vertices @ rotation.T).tolist(), box.faces, box.boundary_conditions)
    normal = rotation @ np.array([0.0, 1.0, 0.0])
    radius = 0.01
    weight = _sphere_mass(material, radius) * 9.81
    # the tilted floor passes through the origin
    centre = (radius - _static_overlap(material, radius, weight * normal[1])) * normal
    particle = Particle3D(float(centre[0]), float(centre[1]), float(centre[2]), radius)
    return DEMSimulation(geometry, material, particles=[particle]), normal


def _split_floor_box() -> Geometry3D:
    """Return a 0.1 m cube whose floor is made of four square faces that meet at its centre."""
    vertices: List[Tuple[float, float, float]] = []

    def index(point: Tuple[float, float, float]) -> int:
        if point not in vertices:
            vertices.append(point)
        return vertices.index(point)

    grid = (0.0, 0.05, 0.1)
    faces: List[Tuple[int, ...]] = []
    for i in range(2):
        for k in range(2):
            corners = [(grid[i], grid[k]), (grid[i + 1], grid[k]), (grid[i + 1], grid[k + 1]), (grid[i], grid[k + 1])]
            faces.append(tuple(index((x, 0.0, z)) for x, z in corners))
    top = [index((0.0, 0.1, 0.0)), index((0.1, 0.1, 0.0)), index((0.1, 0.1, 0.1)), index((0.0, 0.1, 0.1))]
    faces.append((top[1], top[0], top[3], top[2]))
    # the sides list the vertex in the middle of their bottom edge, which the floor faces share
    sides = [[(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.1, 0.1, 0.0), (0.1, 0.0, 0.0), (0.05, 0.0, 0.0)],
             [(0.0, 0.0, 0.1), (0.05, 0.0, 0.1), (0.1, 0.0, 0.1), (0.1, 0.1, 0.1), (0.0, 0.1, 0.1)],
             [(0.0, 0.0, 0.0), (0.0, 0.0, 0.05), (0.0, 0.0, 0.1), (0.0, 0.1, 0.1), (0.0, 0.1, 0.0)],
             [(0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.1, 0.1, 0.1), (0.1, 0.0, 0.1), (0.1, 0.0, 0.05)]]
    faces.extend(tuple(index(point) for point in side) for side in sides)
    conditions: List[BoundaryCondition] = [Wall(), Wall(), Wall(), Wall(), Open(), Wall(), Wall(), Wall(), Wall()]
    return Geometry3D(vertices, faces, conditions)


class TestDEMSimulation3DSetup:
    """Tests of the input checks and the state of a 3D DEMSimulation."""

    def test_state_has_three_dimensional_shapes(self) -> None:
        """Positions, rotations and facet forces must be 3D vectors."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), particles=[Particle3D(0.05, 0.05, 0.05, 0.01)])
        sim.compute_forces()
        assert sim.dimension == 3
        assert sim.pos.shape == sim.vel.shape == sim.force.shape == (1, 3)
        assert sim.omega.shape == sim.torque.shape == (1, 3)
        assert sim.edge_forces.shape == (6, 3) and sim.face_forces is sim.edge_forces
        assert_allclose(sim.force, [[0.0, -_sphere_mass(_material(), 0.01) * 9.81, 0.0]])
        assert_allclose(sim.excess_pore_pressure(np.zeros((2, 3))), [0.0, 0.0])

    def test_empty_domain(self) -> None:
        """An empty domain has its bed at the floor and nothing to move."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.2, 0.1, origin=(0.0, -0.05, 0.0)),
                            _material(),
                            radius_range=(0.01, 0.01))
        assert_allclose(sim.bed_top(), -0.05)
        sim.rest(0.001)
        assert sim.n == 0 and sim.mean_kinetic_energy() == 0.0

    def test_particles_of_the_other_dimension_are_rejected(self) -> None:
        """A 2D particle cannot enter a 3D domain, nor a 3D particle a 2D one."""
        with pytest.raises(ValueError,
                           match=re.escape("Particle 0 is two-dimensional, but the geometry is three-dimensional.")):
            DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), particles=[Particle(0.05, 0.05, 0.01)])
        with pytest.raises(ValueError,
                           match=re.escape("Particle 1 is three-dimensional, but the geometry is two-dimensional.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1),
                          _material(),
                          particles=[Particle(0.05, 0.05, 0.01),
                                     Particle3D(0.05, 0.05, 0.05, 0.01)])

    def test_particle_outside_domain_is_rejected(self) -> None:
        """Initial particles must lie inside the polyhedron."""
        with pytest.raises(ValueError, match=re.escape("Particle 1 at (0.05, 0.05, 0.2) lies outside the domain.")):
            DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                          _material(),
                          particles=[Particle3D(0.05, 0.05, 0.05, 0.01),
                                     Particle3D(0.05, 0.05, 0.2, 0.01)])

    def test_loading_without_platen_face_is_rejected(self) -> None:
        """Loading and seating need a LoadedPlaten or PrescribedPlaten face."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        message = "The geometry has no LoadedPlaten or PrescribedPlaten face to apply a load with."
        with pytest.raises(ValueError, match=re.escape(message)):
            sim.compress(0.1)
        with pytest.raises(ValueError,
                           match=re.escape("The geometry has no LoadedPlaten or PrescribedPlaten face to seat.")):
            sim.seat_platens()


class TestDEMSimulation3DPhysics:
    """Tests of the simulated 3D physics against closed-form results and against the 2D model."""

    def test_particle_rests_at_hertz_overlap(self) -> None:
        """A particle resting on a floor face must settle at the static Hertz overlap, the floor carrying it."""
        material = _material()
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), material, particles=[Particle3D(0.05, 0.01, 0.04, 0.01)])
        sim.rest(0.5)
        weight = _sphere_mass(material, 0.01) * 9.81
        assert_allclose(0.01 - sim.pos[0, 1], _static_overlap(material, 0.01, weight), rtol=1e-3)
        assert_allclose(sim.face_forces[0], [0.0, -weight, 0.0], rtol=1e-3, atol=1e-12)
        assert_allclose(sim.face_forces[1:], 0.0, atol=1e-15)

    @pytest.mark.parametrize("shift", [(0.0, 0.0), (1.0e-4, 0.0), (0.0, -1.0e-4), (1.0e-4, 1.0e-4), (-1.0e-4, 2.0e-4)])
    def test_shared_edges_and_vertex_give_one_contact(self, shift: Tuple[float, float]) -> None:
        """A particle on or next to the edges and the vertex shared by four floor faces must feel one contact."""
        material = _material()
        x, z = 0.05 + shift[0], 0.05 + shift[1]
        sim = DEMSimulation(_split_floor_box(), material, particles=[Particle3D(x, 0.01, z, 0.01)])
        sim.rest(0.5)
        expected = _static_overlap(material, 0.01, _sphere_mass(material, 0.01) * 9.81)
        assert_allclose(0.01 - sim.pos[0, 1], expected, rtol=1e-3)
        # a spurious second contact at an edge or the vertex would push the particle sideways
        assert_allclose(sim.pos[0, [0, 2]], [x, z], atol=1e-9)

    def test_pyramid_tip_gives_one_contact(self) -> None:
        """A pyramidal body of four faces must touch a particle once at its tip and balance its load."""
        material = _material()
        load = 2.0
        pyramid = LoadedPlaten(load=Heaviside(load, rise_time=0.05), mass=0.2)
        vertices = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.0, 0.1), (0.0, 0.0, 0.1), (0.0, 0.1, 0.0),
                    (0.1, 0.1, 0.0), (0.1, 0.1, 0.1), (0.0, 0.1, 0.1), (0.05, 0.08, 0.05)]
        faces = [(0, 1, 2, 3), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0), (4, 8, 5), (5, 8, 6), (6, 8, 7),
                 (7, 8, 4)]
        conditions: List[BoundaryCondition] = [
            Wall(), Wall(), Wall(), Wall(), Wall(), pyramid, pyramid, pyramid, pyramid
        ]
        sim = DEMSimulation(Geometry3D(vertices, faces, conditions),
                            material,
                            particles=[Particle3D(0.05, 0.01, 0.05, 0.01)])
        sim.rest(0.2)
        sim.compress(0.4)

        platen = sim.platens[0]
        weight = _sphere_mass(material, 0.01) * 9.81
        assert platen.faces == (5, 6, 7, 8)
        assert_allclose(platen.direction, [0.0, -1.0, 0.0], atol=1e-12)
        assert_allclose(platen.reaction, load, rtol=1e-2)
        # a contact with each of the four faces at the tip would share the load among four overlaps
        expected = _static_overlap(material, 0.01, load) + _static_overlap(material, 0.01, load + weight) - \
            _static_overlap(material, 0.01, weight)
        assert_allclose(platen.displacement, expected, rtol=2e-2)
        assert_allclose(sim.pos[0, [0, 2]], [0.05, 0.05], atol=1e-9)
        assert sim.frames[-1].platens[0].shape == (4, 3, 3)

    def test_collision_conserves_momentum_and_dissipates_energy(self) -> None:
        """An oblique 3D collision must conserve linear momentum exactly and spin both particles alike."""
        material = _material(restitution=0.5)
        particles = [Particle3D(0.47, 0.5, 0.5, 0.01, vx=0.5, vz=0.1), Particle3D(0.53, 0.505, 0.507, 0.015, vx=-0.5)]
        sim = DEMSimulation(Geometry3D.box(1.0, 1.0, 1.0),
                            material,
                            particles=particles,
                            settings=GlobalSettings(gravity=0.0))
        momentum = (sim.mass[:, None] * sim.vel).sum(axis=0)
        energy = sim.mean_kinetic_energy()
        sim.rest(0.08)
        assert_allclose((sim.mass[:, None] * sim.vel).sum(axis=0), momentum, atol=1e-14)
        assert sim.mean_kinetic_energy() < 0.9 * energy
        # friction at the contact spins the two particles about parallel axes, like meshing gears
        assert np.all(np.linalg.norm(sim.omega, axis=1) > 1e-3)
        assert_allclose(np.cross(sim.omega[0], sim.omega[1]), 0.0, atol=1e-9)
        assert float(sim.omega[0] @ sim.omega[1]) > 0.0

    @pytest.mark.parametrize("rolling_friction", [0.0, 0.2, 0.4])
    def test_particle_rolls_down_oblique_incline(self, rolling_friction: float) -> None:
        """On a floor tilted about an oblique axis a sphere must roll straight down at g (sin - mu_r cos) / 1.4."""
        slope = 0.48
        sim, normal = _sphere_on_oblique_incline(slope, rolling_friction)
        down = np.array([0.0, -1.0, 0.0]) + normal[1] * normal
        down /= np.linalg.norm(down)
        sim.rest(0.1)
        early_time, early_speed = sim.time, float(sim.vel[0] @ down)
        sim.rest(0.2)
        acceleration = (float(sim.vel[0] @ down) - early_speed) / (sim.time - early_time)
        theta = math.atan(slope)
        assert_allclose(acceleration, 9.81 * (math.sin(theta) - rolling_friction * math.cos(theta)) / 1.4, rtol=1e-3)
        # it rolls without slipping, v + omega x (-R n) = 0, and neither drifts across the incline nor twists
        assert_allclose(np.cross(sim.omega[0], sim.radius[0] * normal), sim.vel[0], atol=1e-6)
        assert abs(float(sim.vel[0] @ np.cross(normal, down))) < 1e-9
        assert abs(float(sim.omega[0] @ normal)) < 1e-9

    def test_rolling_friction_holds_particle_on_gentle_oblique_incline(self) -> None:
        """A sphere must stay put on an incline with tan(theta) below the rolling friction coefficient."""
        sim, _ = _sphere_on_oblique_incline(slope=0.32, rolling_friction=0.4)
        start = sim.pos[0].copy()
        sim.rest(0.5)
        assert np.linalg.norm(sim.pos[0] - start) < 1.0e-4
        assert np.linalg.norm(sim.vel[0]) < 1.0e-4

    def test_planar_motion_reproduces_the_2d_model(self) -> None:
        """Particles moving in a plane of an extruded domain must follow the 2D simulation of its section."""
        material = _material(restitution=0.5, rolling_friction=0.3)
        footing = LoadedPlaten(load=Heaviside(2.0, rise_time=0.05), mass=0.2)
        vertices = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.06, 0.1), (0.04, 0.1), (0.0, 0.1)]
        planar = Geometry(vertices, [Wall(), Wall(), Open(), footing, Open(), Wall()])
        sims = [
            DEMSimulation(planar,
                          material,
                          particles=[Particle(0.05, 0.01, 0.01, vx=0.05),
                                     Particle(0.02, 0.012, 0.01)]),
            DEMSimulation(Geometry3D.extrude(planar, 0.1),
                          material,
                          particles=[Particle3D(0.05, 0.01, 0.05, 0.01, vx=0.05),
                                     Particle3D(0.02, 0.012, 0.05, 0.01)])
        ]
        for sim in sims:
            sim.rest(0.1)
            sim.compress(0.2)
        flat, solid = sims
        assert_allclose(solid.pos[:, :2], flat.pos, atol=1e-12)
        assert_allclose(solid.pos[:, 2], 0.05, atol=1e-15)
        assert_allclose(solid.omega[:, 2], flat.omega, rtol=1e-9, atol=1e-9)
        assert_allclose(solid.omega[:, :2], 0.0, atol=1e-15)
        assert_allclose(solid.platens[0].offset, flat.platens[0].offset, rtol=1e-12)
        assert_allclose(solid.face_forces[:6, :2], flat.edge_forces, rtol=1e-9, atol=1e-12)
        assert solid.load_curves[0].faces == flat.load_curves[0].edges == (3, )

    def test_planar_collision_with_rolling_resistance_reproduces_the_2d_model(self) -> None:
        """An oblique collision in a plane, with rolling resistance, must spin the particles as in 2D."""
        material = _material(restitution=0.5, rolling_friction=0.3)
        flat = DEMSimulation(
            Geometry.rectangle(1.0, 1.0),
            material,
            particles=[Particle(0.47, 0.5, 0.01, vx=0.5),
                       Particle(0.53, 0.505, 0.015, vx=-0.5, vy=0.1)],
            settings=GlobalSettings(gravity=0.0))
        solid = DEMSimulation(
            Geometry3D.extrude(Geometry.rectangle(1.0, 1.0), 1.0),
            material,
            particles=[Particle3D(0.47, 0.5, 0.5, 0.01, vx=0.5),
                       Particle3D(0.53, 0.505, 0.5, 0.015, vx=-0.5, vy=0.1)],
            settings=GlobalSettings(gravity=0.0))
        flat.rest(0.08)
        solid.rest(0.08)
        assert np.all(np.abs(flat.omega) > 1.0)
        assert_allclose(solid.pos[:, :2], flat.pos, atol=1e-14)
        assert_allclose(solid.omega[:, 2], flat.omega, rtol=1e-12)
        assert_allclose(solid.omega[:, :2], 0.0, atol=1e-15)

    def test_seated_platen_touches_bed_and_stays_fixed(self) -> None:
        """Seating must lower the lid onto the top particle and hold it there without starting the load."""
        geometry = Geometry3D.box(0.1, 0.1, 0.1, top=LoadedPlaten(load=Heaviside(2.0), mass=0.2))
        sim = DEMSimulation(geometry,
                            _material(),
                            particles=[Particle3D(0.05, 0.01, 0.05, 0.01),
                                       Particle3D(0.05, 0.03, 0.05, 0.01)])
        sim.rest(0.2)
        sim.seat_platens()
        platen = sim.platens[0]
        assert_allclose(platen.vertices[0, :, 1], float(np.max(sim.pos[:, 1] + sim.radius)), atol=1e-12)
        offset = platen.offset
        sim.rest(0.05)
        assert not platen.active
        assert_allclose(platen.offset, offset, atol=1e-15)
        assert sim.frames[-1].platens[0].shape == (1, 4, 3)

    def test_open_face_lets_particles_out(self) -> None:
        """Particles falling through an open floor must be removed once they are clear of the domain."""
        geometry = Geometry3D.box(0.1, 0.1, 0.1, bottom=Open())
        particles = [Particle3D(0.05, 0.05, 0.05, 0.005), Particle3D(0.02, 0.05, 0.07, 0.005)]
        sim = DEMSimulation(geometry, _material(), particles=particles, settings=GlobalSettings(escape_margin=1.0))
        sim.rest(0.2)
        assert sim.n == 0
        assert sim.n_removed == 2

    def test_lattice_packing_deposits_and_comes_to_rest(self) -> None:
        """Particles released from a loose lattice must settle on the floor until the packing is at rest."""
        geometry = Geometry3D.box(0.05, 0.1, 0.05)
        sim = DEMSimulation(geometry, _material(), particles=generate_lattice_particles(geometry, 20, (0.004, 0.005)))
        sim.deposit(20000, min_steps=200, record_every=200)
        assert sim.mean_kinetic_energy() < sim.settings.quiet_tolerance
        assert sim.steps_taken < 20000
        assert sim.bed_top() < 0.03 and np.all(geometry.contains(sim.pos))
        assert sim.frames[0].phase == "deposition"

    def test_rain_fills_polyhedron(self) -> None:
        """Raining must fill an extruded trapezoid with non-overlapping particles that all stay inside it."""
        trapezoid = Geometry([(0.0, 0.0), (0.05, 0.0), (0.07, 0.05), (-0.02, 0.05)], [Wall(), Wall(), Open(), Wall()])
        geometry = Geometry3D.extrude(trapezoid, 0.04)
        sim = DEMSimulation(geometry, _material(), radius_range=(0.004, 0.005))
        rain = RainSettings(batch_size=15, steps_between=150, patience=3, settle_steps=1500, seed=1)
        n = sim.rain_until_full(rain)
        assert n == sim.n and n > 50
        assert np.all(geometry.contains(sim.pos))
        assert sim.bed_top() < 0.05
        gap = np.linalg.norm(sim.pos[:, None, :] - sim.pos[None, :, :], axis=2) - \
            (sim.radius[:, None] + sim.radius[None, :])
        np.fill_diagonal(gap, np.inf)
        assert gap.min() > -0.05 * sim.radius_range[0]


class _StepCounter:
    """Observer that counts the steps a simulation takes; a class, so that a simulation pickles with it."""

    def __init__(self) -> None:
        """Start from zero."""
        self.count = 0

    def __call__(self, sim: DEMSimulation) -> None:
        """Count one step."""
        self.count += 1


class TestContacts:
    """Tests of the contacts of the current state, and of the observers of the recorded frames."""

    def test_particle_on_floor_touches_it_at_the_bottom(self) -> None:
        """A particle resting on the floor must touch the floor edge below its centre and carry its weight there."""
        material = _material()
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), material, particles=[Particle(0.05, 0.01, 0.01)])
        sim.rest(0.5)
        pairs, walls = sim.contacts()
        assert pairs.first.size == 0
        assert walls.first.tolist() == [0] and walls.second.tolist() == [0]
        assert_allclose(walls.normal, [[0.0, 1.0]], atol=1e-12)
        assert_allclose(walls.overlap, 0.01 - sim.pos[:, 1], rtol=1e-12)
        assert_allclose(walls.point, [[0.05, 0.5 * (sim.pos[0, 1] - 0.01)]], atol=1e-12)
        assert_allclose(walls.normal_force, _sphere_mass(material, 0.01) * 9.81, rtol=1e-3)
        assert_allclose(walls.force, [[0.0, walls.normal_force[0]]], atol=1e-9)

    def test_pair_contact_lies_in_the_middle_of_the_overlap(self) -> None:
        """Two overlapping particles must touch in the middle of their overlap, with the normal pointing to i."""
        sim = DEMSimulation(Geometry.rectangle(1.0, 1.0),
                            _material(),
                            particles=[Particle(0.5, 0.5, 0.01), Particle(0.519, 0.5, 0.012)],
                            settings=GlobalSettings(gravity=0.0))
        pairs, walls = sim.contacts()
        assert walls.first.size == 0
        assert pairs.first.tolist() == [0] and pairs.second.tolist() == [1]
        assert_allclose(pairs.normal, [[-1.0, 0.0]], atol=1e-12)
        assert_allclose(pairs.overlap, [0.003], rtol=1e-9)
        # the overlap spans x = 0.507 to 0.51
        assert_allclose(pairs.point, [[0.5085, 0.5]], rtol=1e-12)
        assert pairs.normal_force[0] > 0.0
        assert_allclose(pairs.force[0] / pairs.normal_force[0], [-1.0, 0.0], atol=1e-12)

    def test_contacts_with_facets_name_the_facet_of_the_geometry(self) -> None:
        """Contacts with a facet must give its index in the geometry, which open facets also count."""
        geometry = Geometry3D.box(0.1, 0.1, 0.1, bottom=Open(), right=Wall())
        sim = DEMSimulation(geometry, _material(), particles=[Particle3D(0.095, 0.05, 0.05, 0.01)])
        _, walls = sim.contacts()
        assert walls.second.tolist() == [1]
        assert_allclose(walls.normal, [[-1.0, 0.0, 0.0]], atol=1e-12)

    def test_contacts_do_not_change_the_run(self) -> None:
        """Inspecting the contacts at every step must leave the motion of the particles exactly as it was."""
        runs = []
        for inspect in (False, True):
            geometry = Geometry3D.box(0.05, 0.1, 0.05)
            sim = DEMSimulation(geometry,
                                _material(restitution=0.5, rolling_friction=0.2),
                                particles=generate_lattice_particles(geometry, 20, (0.004, 0.005)))
            for _ in range(600):
                sim.step()
                if inspect:
                    sim.contacts()
            runs.append(sim)
        plain, inspected = runs
        assert inspected.contacts()[0].first.size > 0
        assert_allclose(inspected.pos, plain.pos, rtol=0.0, atol=1e-15)
        assert_allclose(inspected.omega, plain.omega, rtol=0.0, atol=1e-12)

    def test_empty_domain_has_no_contacts(self) -> None:
        """An empty domain must return empty contact lists of the right shapes."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), radius_range=(0.005, 0.01))
        for contacts in sim.contacts():
            assert contacts.first.shape == contacts.second.shape == contacts.normal_force.shape == (0, )
            assert contacts.point.shape == contacts.normal.shape == contacts.tangential_force.shape == (0, 3)

    def test_observers_see_every_step(self) -> None:
        """Every observer must be called once per time step, and keep counting in a pickled copy."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[Particle(0.05, 0.05, 0.01)])
        counters = [_StepCounter(), _StepCounter()]
        for counter in counters:
            sim.add_observer(counter)
        sim.rest(0.01, record_every=10)
        assert sim.steps_taken > 10
        assert [counter.count for counter in counters] == [sim.steps_taken] * 2
        copy = pickle.loads(pickle.dumps(sim))
        copy.rest(0.01, record_every=10)
        assert [counter.count for counter in counters] == [sim.steps_taken] * 2
        assert copy._DEMSimulation__observers[0].count == copy.steps_taken

    def test_removed_observer_is_no_longer_called(self) -> None:
        """An observer that is removed must not be called again, while the others still are."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[Particle(0.05, 0.05, 0.01)])
        kept, removed = _StepCounter(), _StepCounter()
        sim.add_observer(kept)
        sim.add_observer(removed)
        sim.step()
        sim.remove_observer(removed)
        sim.step()
        assert (kept.count, removed.count) == (2, 1)

    def test_removing_a_stranger_is_rejected(self) -> None:
        """Removing a function that does not observe the simulation must be rejected."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[Particle(0.05, 0.05, 0.01)])
        stranger = _StepCounter()
        with pytest.raises(ValueError, match=re.escape(f"{stranger!r} does not observe this simulation.")):
            sim.remove_observer(stranger)

    def test_simulation_pickled_before_observers_existed_still_steps(self) -> None:
        """A simulation whose state has no observers, as one pickled by an earlier version, must still step."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[Particle(0.05, 0.05, 0.01)])
        sim.add_observer(_StepCounter())
        state = {k: v for k, v in sim.__dict__.items() if not k.endswith("__observers")}
        old = DEMSimulation.__new__(DEMSimulation)
        old.__dict__.update(state)
        old.rest(0.01, record_every=10)
        assert old.steps_taken > 0 and len(old.frames) > 0
