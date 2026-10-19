"""Tests of the integrator, the settings and the DEM simulation engine."""

import math
import re

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from DEMcookies import (DEMSimulation, Geometry, GlobalSettings, Heaviside, LoadedPlaten, MaterialProperties, Open,
                        Particle, RainSettings, Wall)
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
        """Loading needs a LoadedPlaten edge."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The geometry has no LoadedPlaten edge to apply a load with.")):
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
        """Seating needs a LoadedPlaten edge."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The geometry has no LoadedPlaten edge to seat.")):
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
