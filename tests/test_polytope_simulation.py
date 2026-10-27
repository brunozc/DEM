"""Tests of simulations with polygonal and polyhedral particles."""

import math
import pickle
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies import (DEMSimulation, Geometry, Geometry3D, GlobalSettings, Heaviside, LoadedPlaten,
                        MaterialProperties, Open, Particle, Particle3D, RainSettings, Wall)
from DEMcookies.polytopes import apply_inertia, rotation_matrices
from DEMcookies.shapes import ConvexPolygon, ConvexPolyhedron

SQUARE = ConvexPolygon.rectangle(1.0, 1.0)
CUBE = ConvexPolyhedron.box(1.0, 1.0, 1.0)


def _material(restitution: float = 0.3, friction: float = 0.5) -> MaterialProperties:
    """Return the reference material."""
    return MaterialProperties(density=2500.0,
                              youngs_modulus=5.0e6,
                              poisson_ratio=0.25,
                              restitution=restitution,
                              friction=friction)


def _e_star(material: MaterialProperties) -> float:
    """Return the effective Young's modulus of two bodies of the material [Pa]."""
    return material.youngs_modulus / (2.0 * (1.0 - material.poisson_ratio**2))


def _weight(material: MaterialProperties, radius: float) -> float:
    """Return the weight of a particle of a radius [N]."""
    return material.density * 4.0 / 3.0 * math.pi * radius**3 * 9.81


class TestPolygons:
    """Tests of 2D simulations with polygons."""

    def test_square_rests_at_the_depth_of_the_contact_law(self) -> None:
        """A square on the floor must carry its weight at the depth delta = W / (2 E* a), with a = L / (2 sqrt 2)."""
        material = _material()
        radius = 0.01
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            material,
                            particles=[Particle(0.05, 0.02, radius, shape=SQUARE)])
        sim.rest(0.3)
        _, walls = sim.contacts()
        side = radius * math.sqrt(math.pi)
        depth = _weight(material, radius) / (2.0 * _e_star(material) * side / (2.0 * math.sqrt(2.0)))
        assert_allclose(walls.normal_force, [_weight(material, radius)], rtol=1e-6)
        assert_allclose(walls.overlap, [depth], rtol=1e-6)
        assert_allclose(sim.pos[0, 1], 0.5 * side - depth, rtol=1e-6)
        assert_allclose(walls.point, [[0.05, -0.5 * depth]], atol=1e-9)

    def test_floor_split_into_two_edges_carries_the_same_square(self) -> None:
        """A square straddling the joint of two collinear floor edges must rest as on a single edge."""
        material = _material()
        radius = 0.01
        geometry = Geometry([(0, 0), (0.047, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)],
                            [Wall(), Wall(), Wall(), Open(), Wall()])
        sim = DEMSimulation(geometry, material, particles=[Particle(0.05, 0.02, radius, shape=SQUARE)])
        sim.rest(0.3)
        _, walls = sim.contacts()
        side = radius * math.sqrt(math.pi)
        depth = _weight(material, radius) / (2.0 * _e_star(material) * side / (2.0 * math.sqrt(2.0)))
        assert sorted(walls.second) == [0, 1]
        assert_allclose(walls.normal_force.sum(), _weight(material, radius), rtol=1e-6)
        assert_allclose(walls.overlap, [depth, depth], rtol=1e-6)

    def test_tilted_square_topples_onto_a_side(self) -> None:
        """A square dropped on a corner must topple and come to rest on one of its sides."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            _material(),
                            particles=[Particle(0.05, 0.03, 0.01, shape=SQUARE, angle=0.3)])
        sim.rest(1.0)
        quarter = math.degrees(sim.orientation[0]) % 90.0
        assert min(quarter, 90.0 - quarter) < 0.01
        assert sim.mean_kinetic_energy() < 1e-12

    def test_disc_rests_on_a_square_at_the_hertz_overlap(self) -> None:
        """A disc on top of a square must press on it with its weight, at the overlap of a sphere on a plane."""
        material = _material()
        side = 0.04
        square = Particle(0.05, 0.5 * side, side / math.sqrt(math.pi), shape=SQUARE)
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            material,
                            particles=[square, Particle(0.05, side + 0.006, 0.005)])
        sim.rest(0.4)
        pairs, walls = sim.contacts()
        weight = _weight(material, 0.005)
        overlap = (3.0 * weight / (4.0 * _e_star(material) * math.sqrt(0.005)))**(2.0 / 3.0)
        assert_allclose(pairs.normal_force, [weight], rtol=1e-5)
        assert_allclose(pairs.overlap, [overlap], rtol=1e-5)
        assert_allclose(walls.normal_force.sum(), weight + _weight(material, square.radius), rtol=1e-5)

    def test_collision_conserves_momentum(self) -> None:
        """Two polygons that collide in free space must keep their total linear and angular momentum."""
        shape = ConvexPolygon.random(6, seed=2, aspect=1.4)
        # a domain large enough that neither particle reaches its walls
        particles = [
            Particle(0.48, 0.5, 0.01, vx=0.4, vy=0.05, shape=shape, angle=0.2),
            Particle(0.52, 0.502, 0.012, vx=-0.3, shape=shape, angle=1.3)
        ]
        sim = DEMSimulation(Geometry.rectangle(1.0, 1.0),
                            _material(),
                            particles=particles,
                            settings=GlobalSettings(gravity=0.0))
        sim.omega[:] = [3.0, -2.0]

        def momenta() -> tuple:
            linear = (sim.mass[:, None] * sim.vel).sum(axis=0)
            orbital = sim.mass * (sim.pos[:, 0] * sim.vel[:, 1] - sim.pos[:, 1] * sim.vel[:, 0])
            return linear, float(orbital.sum() + (sim.inertia * sim.omega).sum())

        linear, angular = momenta()
        touched = False
        for _ in range(1500):
            sim.step()
            touched |= bool(sim.contacts()[0].first.size)
        assert touched
        assert_allclose(momenta()[0], linear, atol=1e-15)
        assert_allclose(momenta()[1], angular, rtol=1e-6)

    def test_polygons_are_rained_and_settle(self) -> None:
        """Rained polygons and discs must fill a box, stay in it and come to rest."""
        shapes = [None, ConvexPolygon.regular(5), ConvexPolygon.rectangle(2.0, 1.0)]
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.15), _material(), radius_range=(0.005, 0.008), shapes=shapes)
        count = sim.rain_until_full(RainSettings(max_particles=40, settle_steps=3000, max_steps=20000))
        assert count == 40
        assert 0 < int(sim.polytope.sum()) < count
        assert np.all(sim.geometry.contains(sim.pos))
        assert sim.mean_kinetic_energy() < 1e-6
        pairs, _ = sim.contacts()
        assert np.all(pairs.overlap < 0.05 * sim.radius[pairs.first])
        # the bed top follows the corners of the polygons
        assert sim.bed_top() > float(np.max(sim.pos[:, 1] + sim.radius * ~sim.polytope))

    def test_footing_presses_on_polygons(self) -> None:
        """A loaded footing seated on polygons must come to rest with the particles pushing back with its load."""
        platen = LoadedPlaten(Heaviside(1.0, rise_time=0.1), mass=0.02, seat_on_bed=True)
        geometry = Geometry([(0, 0), (0.1, 0), (0.1, 0.12), (0.065, 0.12), (0.035, 0.12), (0, 0.12)],
                            [Wall(), Wall(), Open(), platen, Open(), Wall()])
        sim = DEMSimulation(geometry,
                            _material(restitution=0.1),
                            radius_range=(0.005, 0.007),
                            shapes=[ConvexPolygon.regular(6)])
        sim.rain_until_full(RainSettings(max_particles=40, settle_steps=3000, max_steps=20000))
        sim.seat_platens()
        top = sim.bed_top()
        assert top - 1e-9 <= float(sim.platens[0].vertices[:, :, 1].min()) <= 0.12
        sim.compress(0.35)
        assert abs(sim.platens[0].velocity) < 1e-3
        assert_allclose(sim.platens[0].reaction, 1.0, rtol=0.01)


class TestPolyhedra:
    """Tests of 3D simulations with polyhedra."""

    def test_cube_rests_at_the_depth_of_the_contact_law(self) -> None:
        """A cube on the floor must carry its weight at the depth W / (8/3 E* a), with a = sqrt(A / (2 pi))."""
        material = _material()
        radius = 0.01
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                            material,
                            particles=[Particle3D(0.05, 0.02, 0.05, radius, shape=CUBE)])
        sim.rest(0.5)
        _, walls = sim.contacts()
        side = radius * (4.0 * math.pi / 3.0)**(1.0 / 3.0)
        depth = _weight(material, radius) / (8.0 / 3.0 * _e_star(material) * math.sqrt(side * side / (2.0 * math.pi)))
        assert_allclose(walls.normal_force, [_weight(material, radius)], rtol=1e-6)
        assert_allclose(walls.overlap, [depth], rtol=1e-6)
        assert sim.mean_kinetic_energy() < 1e-14

    def test_tilted_cube_settles_on_a_face(self) -> None:
        """A cube dropped on a corner must come to rest flat on one of its faces."""
        orientation = (math.cos(0.3), 0.6 * math.sin(0.3), 0.0, 0.8 * math.sin(0.3))
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                            _material(),
                            particles=[Particle3D(0.05, 0.03, 0.05, 0.01, shape=CUBE, orientation=orientation)])
        sim.rest(1.0)
        up = rotation_matrices(sim.orientation)[0].T @ [0.0, 1.0, 0.0]
        assert_allclose(np.max(np.abs(up)), 1.0, atol=1e-6)

    def test_free_box_tumbles_and_conserves_momentum_and_energy(self) -> None:
        """A box spun about its intermediate axis must tumble, keeping its angular momentum and kinetic energy."""
        box = ConvexPolyhedron.box(1.0, 0.6, 0.8)
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                            _material(),
                            particles=[Particle3D(0.05, 0.05, 0.05, 0.01, shape=box)],
                            settings=GlobalSettings(gravity=0.0))
        sim.omega[0] = [0.1, 30.0, 0.5]

        def invariants() -> tuple:
            momentum = apply_inertia(rotation_matrices(sim.orientation), sim.inertia, sim.omega)[0]
            return momentum, 0.5 * float(sim.omega[0] @ momentum)

        momentum, energy = invariants()
        spins = []
        for _ in range(8000):
            sim.step()
            spins.append((rotation_matrices(sim.orientation)[0].T @ sim.omega[0])[1])
        # the spin about the intermediate axis turns over
        assert min(spins) < -25.0
        assert_allclose(invariants()[0], momentum, rtol=1e-8)
        assert_allclose(invariants()[1], energy, rtol=1e-6)

    def test_sphere_rests_on_a_cube_at_the_hertz_overlap(self) -> None:
        """A sphere on top of a cube must press on it with its weight, at the overlap of a sphere on a plane."""
        material = _material(restitution=0.1)
        side = 0.04
        cube = Particle3D(0.05, 0.5 * side, 0.05, side * (3.0 / (4.0 * math.pi))**(1.0 / 3.0), shape=CUBE)
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1),
                            material,
                            particles=[cube, Particle3D(0.05, side + 0.006, 0.05, 0.005)])
        sim.rest(0.25)
        pairs, _ = sim.contacts()
        weight = _weight(material, 0.005)
        overlap = (3.0 * weight / (4.0 * _e_star(material) * math.sqrt(0.005)))**(2.0 / 3.0)
        assert_allclose(pairs.normal_force, [weight], rtol=1e-5)
        assert_allclose(pairs.overlap, [overlap], rtol=1e-5)

    def test_stacked_cubes_rest(self) -> None:
        """A cube resting on another must press on it with its weight, face to face, without twisting."""
        material = _material(restitution=0.1)
        radius = 0.008
        side = radius * (4.0 * math.pi / 3.0)**(1.0 / 3.0)
        particles = [
            Particle3D(0.05, 0.5 * side, 0.05, radius, shape=CUBE),
            Particle3D(0.05, 1.5 * side + 0.001, 0.05, radius, shape=CUBE)
        ]
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), material, particles=particles)
        sim.rest(0.35)
        pairs, _ = sim.contacts()
        assert_allclose(pairs.normal_force, [_weight(material, radius)], rtol=1e-5)
        assert_allclose(np.abs(pairs.normal), [[0.0, 1.0, 0.0]], atol=1e-6)
        assert sim.mean_kinetic_energy() < 1e-14

    def test_polyhedra_are_rained_and_settle(self) -> None:
        """Rained polyhedra and spheres must fill a box and stay in it."""
        shapes = [None, ConvexPolyhedron.random(10, seed=1), ConvexPolyhedron.tetrahedron()]
        sim = DEMSimulation(Geometry3D.box(0.06, 0.1, 0.06), _material(), radius_range=(0.007, 0.009), shapes=shapes)
        count = sim.rain_until_full(RainSettings(max_particles=12, batch_size=12, settle_steps=500, max_steps=2500))
        assert count == sim.n >= 10
        assert 0 < int(sim.polytope.sum()) < count
        assert np.all(sim.geometry.contains(sim.pos))
        pairs, walls = sim.contacts()
        assert pairs.first.size > 0 and walls.first.size > 0

    def test_open_face_lets_polyhedra_out(self) -> None:
        """Polyhedra that fall through an open floor must be removed, with their shapes."""
        geometry = Geometry3D.box(0.1, 0.1, 0.1, bottom=Open())
        particles = [
            Particle3D(0.03, 0.05, 0.05, 0.008, shape=CUBE),
            Particle3D(0.07, 0.05, 0.05, 0.008),
            Particle3D(0.05, 0.05, 0.03, 0.006, shape=ConvexPolyhedron.tetrahedron())
        ]
        sim = DEMSimulation(geometry, _material(), particles=particles)
        sim.rest(0.4)
        assert sim.n == 0
        assert sim.n_removed == 3
        assert sim.shapes.vertices.shape[0] == 0


class TestBookkeeping:
    """Tests of the setup, the recording and the storage of shaped particles."""

    def test_time_step_follows_the_closest_face(self) -> None:
        """The time step must shrink with the distance from the centroid to the closest face of a flat shape."""
        slab = ConvexPolyhedron.box(1.0, 0.2, 1.0)
        geometry = Geometry3D.box(0.1, 0.1, 0.1)
        spheres = DEMSimulation(geometry, _material(), radius_range=(0.01, 0.01))
        slabs = DEMSimulation(geometry, _material(), radius_range=(0.01, 0.01), shapes=[slab])
        assert_allclose(slabs.dt / spheres.dt, slab.inradius)

    def test_frames_record_orientations_only_with_shapes(self) -> None:
        """Frames must store the orientations and shapes of shaped particles, and nothing more for spheres."""
        geometry = Geometry.rectangle(0.1, 0.1)
        spheres = DEMSimulation(geometry, _material(), particles=[Particle(0.05, 0.02, 0.01)])
        spheres.rest(0.001, record_every=1)
        assert spheres.frames[0].orientations is None and spheres.frames[0].shapes is None
        shaped = DEMSimulation(geometry, _material(), particles=[Particle(0.05, 0.02, 0.01, shape=SQUARE)])
        shaped.rest(0.001, record_every=1)
        frame = shaped.frames[0]
        assert frame.orientations is not None and frame.shapes is shaped.shapes
        rows, vertices = frame.polytopes()
        assert list(rows) == [0] and vertices.shape == (1, 4, 2)
        assert_allclose(frame.top(), float(vertices[0, :, 1].max()))

    def test_pickled_simulation_carries_on_identically(self) -> None:
        """A pickled simulation with polyhedra must continue exactly as the original."""
        particles = [
            Particle3D(0.03, 0.03, 0.05, 0.008, shape=CUBE, orientation=(0.9, 0.1, 0.3, 0.2)),
            Particle3D(0.05, 0.05, 0.05, 0.008, shape=ConvexPolyhedron.random(9, seed=3))
        ]
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), particles=particles)
        sim.rest(0.05)
        copy = pickle.loads(pickle.dumps(sim))
        sim.rest(0.05)
        copy.rest(0.05)
        assert_allclose(copy.pos, sim.pos, rtol=0.0, atol=0.0)
        assert_allclose(copy.orientation, sim.orientation, rtol=0.0, atol=0.0)

    def test_simulation_pickled_before_shapes_existed_carries_on(self) -> None:
        """A simulation of spheres pickled before shapes existed, e.g. a checkpoint, must carry on exactly."""
        platen = LoadedPlaten(Heaviside(1.0), mass=0.1)
        particles = [Particle3D(0.03, 0.02, 0.05, 0.01), Particle3D(0.06, 0.03, 0.05, 0.012)]
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1, top=platen), _material(), particles=particles)
        sim.rest(0.02)
        state = dict(pickle.loads(pickle.dumps(sim)).__dict__)
        # the state of such a pickle: no shapes, no orientations, and moments of inertia that were scalars in 3D
        for name in ("_DEMSimulation__table", "_DEMSimulation__rain_shapes", "_DEMSimulation__rain_reach",
                     "_DEMSimulation__pose", "_DEMSimulation__pose_index", "_DEMSimulation__regions",
                     "_DEMSimulation__solids", "orientation"):
            del state[name]
        state["inertia"] = state["inertia"][:, 0].copy()
        state["inv_inertia"] = state["inv_inertia"][:, 0].copy()
        old = DEMSimulation.__new__(DEMSimulation)
        old.__setstate__(state)
        sim.compress(0.02)
        old.compress(0.02)
        assert_allclose(old.pos, sim.pos, rtol=0.0, atol=0.0)
        assert_allclose(old.omega, sim.omega, rtol=0.0, atol=0.0)
        assert old.inertia.shape == (2, 3)

    def test_particle_with_shape_of_other_dimension_is_rejected(self) -> None:
        """A 2D particle with a polyhedron must be rejected."""
        particle = Particle(0.05, 0.05, 0.01, shape=CUBE)  # type: ignore[arg-type]
        with pytest.raises(ValueError,
                           match=re.escape("Particle 0 has a ConvexPolyhedron, but the geometry is "
                                           "two-dimensional.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), particles=[particle])

    def test_rain_shapes_of_other_dimension_are_rejected(self) -> None:
        """Polygons cannot be rained into a polyhedron."""
        with pytest.raises(ValueError,
                           match=re.escape("Shape 1 is a ConvexPolygon, but a 3D simulation needs "
                                           "polyhedra.")):
            DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), radius_range=(0.01, 0.02), shapes=[CUBE, SQUARE])

    def test_empty_list_of_shapes_is_rejected(self) -> None:
        """An empty list of shapes must be rejected."""
        with pytest.raises(ValueError, match=re.escape("shapes must not be empty; use None for spheres.")):
            DEMSimulation(Geometry.rectangle(0.1, 0.1), _material(), radius_range=(0.01, 0.02), shapes=[])

    def test_zero_quaternion_is_rejected(self) -> None:
        """A particle oriented by a zero quaternion must be rejected."""
        particle = Particle3D(0.05, 0.05, 0.05, 0.01, shape=CUBE, orientation=(0.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match=re.escape("Particle 0 has a zero quaternion as its orientation.")):
            DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), _material(), particles=[particle])
