"""Tests of the table of particle shapes and of the rotations of rigid bodies."""

import math

import numpy as np
from numpy.testing import assert_allclose

from DEMcookies.polytopes import (PolytopeTable, apply_inertia, frame_polygons, identity_orientations, place,
                                  random_orientations, rotate, rotation_matrices)
from DEMcookies.shapes import ConvexPolygon, ConvexPolyhedron


class TestPolytopeTable:
    """Tests of PolytopeTable."""

    def test_spheres_have_empty_rows(self) -> None:
        """A sphere must have no vertices, planes that cut nothing, and the inertia and reach of a sphere."""
        table = PolytopeTable.build(2, [None, ConvexPolygon.regular(5)], np.array([0.01, 0.02]))
        assert_allclose(table.vertex_count, [0, 5])
        assert_allclose(table.normals[0], 0.0)
        assert_allclose(table.offsets[0], 1.0)
        assert_allclose(table.inertia_factor[0], 0.4)
        assert_allclose(table.reach[0], 0.01)
        assert_allclose(table.inradius[0], 0.01)
        assert list(table.polytope) == [False, True]

    def test_shapes_are_scaled_to_the_radius(self) -> None:
        """A shape must be scaled to the radius of its particle."""
        shape = ConvexPolyhedron.box(1.0, 1.0, 1.0)
        table = PolytopeTable.build(3, [shape], np.array([0.5]))
        assert_allclose(table.vertices[0], 0.5 * shape.vertices)
        assert_allclose(table.offsets[0], 0.5 * shape.offsets)
        assert_allclose(table.reach[0], 0.5 * shape.circumradius)
        assert_allclose(table.inertia_factor[0], shape.inertia_factor)

    def test_polygons_are_closed_by_padding(self) -> None:
        """A polygon with fewer vertices than the widest must be padded with its first vertex."""
        table = PolytopeTable.build(2, [ConvexPolygon.regular(3), ConvexPolygon.regular(6)], np.ones(2))
        assert table.vertices.shape == (2, 6, 2)
        assert_allclose(table.vertices[0, 3:], np.repeat(table.vertices[0, :1], 3, axis=0))
        assert_allclose(table.normals[0, 3:], 0.0)

    def test_concatenate_and_take(self) -> None:
        """Joining two tables must pad the narrower one, and taking rows must keep their shapes."""
        first = PolytopeTable.build(3, [None, ConvexPolyhedron.tetrahedron()], np.ones(2))
        second = PolytopeTable.build(3, [ConvexPolyhedron.box(1.0, 2.0, 3.0)], np.full(1, 2.0))
        joined = first.concatenate(second)
        assert joined.vertices.shape == (3, 8, 3)
        assert joined.faces.shape == (3, 6, 4)
        assert_allclose(joined.vertex_count, [0, 4, 8])
        kept = joined.take(np.array([False, True, True]))
        assert_allclose(kept.vertices[0, :4], first.vertices[1])
        assert_allclose(kept.vertices[1], second.vertices[0])
        # the padding faces of the tetrahedron stay all zero
        assert_allclose(kept.faces[0, 4:], 0)


class TestRotations:
    """Tests of the orientation helpers."""

    def test_quaternion_matrices_are_rotations(self) -> None:
        """The matrix of a unit quaternion must be orthogonal with determinant one."""
        quaternions = random_orientations(np.random.default_rng(0), 5, 3)
        matrices = rotation_matrices(quaternions)
        assert_allclose(np.einsum("kji,kjl->kil", matrices, matrices),
                        np.broadcast_to(np.eye(3), (5, 3, 3)),
                        atol=1e-14)
        assert_allclose(np.linalg.det(matrices), 1.0)

    def test_rotate_about_an_axis(self) -> None:
        """A turn by a rotation vector must add up with the current orientation."""
        start = identity_orientations(1, 3)
        quarter = rotate(start, np.array([[0.0, 0.0, 0.5 * math.pi]]))
        assert_allclose(rotation_matrices(quarter)[0] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)
        # turning about x afterwards acts in space
        both = rotate(quarter, np.array([[0.5 * math.pi, 0.0, 0.0]]))
        assert_allclose(rotation_matrices(both)[0] @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-15)
        # a very small turn keeps its accuracy
        tiny = rotate(start, np.array([[0.0, 1.0e-12, 0.0]]))
        assert_allclose(tiny[0], [1.0, 0.0, 0.5e-12, 0.0])

    def test_planar_rotation(self) -> None:
        """An angle must give the counter-clockwise rotation of the plane, and turns must add."""
        assert_allclose(rotation_matrices(np.array([0.5 * math.pi]))[0] @ [1.0, 0.0], [0.0, 1.0], atol=1e-15)
        assert_allclose(rotate(np.array([0.3]), np.array([0.2])), [0.5])

    def test_inertia_and_its_inverse(self) -> None:
        """Multiplying by R diag(I) R^T and then by its inverse must give the vector back."""
        rotation = rotation_matrices(random_orientations(np.random.default_rng(1), 4, 3))
        principal = np.array([[1.0, 2.0, 3.0]] * 4)
        vectors = np.random.default_rng(2).normal(size=(4, 3))
        momenta = apply_inertia(rotation, principal, vectors)
        assert_allclose(apply_inertia(rotation, 1.0 / principal, momenta), vectors)
        # the kinetic energy is positive for any spin
        assert np.all(np.einsum("ij,ij->i", momenta, vectors) > 0.0)

    def test_random_angles_cover_the_circle(self) -> None:
        """Random planar orientations must lie in [0, 2 pi)."""
        angles = random_orientations(np.random.default_rng(3), 1000, 2)
        assert angles.min() >= 0.0 and angles.max() < 2.0 * math.pi
        assert abs(np.mean(np.cos(angles))) < 0.1


class TestPlacement:
    """Tests of place and frame_polygons."""

    def test_placed_planes_keep_the_vertices_inside(self) -> None:
        """The vertices of a placed shape must lie on or inside its placed face planes."""
        table = PolytopeTable.build(3, [ConvexPolyhedron.random(10, seed=1)], np.array([0.3]))
        positions = np.array([[1.0, 2.0, 3.0]])
        pose = place(table, np.arange(1), positions, random_orientations(np.random.default_rng(4), 1, 3))
        heights = np.einsum("pfd,pvd->pvf", pose.normals, pose.vertices) - pose.offsets[:, None, :]
        assert np.all(heights <= 1e-12)
        assert_allclose(pose.vertices[0].mean(axis=0), positions[0], atol=0.3)

    def test_frame_polygons_skip_the_spheres(self) -> None:
        """Only the polytopes of a state must be returned, at their positions."""
        table = PolytopeTable.build(2, [None, ConvexPolygon.regular(4)], np.ones(2))
        rows, vertices = frame_polygons(table, np.array([[0.0, 0.0], [5.0, 0.0]]), np.zeros(2))
        assert list(rows) == [1]
        assert_allclose(vertices[0].mean(axis=0), [5.0, 0.0], atol=1e-15)
