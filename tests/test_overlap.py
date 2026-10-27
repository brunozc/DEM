"""Tests of the overlap of convex polytopes and of their distance from points."""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose

from DEMcookies.overlap import (Polyhedra, clip_lines, polygon_distance, polygon_overlap, polygon_region_overlap,
                                polyhedron_distance, polyhedron_overlap, polyhedron_region_overlap)
from DEMcookies.polytopes import PolytopeTable, place
from DEMcookies.shapes import ConvexPolygon, ConvexPolyhedron, Shape


def _placed(
    shape: Shape, size: float, centres: npt.ArrayLike, orientations: npt.ArrayLike
) -> Tuple[PolytopeTable, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return copies of a shape scaled to a size (its equivalent radius) and placed at centres, in space."""
    positions = np.asarray(centres, dtype=np.float64)
    count = positions.shape[0]
    table = PolytopeTable.build(positions.shape[1], [shape] * count, np.full(count, size))
    pose = place(table, np.arange(count), positions, np.asarray(orientations, dtype=np.float64))
    return table, pose.vertices, pose.normals, pose.offsets


def _cubes(centres: npt.ArrayLike, orientations: npt.ArrayLike, side: float = 1.0) -> Polyhedra:
    """Return cubes of a side length at centres, as a batch of polyhedra."""
    cube = ConvexPolyhedron.box(1.0, 1.0, 1.0)
    table, vertices, normals, offsets = _placed(cube, side * cube.equivalent_radius, centres, orientations)
    return Polyhedra(vertices, normals, offsets, table.edges, table.edge_faces, table.faces)


def _squares(centres: npt.ArrayLike,
             angles: npt.ArrayLike) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return unit squares at centres, as vertices, normals and offsets."""
    square = ConvexPolygon.rectangle(1.0, 1.0)
    _, vertices, normals, offsets = _placed(square, square.equivalent_radius, centres, angles)
    return vertices, normals, offsets


def _quaternion_about_z(angle: float) -> Tuple[float, float, float, float]:
    """Return the quaternion of a rotation about the z axis."""
    return (math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle))


class TestClipLines:
    """Tests of the Cyrus-Beck clipping."""

    def test_segment_is_clipped_to_a_square(self) -> None:
        """A segment that crosses a unit square must keep only its part inside."""
        normals = np.array([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
        offsets = np.array([[1.0, 0.0, 1.0, 0.0]])
        t0, t1, inside = clip_lines(np.array([[-1.0, 0.5]]), np.array([[3.0, 0.0]]), normals, offsets, 0.0, 1.0)
        assert inside[0]
        assert_allclose([t0[0], t1[0]], [1.0 / 3.0, 2.0 / 3.0])

    def test_parallel_line_outside_is_empty(self) -> None:
        """A line parallel to a half-plane and outside it must be empty, whatever its extent."""
        normals = np.array([[[0.0, 1.0]]])
        _, _, inside = clip_lines(np.array([[0.0, 2.0]]), np.array([[1.0, 0.0]]), normals, np.array([[1.0]]), -np.inf,
                                  np.inf)
        assert not inside[0]


class TestPolygonOverlap:
    """Tests of polygon_overlap and polygon_region_overlap."""

    def test_offset_squares(self) -> None:
        """Two unit squares offset by (0.5, 0.25) must overlap in a 0.5 x 0.75 rectangle, crossed by their chord."""
        vertices, normals, offsets = _squares([[0.0, 0.0], [0.5, 0.25]], [0.0, 0.0])
        overlap = polygon_overlap(vertices[:1], normals[:1], offsets[:1], vertices[1:], normals[1:], offsets[1:])
        assert_allclose(overlap.measure, [0.375])
        assert_allclose(overlap.centroid, [[0.25, 0.125]], atol=1e-15)
        # the outlines cross at (0, 0.5) and (0.5, -0.25); the chord points from j towards i
        assert_allclose(overlap.surface, [[-0.75, -0.5]])

    def test_swapping_the_bodies_reverses_the_surface(self) -> None:
        """The contact surface must point from j towards i whichever body is which."""
        vertices, normals, offsets = _squares([[0.0, 0.0], [0.6, 0.3]], [0.2, -0.4])
        forward = polygon_overlap(vertices[:1], normals[:1], offsets[:1], vertices[1:], normals[1:], offsets[1:])
        backward = polygon_overlap(vertices[1:], normals[1:], offsets[1:], vertices[:1], normals[:1], offsets[:1])
        assert_allclose(forward.measure, backward.measure)
        assert_allclose(forward.centroid, backward.centroid)
        assert_allclose(forward.surface, -backward.surface)

    def test_vertex_into_edge(self) -> None:
        """A square turned by 45 degrees whose corner sinks d into the top of another must overlap in a triangle."""
        depth = 0.1
        half_diagonal = math.sqrt(0.5)
        vertices, normals, offsets = _squares([[0.0, 0.0], [0.0, 0.5 + half_diagonal - depth]], [0.0, 0.25 * math.pi])
        overlap = polygon_overlap(vertices[1:], normals[1:], offsets[1:], vertices[:1], normals[:1], offsets[:1])
        assert_allclose(overlap.measure, [depth * depth])
        assert_allclose(overlap.centroid, [[0.0, 0.5 - depth / 3.0]], atol=1e-14)
        # the chord is the part of the top edge inside the turned square, pointing up from the lower square
        assert_allclose(overlap.surface, [[0.0, 2.0 * depth]], atol=1e-14)

    def test_separated_and_identical_polygons(self) -> None:
        """Separated polygons must not overlap, and a polygon must overlap itself in its whole area."""
        vertices, normals, offsets = _squares([[0.0, 0.0], [1.5, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0])
        overlap = polygon_overlap(vertices[[0, 0]], normals[[0, 0]], offsets[[0, 0]], vertices[[1, 2]], normals[[1, 2]],
                                  offsets[[1, 2]])
        assert_allclose(overlap.measure, [0.0, 1.0])

    def test_aligned_squares_side_by_side(self) -> None:
        """Squares whose top and bottom edges lie on the same lines must still overlap only once."""
        vertices, normals, offsets = _squares([[0.0, 0.0], [0.9, 0.0]], [0.0, 0.0])
        overlap = polygon_overlap(vertices[:1], normals[:1], offsets[:1], vertices[1:], normals[1:], offsets[1:])
        assert_allclose(overlap.measure, [0.1], rtol=1e-6)
        assert_allclose(overlap.centroid, [[0.45, 0.0]], atol=1e-9)
        assert_allclose(overlap.surface, [[-1.0, 0.0]], atol=1e-8)

    def test_squares_touching_along_an_edge_barely_overlap(self) -> None:
        """Squares that touch along an edge must overlap in no more than a sliver of the order of rounding."""
        vertices, normals, offsets = _squares([[0.0, 0.0], [1.0, 0.0]], [0.0, 0.0])
        overlap = polygon_overlap(vertices[:1], normals[:1], offsets[:1], vertices[1:], normals[1:], offsets[1:])
        assert overlap.measure[0] < 1e-8

    def test_region_behind_a_floor(self) -> None:
        """A square sunk 0.05 into a floor must overlap the region behind it in a 1 x 0.05 strip."""
        vertices, normals, offsets = _squares([[0.2, 0.45]], [0.0])
        region_normals = np.array([[[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]])
        region_offsets = np.array([[0.0, 1.0, 1.0]])
        overlap = polygon_region_overlap(vertices, normals, offsets, region_normals, region_offsets)
        assert_allclose(overlap.measure, [0.05])
        assert_allclose(overlap.centroid, [[0.2, -0.025]], atol=1e-15)
        assert_allclose(overlap.surface, [[0.0, 1.0]], atol=1e-15)

    def test_region_cuts_the_square_at_its_side(self) -> None:
        """The side planes of a region must clip the overlap and the contact surface."""
        vertices, normals, offsets = _squares([[0.9, 0.45]], [0.0])
        region_normals = np.array([[[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]])
        region_offsets = np.array([[0.0, 1.0, 1.0]])
        overlap = polygon_region_overlap(vertices, normals, offsets, region_normals, region_offsets)
        # the square spans x = 0.4 to 1.4, and the region ends at x = 1
        assert_allclose(overlap.measure, [0.6 * 0.05])
        assert_allclose(overlap.surface, [[0.0, 0.6]], atol=1e-15)


class TestPolyhedronOverlap:
    """Tests of polyhedron_overlap and polyhedron_region_overlap."""

    def test_offset_cubes(self) -> None:
        """Two unit cubes offset by (0.5, 0.25, 0.1) must overlap in a box, with the faces of j inside i as surface."""
        identity = (1.0, 0.0, 0.0, 0.0)
        overlap = polyhedron_overlap(_cubes([[0.0, 0.0, 0.0]], [identity]), _cubes([[0.5, 0.25, 0.1]], [identity]))
        assert_allclose(overlap.measure, [0.5 * 0.75 * 0.9])
        assert_allclose(overlap.centroid, [[0.25, 0.125, 0.05]], atol=1e-14)
        assert_allclose(overlap.surface, [[-0.75 * 0.9, -0.5 * 0.9, -0.5 * 0.75]])

    def test_edge_into_face(self) -> None:
        """A cube turned by 45 degrees about z whose edge sinks d into the top of another overlaps in a prism."""
        depth = 0.1
        half_diagonal = math.sqrt(0.5)
        lower = _cubes([[0.0, 0.0, 0.0]], [(1.0, 0.0, 0.0, 0.0)])
        upper = _cubes([[0.0, 0.5 + half_diagonal - depth, 0.0]], [_quaternion_about_z(0.25 * math.pi)])
        overlap = polyhedron_overlap(upper, lower)
        assert_allclose(overlap.measure, [depth * depth])
        assert_allclose(overlap.centroid, [[0.0, 0.5 - depth / 3.0, 0.0]], atol=1e-14)
        assert_allclose(overlap.surface, [[0.0, 2.0 * depth, 0.0]], atol=1e-14)

    def test_overlap_of_random_shapes_matches_sampling(self) -> None:
        """The volume and centroid of the overlap of two turned random polyhedra must match a Monte Carlo sample."""
        shape = ConvexPolyhedron.random(12, seed=4)
        rng = np.random.default_rng(1)
        quaternions = rng.normal(size=(2, 4))
        quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
        table, vertices, normals, offsets = _placed(shape, 0.7, [[0.0, 0.0, 0.0], [0.4, 0.5, -0.3]], quaternions)
        bodies = Polyhedra(vertices, normals, offsets, table.edges, table.edge_faces, table.faces)
        first = Polyhedra(*(getattr(bodies, name)[:1] for name in Polyhedra.__dataclass_fields__))
        second = Polyhedra(*(getattr(bodies, name)[1:] for name in Polyhedra.__dataclass_fields__))
        overlap = polyhedron_overlap(first, second)
        points = rng.uniform(-1.5, 1.5, (2000000, 3))
        inside = (np.all(points @ normals[0].T - offsets[0] <= 0.0, axis=1)
                  & np.all(points @ normals[1].T - offsets[1] <= 0.0, axis=1))
        assert_allclose(overlap.measure, [inside.mean() * 27.0], rtol=0.02)
        assert_allclose(overlap.centroid[0], points[inside].mean(axis=0), atol=0.01)

    def test_separated_and_identical_cubes(self) -> None:
        """Separated cubes must not overlap, and a cube must overlap itself in its whole volume."""
        identity = (1.0, 0.0, 0.0, 0.0)
        overlap = polyhedron_overlap(_cubes([[0.0, 0.0, 0.0]] * 2, [identity] * 2),
                                     _cubes([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]], [identity] * 2))
        assert_allclose(overlap.measure, [0.0, 1.0])

    def test_aligned_cubes_side_by_side(self) -> None:
        """Cubes whose top, bottom, front and back faces lie in the same planes must still overlap only once."""
        identity = (1.0, 0.0, 0.0, 0.0)
        overlap = polyhedron_overlap(_cubes([[0.0, 0.0, 0.0]], [identity]), _cubes([[0.9, 0.0, 0.0]], [identity]))
        assert_allclose(overlap.measure, [0.1], rtol=1e-6)
        assert_allclose(overlap.centroid, [[0.45, 0.0, 0.0]], atol=1e-9)
        assert_allclose(overlap.surface, [[-1.0, 0.0, 0.0]], atol=1e-8)

    def test_cubes_touching_face_to_face_barely_overlap(self) -> None:
        """Cubes that touch face to face must overlap in no more than a sliver of the order of rounding."""
        identity = (1.0, 0.0, 0.0, 0.0)
        overlap = polyhedron_overlap(_cubes([[0.0, 0.0, 0.0]], [identity]), _cubes([[0.0, 1.0, 0.0]], [identity]))
        assert overlap.measure[0] < 1e-8

    def test_region_behind_a_floor(self) -> None:
        """A unit cube sunk 0.1 into a floor must overlap the region behind it in a 1 x 0.1 x 1 slab."""
        cube = _cubes([[0.1, 0.4, 0.2]], [(1.0, 0.0, 0.0, 0.0)])
        region_normals = np.array([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]])
        region_offsets = np.array([[0.0, 1.0, 1.0, 1.0]])
        overlap = polyhedron_region_overlap(cube, region_normals, region_offsets)
        assert_allclose(overlap.measure, [0.1])
        assert_allclose(overlap.centroid, [[0.1, -0.05, 0.2]], atol=1e-14)
        assert_allclose(overlap.surface, [[0.0, 1.0, 0.0]], atol=1e-14)

    def test_region_with_a_tilted_side(self) -> None:
        """A side plane at 45 degrees must cut the slab under a cube along a slant."""
        cube = _cubes([[0.0, 0.4, 0.0]], [(1.0, 0.0, 0.0, 0.0)])
        side = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        region_normals = np.array([[[0.0, 1.0, 0.0], side, [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
        region_offsets = np.array([[0.0, 0.0, 2.0, 2.0, 2.0]])
        overlap = polyhedron_region_overlap(cube, region_normals, region_offsets)
        # the slab spans x = -0.5 to 0.5 and y = -0.1 to 0; the side plane x + y <= 0 cuts off its part at x > -y
        volume = 0.5 * 0.1 + 0.5 * 0.1 * 0.1
        assert_allclose(overlap.measure, [volume])
        assert_allclose(overlap.surface, [[0.0, 0.5, 0.0]], atol=1e-14)


class TestDistances:
    """Tests of polygon_distance and polyhedron_distance."""

    def test_polygon_distance(self) -> None:
        """Points beyond an edge, beyond a corner and inside a unit square must get their signed distances."""
        vertices, normals, offsets = _squares([[0.0, 0.0]] * 3, [0.0] * 3)
        points = np.array([[2.0, 0.0], [0.6, 0.6], [0.1, 0.0]])
        distance, normal = polygon_distance(points, vertices, normals, offsets)
        assert_allclose(distance, [1.5, math.sqrt(0.02), -0.4])
        assert_allclose(normal, [[1.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5)], [1.0, 0.0]], atol=1e-15)

    def test_polyhedron_distance(self) -> None:
        """Points beyond a face, beyond an edge and inside a unit cube must get their signed distances."""
        cubes = _cubes([[0.0, 0.0, 0.0]] * 3, [(1.0, 0.0, 0.0, 0.0)] * 3)
        points = np.array([[2.0, 0.0, 0.0], [0.8, 0.8, 0.0], [0.1, 0.0, 0.05]])
        distance, normal = polyhedron_distance(points, cubes)
        assert_allclose(distance, [1.5, math.sqrt(0.18), -0.4])
        assert_allclose(normal, [[1.0, 0.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5), 0.0], [1.0, 0.0, 0.0]], atol=1e-15)
