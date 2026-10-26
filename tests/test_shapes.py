"""Tests of the convex polygonal and polyhedral particle shapes."""

import math
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies.shapes import ConvexPolygon, ConvexPolyhedron


class TestConvexPolygon:
    """Tests of ConvexPolygon."""

    def test_rectangle_is_scaled_to_unit_equivalent_radius(self) -> None:
        """A rectangle must be stored about its centroid, counter-clockwise, at an equivalent radius of one."""
        shape = ConvexPolygon.rectangle(2.0, 1.0)
        assert_allclose(shape.equivalent_radius, math.sqrt(2.0 / math.pi))
        assert_allclose(shape.vertices.mean(axis=0), [0.0, 0.0], atol=1e-15)
        assert_allclose(np.ptp(shape.vertices, axis=0) * shape.equivalent_radius, [2.0, 1.0])
        edges = np.roll(shape.vertices, -1, axis=0) - shape.vertices
        assert np.all(edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] -
                      edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0] > 0.0)

    def test_rectangle_inertia_factor(self) -> None:
        """The moment of inertia must be 4/5 of that of a prism of the section, over m R^2."""
        shape = ConvexPolygon.rectangle(2.0, 1.0)
        polar = 2.0 * (4.0 + 1.0) / 12.0
        assert_allclose(shape.inertia_factor, 0.8 * polar / (2.0 * shape.equivalent_radius**2))

    def test_many_sided_polygon_tends_to_a_sphere(self) -> None:
        """A regular polygon with many sides must have the inertia factor 2/5 of a sphere and unit radii."""
        shape = ConvexPolygon.regular(400)
        assert_allclose(shape.inertia_factor, 0.4, rtol=1e-4)
        assert_allclose(shape.circumradius, 1.0, rtol=1e-4)
        assert_allclose(shape.inradius, 1.0, rtol=1e-4)

    def test_normals_and_offsets_describe_the_edges(self) -> None:
        """Every vertex must lie on the lines of its two edges and inside all the others."""
        shape = ConvexPolygon.random(7, seed=3, aspect=1.5)
        height = shape.vertices @ shape.normals.T - shape.offsets
        assert np.all(height <= 1e-12)
        count = len(shape.vertices)
        for k in range(count):
            assert_allclose(height[k, k], 0.0, atol=1e-12)
            assert_allclose(height[(k + 1) % count, k], 0.0, atol=1e-12)
        assert_allclose(np.linalg.norm(shape.normals, axis=1), 1.0)

    def test_interior_and_collinear_points_are_dropped(self) -> None:
        """The hull must keep only the corners of the points."""
        shape = ConvexPolygon([(0, 0), (1, 0), (2, 0), (2, 2), (0, 2), (1, 1), (0, 1)])
        assert len(shape.vertices) == 4

    def test_random_polygon_is_reproducible(self) -> None:
        """The same seed must give the same polygon, and another seed another one."""
        assert_allclose(ConvexPolygon.random(6, seed=4).vertices, ConvexPolygon.random(6, seed=4).vertices)
        assert len(ConvexPolygon.random(6, seed=4).vertices) == 6
        assert not np.allclose(
            ConvexPolygon.random(6, seed=4).vertices[:3],
            ConvexPolygon.random(6, seed=5).vertices[:3])

    def test_vertices_are_read_only(self) -> None:
        """A shape is shared by many particles, so its vertices must not be writable."""
        with pytest.raises(ValueError, match="read-only"):
            ConvexPolygon.regular(5).vertices[0, 0] = 1.0

    @pytest.mark.parametrize("points, message", [
        ([(0.0, 0.0, 0.0)] * 3, "vertices must be a sequence of (x, y) points."),
        ([(0.0, 0.0), (1.0, 0.0)], "A polygon needs at least 3 points, got 2."),
        ([(0.0, 0.0), (1.0, math.nan), (0.0, 1.0)], "Vertex coordinates must be finite."),
        ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], "The points are collinear; a polygon needs an area."),
    ])
    def test_invalid_points_are_rejected(self, points: list, message: str) -> None:
        """Points that make no polygon must be rejected."""
        with pytest.raises(ValueError, match=re.escape(message)):
            ConvexPolygon(points)

    @pytest.mark.parametrize("call, message", [
        (lambda: ConvexPolygon.regular(2), "A polygon needs at least three sides, got 2."),
        (lambda: ConvexPolygon.rectangle(0.0, 1.0), "Rectangle width and height must be positive, got 0.0 and 1.0."),
        (lambda: ConvexPolygon.random(2), "A polygon needs at least three vertices, got 2."),
        (lambda: ConvexPolygon.random(5, aspect=0.0), "aspect must be positive, got 0.0."),
    ])
    def test_invalid_constructors_are_rejected(self, call: object, message: str) -> None:
        """Constructors with impossible arguments must be rejected."""
        with pytest.raises(ValueError, match=re.escape(message)):
            call()  # type: ignore[operator]


class TestConvexPolyhedron:
    """Tests of ConvexPolyhedron."""

    def test_box_has_six_faces_and_its_inertia(self) -> None:
        """A box must merge its coplanar triangles into six faces and have the inertia of a cuboid."""
        shape = ConvexPolyhedron.box(1.0, 2.0, 3.0)
        assert len(shape.faces) == 6
        assert shape.edges.shape == (12, 2)
        radius = shape.equivalent_radius
        assert_allclose(radius, (3.0 * 6.0 / (4.0 * math.pi))**(1.0 / 3.0))
        # the body frame orders the principal axes by increasing extent
        assert_allclose(np.ptp(shape.vertices, axis=0) * radius, [1.0, 2.0, 3.0])
        expected = np.array([4.0 + 9.0, 1.0 + 9.0, 1.0 + 4.0]) / 12.0 / radius**2
        assert_allclose(shape.inertia_factor, expected)

    def test_tetrahedron(self) -> None:
        """A regular tetrahedron must have four faces, six edges and a circumradius three times its inradius."""
        shape = ConvexPolyhedron.tetrahedron()
        assert len(shape.faces) == 4
        assert shape.edges.shape == (6, 2)
        assert_allclose(shape.circumradius / shape.inradius, 3.0)
        assert_allclose(shape.inertia_factor, shape.inertia_factor[0])

    def test_random_polyhedron_is_closed_and_oriented(self) -> None:
        """Every face must be counter-clockwise about its outward normal, and every edge separate two faces."""
        shape = ConvexPolyhedron.random(14, seed=2, aspect=(1.3, 1.0, 0.7))
        assert len(shape.vertices) - len(shape.edges) + len(shape.faces) == 2
        assert np.all(shape.vertices @ shape.normals.T - shape.offsets <= 1e-12)
        for k, face in enumerate(shape.faces):
            corners = shape.vertices[list(face)]
            area = 0.5 * np.cross(corners - corners[0], np.roll(corners, -1, axis=0) - corners[0]).sum(axis=0)
            assert float(area @ shape.normals[k]) > 0.0
        for (a, b), (left, right) in zip(shape.edges, shape.edge_faces):
            loop = list(shape.faces[left])
            assert loop[(loop.index(a) + 1) % len(loop)] == b
            loop = list(shape.faces[right])
            assert loop[(loop.index(b) + 1) % len(loop)] == a

    def test_body_frame_is_principal(self) -> None:
        """The second moments of the vertices of a random shape about the centroid must have no cross terms."""
        shape = ConvexPolyhedron.random(30, seed=7, aspect=(1.5, 1.0, 0.6))
        # the volume integral of x x^T over a fine sample of the hull
        rng = np.random.default_rng(0)
        points = rng.uniform(-2.0, 2.0, (400000, 3))
        inside = np.all(points @ shape.normals.T - shape.offsets <= 0.0, axis=1)
        sample = points[inside]
        assert_allclose(sample.mean(axis=0), [0.0, 0.0, 0.0], atol=0.01)
        second = sample.T @ sample / sample.shape[0]
        assert_allclose(second - np.diag(np.diag(second)), 0.0, atol=0.01)

    def test_unit_equivalent_radius(self) -> None:
        """The stored shape must have the volume of a unit sphere."""
        shape = ConvexPolyhedron.octahedron()
        # an octahedron with unit circumradius has the volume 4/3
        scale = 1.0 / shape.circumradius
        assert_allclose(4.0 / 3.0 / scale**3, 4.0 / 3.0 * math.pi)

    @pytest.mark.parametrize("points, message", [
        ([(0.0, 0.0)] * 4, "vertices must be a sequence of (x, y, z) points."),
        ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], "A polyhedron needs at least 4 points, got 3."),
        ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
          (1.0, 1.0, 0.0)], "The points are coplanar; a polyhedron needs a volume."),
    ])
    def test_invalid_points_are_rejected(self, points: list, message: str) -> None:
        """Points that make no polyhedron must be rejected."""
        with pytest.raises(ValueError, match=re.escape(message)):
            ConvexPolyhedron(points)

    @pytest.mark.parametrize("call, message", [
        (lambda: ConvexPolyhedron.box(1.0, -1.0, 1.0), "Box extents must be positive, got 1.0, -1.0 and 1.0."),
        (lambda: ConvexPolyhedron.random(3), "A polyhedron needs at least four vertices, got 3."),
        (lambda: ConvexPolyhedron.random(8, aspect=(1.0, 0.0, 1.0)), "aspect must be positive, got (1.0, 0.0, 1.0)."),
    ])
    def test_invalid_constructors_are_rejected(self, call: object, message: str) -> None:
        """Constructors with impossible arguments must be rejected."""
        with pytest.raises(ValueError, match=re.escape(message)):
            call()  # type: ignore[operator]
