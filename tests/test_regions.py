"""Tests of the regions of solid behind the facets of a domain."""

import math
import re

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from DEMcookies import Geometry, Geometry3D, LoadedPlaten, Heaviside, Open, Wall
from DEMcookies.regions import OPEN_BODY, _triangulate, facet_regions


def _twice_area(points: npt.NDArray[np.float64], a: int, b: int, c: int) -> float:
    """Return twice the signed area of a triangle of points in a plane."""
    u, v = points[b] - points[a], points[c] - points[a]
    return float(u[0] * v[1] - u[1] * v[0])


class TestRegions2D:
    """Tests of the regions behind the edges of a polygon."""

    def test_corners_of_a_box_are_bisected(self) -> None:
        """The regions of the floor and of a side wall of a box must meet on the bisector of their corner."""
        geometry = Geometry.rectangle(1.0, 1.0)
        regions = facet_regions(geometry, [0, 1, 3], [-1, -1, OPEN_BODY, -1])
        assert_allclose(regions.start, [0, 1, 2, 3])
        floor = regions.normals[0]
        # the floor (inward normal +y) ends on the 45 degree planes through its two corners
        assert_allclose(floor[0], [0.0, 1.0])
        assert_allclose(floor[1], [-math.sqrt(0.5), math.sqrt(0.5)])
        assert_allclose(floor[2], [math.sqrt(0.5), math.sqrt(0.5)])
        assert_allclose(regions.offsets[0], [0.0, 0.0, math.sqrt(0.5)])

    def test_edge_next_to_an_open_edge_ends_at_right_angles(self) -> None:
        """A side wall must end at right angles to itself where it meets the open top."""
        geometry = Geometry.rectangle(1.0, 1.0)
        regions = facet_regions(geometry, [0, 1, 3], [-1, -1, OPEN_BODY, -1])
        right = regions.normals[1]
        # the right wall runs up from (1, 0) to (1, 1); its top end meets the open edge
        assert_allclose(right[2], [0.0, 1.0])
        assert_allclose(regions.offsets[1, 2], 1.0)

    def test_edges_of_different_bodies_end_at_right_angles(self) -> None:
        """A loaded body may move away from its neighbours, so its region must not lean towards theirs."""
        platen = LoadedPlaten(Heaviside(1.0), mass=1.0)
        geometry = Geometry([(0, 0), (1, 0), (1, 1), (0.6, 1), (0.4, 1), (0, 1)],
                            [Wall(), Wall(), Wall(), platen, Wall(), Wall()])
        regions = facet_regions(geometry, [0, 1, 2, 3, 4, 5], [-1, -1, -1, 0, -1, -1])
        # the platen runs from (0.6, 1) to (0.4, 1), with its inward normal pointing down
        assert_allclose(regions.normals[3, 0], [0.0, -1.0], atol=1e-15)
        assert_allclose(regions.normals[3, 1], [1.0, 0.0], atol=1e-15)
        assert_allclose(regions.normals[3, 2], [-1.0, 0.0], atol=1e-15)

    def test_tip_of_a_wedge_is_split_down_its_middle(self) -> None:
        """The two faces of a wedge that juts into the domain must share its solid along its axis."""
        platen = LoadedPlaten(Heaviside(1.0), mass=1.0)
        geometry = Geometry(
            [(0, 0), (2, 0), (2, 2), (1.5, 2), (1, 1), (0.5, 2), (0, 2)],
            [Wall(), Wall(), Open(), platen, platen, Open(), Wall()])
        regions = facet_regions(geometry, [0, 1, 3, 4, 6], [-1, -1, OPEN_BODY, 0, 0, OPEN_BODY, -1])
        # the right face of the wedge runs from (1.5, 2) down to the tip (1, 1); its region ends at x = 1
        assert_allclose(regions.normals[2, 2], [-1.0, 0.0], atol=1e-12)
        assert_allclose(regions.offsets[2, 2], -1.0, atol=1e-12)
        assert_allclose(regions.normals[3, 1], [1.0, 0.0], atol=1e-12)


class TestRegions3D:
    """Tests of the regions behind the faces of a polyhedron."""

    def test_box_faces_are_split_into_two_triangles(self) -> None:
        """Every face of a box must have two regions, whose first plane is the face itself."""
        geometry = Geometry3D.box(1.0, 1.0, 1.0, top=Wall())
        regions = facet_regions(geometry, list(range(6)), [-1] * 6)
        assert regions.slot.size == 12
        assert_allclose(np.diff(regions.start), 2)
        assert_allclose(regions.normals[:, 0], np.repeat(geometry.inward_normals, 2, axis=0))

    def test_triangles_of_a_face_meet_at_right_angles(self) -> None:
        """The regions of the two triangles of a face must share their diagonal plane, facing each other."""
        geometry = Geometry3D.box(1.0, 1.0, 1.0, top=Wall())
        regions = facet_regions(geometry, list(range(6)), [-1] * 6)
        normals, offsets = regions.normals[:2], regions.offsets[:2]
        shared = [
            (a, b) for a in range(1, 4) for b in range(1, 4)
            if np.allclose(normals[0, a], -normals[1, b]) and math.isclose(offsets[0, a], -offsets[1, b], abs_tol=1e-12)
        ]
        assert len(shared) == 1
        assert_allclose(normals[0, shared[0][0]] @ normals[0, 0], 0.0, atol=1e-15)

    def test_every_point_behind_a_box_lies_in_one_region(self) -> None:
        """The regions of the walls of a closed box must split the solid around it without gaps or overlaps."""
        geometry = Geometry3D.box(1.0, 1.0, 1.0, top=Wall())
        regions = facet_regions(geometry, list(range(6)), [-1] * 6)
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.3, 1.3, (20000, 3))
        outside = ~np.all((points > 0.0) & (points < 1.0), axis=1)
        inside_region = np.all(np.einsum("trd,pd->ptr", regions.normals, points) <= regions.offsets[None], axis=2)
        assert np.all(inside_region[outside].sum(axis=1) == 1)

    def test_non_convex_face_is_triangulated(self) -> None:
        """An L-shaped face must be split into triangles that cover its area."""
        points = np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], dtype=float)
        triangles = _triangulate(points, 1e-12)
        assert_allclose(sum(0.5 * _twice_area(points, *triangle) for triangle in triangles), 3.0)
        assert all(_twice_area(points, *triangle) > 0.0 for triangle in triangles)

    def test_collinear_vertices_are_kept_on_the_outline(self) -> None:
        """A vertex in the middle of a straight side must not leave a gap in the triangles."""
        points = np.array([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)], dtype=float)
        triangles = _triangulate(points, 1e-12)
        assert_allclose(sum(0.5 * _twice_area(points, *triangle) for triangle in triangles), 2.0)

    def test_clockwise_outline_is_rejected(self) -> None:
        """A clockwise outline has no ear to clip and must be rejected."""
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=float)
        with pytest.raises(ValueError, match=re.escape("A face of the domain could not be split into triangles.")):
            _triangulate(points, 1e-12)
