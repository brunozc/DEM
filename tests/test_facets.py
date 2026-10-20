"""Tests of the facets of a domain boundary: segments in 2D and planar polygons in 3D."""

import math
import re

import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.path import Path
from numpy.testing import assert_allclose

from DEMcookies.facets import PolygonFacets, SegmentFacets
from DEMcookies.segments import project_onto_segments


def _random_polygon(rng: np.random.Generator, convex: bool) -> npt.NDArray[np.float64]:
    """Return a polygon of 3 to 7 vertices in a random plane: convex, or star-shaped and non-convex."""
    m = int(rng.integers(3, 8))
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, m))
    radii = np.ones(m) if convex else np.where(np.arange(m) % 2 == 0, 1.0, 0.35)
    flat = 0.1 * np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    first = np.cross(normal, rng.normal(size=3))
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.asarray(0.05 * rng.normal(size=3) + flat[:, :1] * first + flat[:, 1:] * second)


def _unit_square(height: float = 0.0) -> npt.NDArray[np.float64]:
    """Return the unit square in the plane y = height, its normal pointing up."""
    return np.array([[0.0, height, 0.0], [0.0, height, 1.0], [1.0, height, 1.0], [1.0, height, 0.0]])


def _sampled_distance(polygon: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the distance from each point to a dense sampling of the polygon, its interior and its boundary."""
    origin = polygon[0]
    normal = np.cross(polygon[1] - origin, polygon[2] - origin)
    normal /= np.linalg.norm(normal)
    first = (polygon[1] - origin) / np.linalg.norm(polygon[1] - origin)
    second = np.cross(normal, first)
    flat = np.column_stack(((polygon - origin) @ first, (polygon - origin) @ second))
    grid = np.linspace(flat.min() - 0.01, flat.max() + 0.01, 241)
    u, v = np.meshgrid(grid, grid)
    plane = np.column_stack((u.ravel(), v.ravel()))
    plane = plane[Path(flat).contains_points(plane)]
    region = [origin + plane[:, :1] * first + plane[:, 1:] * second]
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        region.append(start + np.linspace(0.0, 1.0, 400)[:, None] * (end - start))
    samples = np.vstack(region)
    return np.asarray(np.min(np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2), axis=1))


class TestSegmentFacets:
    """Tests of the segments of a polygonal domain."""

    def test_queries_match_segment_projection(self) -> None:
        """Distances and pair offsets must be those of the closest points of the segments."""
        rng = np.random.default_rng(0)
        starts, ends = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
        points = rng.normal(size=(6, 2))
        facets = SegmentFacets(starts, ends)
        _, offset = project_onto_segments(points, starts, ends)
        assert_allclose(facets.distances(points), np.hypot(offset[..., 0], offset[..., 1]))
        pairs = np.array([0, 3, 1, 2, 2, 0])
        assert_allclose(facets.pair_offsets(points, pairs), offset[np.arange(6), pairs], atol=1e-15)

    def test_touching_reports_overlapping_pairs_only(self) -> None:
        """Only particles closer to a segment than their radius must touch it, sorted by particle."""
        facets = SegmentFacets([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]])
        points = np.array([[0.5, 0.05], [0.95, 0.5], [0.97, 0.03], [0.5, 0.5]])
        particle, facet, offset, dist_sq = facets.touching(points, np.full(4, 0.1))
        assert particle.tolist() == [0, 1, 2, 2]
        assert facet.tolist() == [0, 1, 0, 1]
        assert_allclose(offset, [[0.0, 0.05], [-0.05, 0.0], [0.0, 0.03], [-0.03, 0.0]], atol=1e-15)
        assert_allclose(dist_sq, [0.0025, 0.0025, 0.0009, 0.0009])

    def test_moved_select_and_vertices(self) -> None:
        """Moving and selecting segments must translate and pick their end points."""
        facets = SegmentFacets([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 2.0]])
        moved = facets.moved(np.array([0.5, -1.0])).select([1])
        assert moved.count == 1 and moved.dimension == 2
        assert_allclose(moved.vertex_array(), [[[1.5, -1.0], [1.5, 1.0]]])
        assert_allclose(facets.measures, [1.0, 2.0])


class TestPolygonFacets:
    """Tests of the planar polygons of a polyhedral domain."""

    @pytest.mark.parametrize("convex", [True, False])
    def test_closest_points_against_sampling(self, convex: bool) -> None:
        """The closest point must lie on the polygon, and no sample of the polygon may be closer."""
        rng = np.random.default_rng(1 if convex else 2)
        for _ in range(20):
            polygon = _random_polygon(rng, convex)
            facets = PolygonFacets([polygon])
            points = 0.12 * rng.normal(size=(8, 3))
            offsets = facets.pair_offsets(points, np.zeros(8, dtype=np.int64))
            distance = np.linalg.norm(offsets, axis=1)
            sampled = _sampled_distance(polygon, points)
            # the exact distance can only fall below the sampled one, and by less than the sampling step
            assert np.all(distance <= sampled + 1e-12)
            assert np.all(sampled - distance < 1.5e-3)
            closest = points - offsets
            assert_allclose((closest - polygon[0]) @ facets.normals[0], 0.0, atol=1e-12)

    def test_point_over_interior_and_beyond_edge_and_vertex(self) -> None:
        """Over the interior the offset is the height; beyond an edge or a vertex it points from that feature."""
        facets = PolygonFacets([_unit_square()])
        points = np.array([[0.3, 0.2, 0.6], [1.5, -0.4, 0.5], [1.3, 0.0, 1.4], [-0.2, 0.0, 0.5]])
        offsets = facets.pair_offsets(points, np.zeros(4, dtype=np.int64))
        assert_allclose(offsets, [[0.0, 0.2, 0.0], [0.5, -0.4, 0.0], [0.3, 0.0, 0.4], [-0.2, 0.0, 0.0]], atol=1e-15)

    def test_padded_polygons_of_different_sizes(self) -> None:
        """A triangle processed with a pentagon must give the same closest points as alone."""
        triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pentagon = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.5, 1.0, 1.0], [0.5, 1.5, 1.0], [-0.5, 1.0, 1.0]])
        together = PolygonFacets([triangle, pentagon])
        alone = PolygonFacets([triangle])
        points = np.random.default_rng(3).normal(size=(10, 3))
        assert_allclose(together.distances(points)[:, 0], alone.distances(points)[:, 0], rtol=1e-12)
        assert together.vertex_counts.tolist() == [3, 5] and together.dimension == 3 and together.count == 2
        assert together.vertex_array().shape == (2, 5, 3)
        # the triangle is padded with copies of its last vertex, which a selection of it alone drops
        assert_allclose(together.vertex_array()[0, 3:], [[0.0, 1.0, 0.0]] * 2)
        assert_allclose(together.select([0]).vertex_array(), [triangle])

    def test_touching_matches_dense_distances(self) -> None:
        """The sparse search for touching pairs must find exactly the pairs closer than the radius."""
        rng = np.random.default_rng(4)
        facets = PolygonFacets([_random_polygon(rng, True), _random_polygon(rng, False), _unit_square(0.2)])
        points = 0.15 * rng.normal(size=(300, 3))
        radii = rng.uniform(0.01, 0.06, 300)
        particle, facet, offset, dist_sq = facets.touching(points, radii)
        expected = np.nonzero(facets.distances(points) < radii[:, None])
        assert particle.tolist() == expected[0].tolist() and facet.tolist() == expected[1].tolist()
        assert_allclose(dist_sq, np.einsum("ij,ij->i", offset, offset))

    def test_moved_and_selected_polygons(self) -> None:
        """A moved and selected polygon must answer queries as if it had been built there."""
        facets = PolygonFacets([_unit_square(), _unit_square(1.0)])
        shifted = facets.moved(np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.0]])).select([1])
        rebuilt = PolygonFacets([_unit_square(1.25) + np.array([0.5, 0.0, 0.0])])
        points = np.random.default_rng(5).normal(size=(10, 3))
        assert_allclose(shifted.distances(points), rebuilt.distances(points), rtol=1e-12)
        assert_allclose(shifted.vertex_array(), rebuilt.vertex_array())
        assert_allclose(facets.measures, [1.0, 1.0])

    @pytest.mark.parametrize("convex", [True, False])
    def test_seat_travel_against_marching(self, convex: bool) -> None:
        """The swept polygon must first touch a particle where marching it along its direction finds the contact."""
        rng = np.random.default_rng(6 if convex else 7)
        for _ in range(10):
            polygon = _random_polygon(rng, convex)
            facets = PolygonFacets([polygon])
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            points = polygon.mean(axis=0) + direction * rng.uniform(0.05, 0.3,
                                                                    (12, 1)) + 0.06 * rng.normal(size=(12, 3))
            radii = rng.uniform(0.005, 0.02, 12)
            if np.any(facets.distances(points)[:, 0] <= radii):
                continue
            travel = facets.seat_travel(points, radii, direction)

            def gap(s: float) -> float:
                return float(np.min(facets.moved(s * direction).distances(points)[:, 0] - radii))

            steps = np.linspace(0.0, 0.6, 3001)
            hit = next((k for k, s in enumerate(steps) if gap(s) <= 0.0), None)
            if hit is None:
                assert math.isinf(travel)
                continue
            low, high = steps[hit - 1], steps[hit]
            for _ in range(60):
                middle = 0.5 * (low + high)
                low, high = (low, middle) if gap(middle) <= 0.0 else (middle, high)
            assert_allclose(travel, high, atol=1e-9)

    def test_seat_travel_is_blocked_by_particle_ahead(self) -> None:
        """A particle that already touches the polygon on the side it moves to must stop it."""
        facets = PolygonFacets([_unit_square()])
        down = np.array([0.0, -1.0, 0.0])
        assert facets.seat_travel(np.array([[0.5, -0.05, 0.5]]), np.array([0.1]), down) == 0.0
        # the same particle behind the polygon does not block it, and nothing is ahead
        assert math.isinf(facets.seat_travel(np.array([[0.5, 0.05, 0.5]]), np.array([0.1]), down))
        assert_allclose(facets.seat_travel(np.array([[0.5, -0.5, 0.5]]), np.array([0.1]), down), 0.4)
        assert math.isinf(facets.seat_travel(np.zeros((0, 3)), np.zeros(0), down))

    def test_degenerate_polygons_are_rejected(self) -> None:
        """Polygons with too few vertices or no area must be rejected."""
        with pytest.raises(ValueError, match=re.escape("Polygon 0 must be a sequence of at least three (x, y, z)")):
            PolygonFacets([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        with pytest.raises(ValueError, match=re.escape("Polygon 1 has zero area.")):
            PolygonFacets([_unit_square(), [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
