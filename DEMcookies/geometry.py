"""Polygonal simulation domain whose edges each carry a boundary condition.

``Geometry3D``, in ``geometry3d``, is its three-dimensional counterpart, a polyhedron whose faces carry the boundary
conditions.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .boundary import PLATEN_CONDITIONS, BoundaryCondition, Open, PlatenCondition, Wall, resolve_direction
from .facets import SegmentFacets
from .globals import GEOMETRY_TOLERANCE
from .segments import project_onto_segments

# the closest-point query stays importable from here, where existing scripts and tests import it from
__all__ = ["Geometry", "project_onto_segments"]


def _cross(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Return the z component of the cross product of two 2D vectors.

    Args:
        a: first vector.
        b: second vector.

    Returns:
        The z component of a x b.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_intersect(p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64], q1: npt.NDArray[np.float64],
                        q2: npt.NDArray[np.float64], tolerance: float) -> bool:
    """Check whether two closed segments share at least one point.

    Args:
        p1: start of the first segment [m].
        p2: end of the first segment [m].
        q1: start of the second segment [m].
        q2: end of the second segment [m].
        tolerance: distance below which two points are taken to touch [m].

    Returns:
        True if the segments cross, touch or overlap.
    """
    p_length = float(np.hypot(*(p2 - p1)))
    q_length = float(np.hypot(*(q2 - q1)))
    # signed distances of each end point from the line through the other segment
    d1 = _cross(q2 - q1, p1 - q1) / q_length
    d2 = _cross(q2 - q1, p2 - q1) / q_length
    d3 = _cross(p2 - p1, q1 - p1) / p_length
    d4 = _cross(p2 - p1, q2 - p1) / p_length
    if ((d1 > tolerance and d2 < -tolerance) or (d1 < -tolerance and d2 > tolerance)) and \
            ((d3 > tolerance and d4 < -tolerance) or (d3 < -tolerance and d4 > tolerance)):
        return True
    # collinear or touching configurations
    return ((abs(d1) <= tolerance and _in_box(q1, q2, p1, tolerance))
            or (abs(d2) <= tolerance and _in_box(q1, q2, p2, tolerance))
            or (abs(d3) <= tolerance and _in_box(p1, p2, q1, tolerance))
            or (abs(d4) <= tolerance and _in_box(p1, p2, q2, tolerance)))


def _in_box(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], point: npt.NDArray[np.float64],
            tolerance: float) -> bool:
    """Check whether a point lies in the axis-aligned bounding box of a segment.

    Args:
        a: start of the segment.
        b: end of the segment.
        point: point to test.
        tolerance: tolerance on the coordinates [m].

    Returns:
        True if the point lies inside the box, up to the tolerance.
    """
    return bool(
        min(a[0], b[0]) - tolerance <= point[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= point[1] <= max(a[1], b[1]) + tolerance)


class Geometry:
    """Simple polygon whose edges carry boundary conditions.

    Edge ``k`` runs from vertex ``k`` to vertex ``k + 1`` (the last edge closes the polygon back to vertex 0) and
    carries ``boundary_conditions[k]``. The vertices may be given in either orientation. To apply a load to part of
    a side only, split that side into several collinear edges and make the middle one a ``LoadedPlaten``, or a
    ``PrescribedPlaten`` to prescribe its displacement instead. Edges that carry the same platen object form one
    rigid body, which can have any polyline shape.
    """

    def __init__(self, vertices: Sequence[Sequence[float]], boundary_conditions: Sequence[BoundaryCondition]) -> None:
        """Create and validate the domain.

        Args:
            vertices: polygon vertices (x, y) [m].
            boundary_conditions: one boundary condition per edge.

        Raises:
            ValueError: if the vertices are not a sequence of (x, y) pairs.
            ValueError: if fewer than three vertices are given.
            ValueError: if a vertex coordinate is not finite.
            ValueError: if the number of boundary conditions differs from the number of edges.
            ValueError: if an edge has zero length.
            ValueError: if the polygon has zero area.
            ValueError: if two adjacent edges fold back onto each other.
            ValueError: if two non-adjacent edges intersect.
            ValueError: if a loaded body without an explicit direction has edges whose inward normals cancel out.
        """
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("vertices must be a sequence of (x, y) pairs.")
        n_vertices = points.shape[0]
        if n_vertices < 3:
            raise ValueError(f"A polygon needs at least three vertices, got {n_vertices}.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Vertex coordinates must be finite.")
        if len(boundary_conditions) != n_vertices:
            raise ValueError(f"Expected one boundary condition per edge ({n_vertices}), "
                             f"got {len(boundary_conditions)}.")

        self.__vertices = points
        self.__boundary_conditions: Tuple[BoundaryCondition, ...] = tuple(boundary_conditions)
        self.__starts = points
        self.__ends = np.roll(points, -1, axis=0)
        self.__facets = SegmentFacets(self.__starts, self.__ends)
        edge_vectors = self.__ends - self.__starts
        lengths = np.hypot(edge_vectors[:, 0], edge_vectors[:, 1])

        size = float(np.max(np.ptp(points, axis=0)))
        for k in range(n_vertices):
            if lengths[k] <= GEOMETRY_TOLERANCE * size:
                raise ValueError(f"Edge {k} has zero length.")
        signed_area = 0.5 * float(
            np.sum(self.__starts[:, 0] * self.__ends[:, 1] - self.__ends[:, 0] * self.__starts[:, 1]))
        if abs(signed_area) <= GEOMETRY_TOLERANCE * size**2:
            raise ValueError("The polygon has zero area.")

        self.__tangents = edge_vectors / lengths[:, None]
        self.__lengths = lengths
        self.__area = abs(signed_area)
        # the inward normal is the left normal of a counter-clockwise polygon and the right normal otherwise
        orientation = 1.0 if signed_area > 0.0 else -1.0
        self.__normals = orientation * np.column_stack((-self.__tangents[:, 1], self.__tangents[:, 0]))
        self.__check_simple(GEOMETRY_TOLERANCE * size)

        bodies: Dict[PlatenCondition, List[int]] = {}
        for k, condition in enumerate(self.__boundary_conditions):
            if isinstance(condition, PLATEN_CONDITIONS):
                bodies.setdefault(condition, []).append(k)
        self.__platen_groups = tuple((spec, tuple(edges)) for spec, edges in bodies.items())
        for spec, edges in self.__platen_groups:
            resolve_direction(spec, self.__starts[list(edges)], self.__ends[list(edges)], self.__normals[list(edges)])

    @classmethod
    def rectangle(cls,
                  width: float,
                  height: float,
                  bottom: Optional[BoundaryCondition] = None,
                  right: Optional[BoundaryCondition] = None,
                  top: Optional[BoundaryCondition] = None,
                  left: Optional[BoundaryCondition] = None,
                  origin: Tuple[float, float] = (0.0, 0.0)) -> "Geometry":
        """Create a rectangular box, open at the top unless stated otherwise.

        The edges are numbered bottom (0), right (1), top (2) and left (3).

        Args:
            width: width of the box [m].
            height: height of the box [m].
            bottom: boundary condition of the bottom edge; None gives a wall.
            right: boundary condition of the right edge; None gives a wall.
            top: boundary condition of the top edge; None leaves it open.
            left: boundary condition of the left edge; None gives a wall.
            origin: lower left corner of the box [m].

        Returns:
            The rectangular domain.

        Raises:
            ValueError: if the width or height is not positive.
        """
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Rectangle width and height must be positive, got {width} and {height}.")
        x0, y0 = origin
        vertices = [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]
        conditions: Sequence[BoundaryCondition] = (bottom or Wall(), right or Wall(), top or Open(), left or Wall())
        return cls(vertices, conditions)

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 2."""
        return 2

    @property
    def facet_name(self) -> str:
        """Name of the facets of the domain, which are edges."""
        return "edge"

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Polygon vertices, shape (n, 2) [m]."""
        return self.__vertices.copy()

    @property
    def boundary_conditions(self) -> Tuple[BoundaryCondition, ...]:
        """Boundary condition of each edge."""
        return self.__boundary_conditions

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return int(self.__vertices.shape[0])

    @property
    def n_facets(self) -> int:
        """Number of facets, i.e. of edges."""
        return self.n_edges

    @property
    def facets(self) -> SegmentFacets:
        """The edges as segments, for the contact queries."""
        return self.__facets

    @property
    def edge_starts(self) -> npt.NDArray[np.float64]:
        """Start vertex of each edge, shape (n, 2) [m]."""
        return self.__starts.copy()

    @property
    def edge_ends(self) -> npt.NDArray[np.float64]:
        """End vertex of each edge, shape (n, 2) [m]."""
        return self.__ends.copy()

    @property
    def edge_lengths(self) -> npt.NDArray[np.float64]:
        """Length of each edge [m]."""
        return self.__lengths.copy()

    @property
    def inward_normals(self) -> npt.NDArray[np.float64]:
        """Unit normal of each edge pointing into the domain, shape (n, 2)."""
        return self.__normals.copy()

    @property
    def platen_groups(self) -> Tuple[Tuple[PlatenCondition, Tuple[int, ...]], ...]:
        """Each moving body with the indices of its edges, ordered by their first edge."""
        return self.__platen_groups

    @property
    def area(self) -> float:
        """Area enclosed by the polygon [m^2]."""
        return self.__area

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Bounds (x_min, y_min, x_max, y_max) of the polygon [m]."""
        x_min, y_min = self.__vertices.min(axis=0)
        x_max, y_max = self.__vertices.max(axis=0)
        return float(x_min), float(y_min), float(x_max), float(y_max)

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lowest and highest coordinates (x, y) of the polygon [m]."""
        return self.__vertices.min(axis=0), self.__vertices.max(axis=0)

    def contains(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Check which points lie inside the polygon, using the even-odd crossing rule.

        Args:
            points: points to test, shape (n, 2) [m].

        Returns:
            True for every point inside the polygon.
        """
        x = points[:, 0:1]
        y = points[:, 1:2]
        x1, y1 = self.__starts[:, 0], self.__starts[:, 1]
        x2, y2 = self.__ends[:, 0], self.__ends[:, 1]
        straddles = (y1 > y) != (y2 > y)
        # horizontal edges never straddle, so their (zero) denominator is never used
        dy = np.where(np.abs(y2 - y1) > 0.0, y2 - y1, 1.0)
        x_cross = x1 + (y - y1) * (x2 - x1) / dy
        crossings = np.count_nonzero(straddles & (x < x_cross), axis=1)
        return np.asarray(crossings % 2 == 1)

    def distance_to_edges(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance from every point to every edge.

        Args:
            points: points, shape (n, 2) [m].

        Returns:
            The distances, shape (n, edges) [m].
        """
        return self.__facets.distances(points)

    def distance_to_facets(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance from every point to every facet, i.e. to every edge.

        Args:
            points: points, shape (n, 2) [m].

        Returns:
            The distances, shape (n, edges) [m].
        """
        return self.__facets.distances(points)

    def __check_simple(self, tolerance: float) -> None:
        """Check that adjacent edges do not fold back and non-adjacent edges do not intersect.

        Args:
            tolerance: distance below which two points are taken to touch [m].

        Raises:
            ValueError: if two adjacent edges fold back onto each other.
            ValueError: if two non-adjacent edges intersect.
        """
        n = self.n_edges
        for k in range(n):
            following = (k + 1) % n
            a = self.__tangents[k]
            b = self.__tangents[following]
            if abs(_cross(a, b)) <= GEOMETRY_TOLERANCE and float(a @ b) < 0.0:
                raise ValueError(f"Edges {k} and {following} fold back onto each other.")
        for k in range(n):
            for m in range(k + 2, n):
                if k == 0 and m == n - 1:
                    continue
                if _segments_intersect(self.__starts[k], self.__ends[k], self.__starts[m], self.__ends[m], tolerance):
                    raise ValueError(f"Edges {k} and {m} intersect; the polygon must be simple.")
