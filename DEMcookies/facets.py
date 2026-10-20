"""Facets of the boundary of a domain: line segments in 2D and planar polygons in 3D.

A facet set answers the geometric queries of the boundary contacts: the vector from the closest point of a facet to a
point, the particles that touch a facet, and how far the facets can travel along a direction before they touch a
particle. A facet set never changes; ``moved`` returns a translated copy, which is how a loaded body moves.
"""

import copy
from typing import Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .globals import QUERY_CHUNK_SIZE
from .segments import project_onto_segments

Contacts = Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]


class FacetSet(Protocol):
    """Geometric queries on the facets of a domain boundary, shared by segments and polygons."""

    @property
    def dimension(self) -> int:
        """Number of space dimensions."""
        ...

    @property
    def count(self) -> int:
        """Number of facets."""
        ...

    @property
    def measures(self) -> npt.NDArray[np.float64]:
        """Length (2D) or area (3D) of each facet [m or m^2]."""
        ...

    def moved(self, shifts: npt.NDArray[np.float64]) -> "FacetSet":
        """Return the facets translated by a vector each, shape (facets, d) or (d,) [m]."""
        ...

    def select(self, indices: npt.ArrayLike) -> "FacetSet":
        """Return the facets with the given indices."""
        ...

    def vertex_array(self) -> npt.NDArray[np.float64]:
        """Return the vertices of each facet, shape (facets, vertices per facet, d) [m]."""
        ...

    def distances(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the distance from every point to every facet, shape (n, facets) [m]."""
        ...

    def pair_offsets(self, points: npt.NDArray[np.float64], facets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Return the vector from the closest point of facet k to point k, for each pair k, shape (k, d) [m]."""
        ...

    def touching(self, points: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> Contacts:
        """Return the (particle, facet) pairs closer than the radius, their offsets and squared distances."""
        ...

    def seat_travel(self, points: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                    direction: npt.NDArray[np.float64]) -> float:
        """Return how far the facets can travel along a direction before they touch a particle [m]."""
        ...


def _time_to_contact(positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64], start: npt.NDArray[np.float64],
                     end: npt.NDArray[np.float64], direction: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Distance a segment can travel along a direction before it touches each particle.

    Args:
        positions: particle centres, shape (n, 2) [m].
        radii: particle radii [m].
        start: start point of the segment [m].
        end: end point of the segment [m].
        direction: unit direction of travel.

    Returns:
        The travel distance for each particle, infinite for particles the segment never touches [m].
    """
    # seen from the segment, the particles move against the direction of travel
    motion = -direction
    edge = end - start
    length = float(np.hypot(edge[0], edge[1]))
    tangent = edge / length
    normal = np.array([-tangent[1], tangent[0]])
    travel = np.full(radii.shape, np.inf)

    # contact with the interior of the segment: the distance to its line drops to the radius
    relative = positions - start
    height = relative @ normal
    rate = float(motion @ normal)
    if abs(rate) > 0.0:
        approaching = height * rate < 0.0
        line_travel = (np.abs(height) - radii) / abs(rate)
        foot = (relative + line_travel[:, None] * motion) @ tangent
        hits = approaching & (line_travel >= 0.0) & (foot >= 0.0) & (foot <= length)
        travel = np.where(hits, line_travel, travel)

    # contact with an end point: the first root of |w + s motion| = r
    for point in (start, end):
        offset = positions - point
        along = offset @ motion
        discriminant = along**2 - (np.einsum("ij,ij->i", offset, offset) - radii**2)
        first = -along - np.sqrt(np.maximum(discriminant, 0.0))
        hits = (discriminant >= 0.0) & (first >= 0.0)
        travel = np.where(hits, np.minimum(travel, first), travel)
    return travel


def _sphere_travel(offset: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                   motion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Distance a point can travel before it comes within a radius of each particle.

    Args:
        offset: vectors from the point to the particle centres, shape (..., d) [m].
        radii: particle radii, broadcastable to the leading shape of the offsets [m].
        motion: unit direction in which the particles move relative to the point.

    Returns:
        The first root of |offset + s motion| = r, infinite where there is none ahead [m].
    """
    along = offset @ motion
    discriminant = along**2 - (np.einsum("...k,...k->...", offset, offset) - radii**2)
    first = -along - np.sqrt(np.maximum(discriminant, 0.0))
    return np.asarray(np.where((discriminant >= 0.0) & (first >= 0.0), first, np.inf))


class SegmentFacets:
    """Line segments, the facets of a polygonal domain."""

    def __init__(self, starts: npt.ArrayLike, ends: npt.ArrayLike) -> None:
        """Store the segments.

        Args:
            starts: start point of each segment, shape (s, 2) [m].
            ends: end point of each segment, shape (s, 2) [m].
        """
        self.__starts = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
        self.__ends = np.asarray(ends, dtype=np.float64).reshape(-1, 2)

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 2."""
        return 2

    @property
    def count(self) -> int:
        """Number of segments."""
        return int(self.__starts.shape[0])

    @property
    def starts(self) -> npt.NDArray[np.float64]:
        """Start point of each segment, shape (s, 2) [m]."""
        return self.__starts.copy()

    @property
    def ends(self) -> npt.NDArray[np.float64]:
        """End point of each segment, shape (s, 2) [m]."""
        return self.__ends.copy()

    @property
    def measures(self) -> npt.NDArray[np.float64]:
        """Length of each segment [m]."""
        edges = self.__ends - self.__starts
        return np.asarray(np.hypot(edges[:, 0], edges[:, 1]))

    def moved(self, shifts: npt.NDArray[np.float64]) -> "SegmentFacets":
        """Return the segments translated by a vector each.

        Args:
            shifts: translation of each segment, shape (s, 2), or of all of them, shape (2,) [m].

        Returns:
            The translated segments.
        """
        return SegmentFacets(self.__starts + shifts, self.__ends + shifts)

    def select(self, indices: npt.ArrayLike) -> "SegmentFacets":
        """Return the segments with the given indices.

        Args:
            indices: indices of the segments to keep.

        Returns:
            The selected segments.
        """
        chosen = np.asarray(indices, dtype=np.int64)
        return SegmentFacets(self.__starts[chosen], self.__ends[chosen])

    def vertex_array(self) -> npt.NDArray[np.float64]:
        """Return the start and end point of each segment, shape (s, 2, 2) [m]."""
        return np.stack((self.__starts, self.__ends), axis=1)

    def distances(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance from every point to every segment.

        Args:
            points: points, shape (n, 2) [m].

        Returns:
            The distances, shape (n, s) [m].
        """
        _, offset = project_onto_segments(points, self.__starts, self.__ends)
        return np.asarray(np.hypot(offset[..., 0], offset[..., 1]))

    def pair_offsets(self, points: npt.NDArray[np.float64], facets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Vector from the closest point of a segment to a point, for pairs of points and segments.

        Args:
            points: points, shape (k, 2) [m].
            facets: index of the segment paired with each point, shape (k,).

        Returns:
            The vectors from the closest points to the points, shape (k, 2) [m].
        """
        starts = self.__starts[facets]
        direction = self.__ends[facets] - starts
        relative = points - starts
        length_sq = np.einsum("ij,ij->i", direction, direction)
        t = np.clip(np.einsum("ij,ij->i", relative, direction) / length_sq, 0.0, 1.0)
        return np.asarray(relative - t[:, None] * direction)

    def touching(self, points: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> Contacts:
        """Find the particles that overlap a segment.

        Args:
            points: particle centres, shape (n, 2) [m].
            radii: particle radii, shape (n,) [m].

        Returns:
            The particle and segment of each overlapping pair, sorted by particle and then by segment, the vector
            from the closest point of the segment to the centre, shape (k, 2) [m], and its squared length [m^2].
        """
        _, offset = project_onto_segments(points, self.__starts, self.__ends)
        dist_sq = offset[..., 0]**2 + offset[..., 1]**2
        particle, facet = np.nonzero(dist_sq < (radii**2)[:, None])
        return particle, facet, offset[particle, facet], dist_sq[particle, facet]

    def seat_travel(self, points: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                    direction: npt.NDArray[np.float64]) -> float:
        """How far the segments can travel along a direction before one of them touches a particle.

        Args:
            points: particle centres, shape (n, 2) [m].
            radii: particle radii [m].
            direction: unit direction of travel.

        Returns:
            The travel [m]: zero if a particle already touches a segment on the side it moves to, infinite if no
            particle is in the way.
        """
        travel = np.inf
        for start, end in zip(self.__starts, self.__ends):
            _, offset = project_onto_segments(points, start[None, :], end[None, :])
            gap = np.hypot(offset[:, 0, 0], offset[:, 0, 1]) - radii
            # a particle already touching the edge on the side the body moves to blocks it
            if np.any((gap <= 0.0) & (offset[:, 0, :] @ direction > 0.0)):
                return 0.0
            travel = min(travel, float(np.min(_time_to_contact(points, radii, start, end, direction))))
        return travel


class PolygonFacets:
    """Planar polygons, the facets of a polyhedral domain.

    Each polygon is simple and planar but need not be convex. The closest point of a polygon to a point is the
    projection of the point onto the plane of the polygon if that falls inside it, and the closest point of its
    boundary otherwise. Polygons with fewer vertices than the largest one are padded with copies of their last
    vertex, so that all of them can be processed together.
    """

    def __init__(self, polygons: Sequence[npt.ArrayLike]) -> None:
        """Precompute the plane, an in-plane frame and the edges of every polygon.

        Args:
            polygons: vertices of each polygon in order around it, each of shape (m, 3) [m].

        Raises:
            ValueError: if a polygon is not a sequence of at least three (x, y, z) points.
            ValueError: if a polygon has zero area.
        """
        arrays = [np.asarray(polygon, dtype=np.float64) for polygon in polygons]
        for k, polygon in enumerate(arrays):
            if polygon.ndim != 2 or polygon.shape[1] != 3 or polygon.shape[0] < 3:
                raise ValueError(f"Polygon {k} must be a sequence of at least three (x, y, z) points.")
        count = len(arrays)
        size = max([polygon.shape[0] for polygon in arrays], default=3)
        self.__counts = np.array([polygon.shape[0] for polygon in arrays], dtype=np.int64)
        self.__vertices = np.zeros((count, size, 3))
        self.__valid = np.zeros((count, size), dtype=bool)
        self.__normals = np.zeros((count, 3))
        self.__areas = np.zeros(count)
        for k, polygon in enumerate(arrays):
            m = polygon.shape[0]
            self.__vertices[k, :m] = polygon
            self.__vertices[k, m:] = polygon[-1]
            self.__valid[k, :m] = True
            # Newell's area vector, exact for any planar polygon whether convex or not
            area_vector = 0.5 * np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
            area = float(np.sqrt(area_vector @ area_vector))
            if area <= 0.0:
                raise ValueError(f"Polygon {k} has zero area.")
            self.__normals[k] = area_vector / area
            self.__areas[k] = area

        following = np.where(self.__valid, (np.arange(size)[None, :] + 1) % np.maximum(self.__counts[:, None], 1),
                             np.arange(size)[None, :])
        ends = np.take_along_axis(self.__vertices, following[:, :, None], axis=1)
        self.__edge_starts = self.__vertices.copy()
        self.__edge_vectors = ends - self.__vertices
        lengths_sq = np.einsum("fmk,fmk->fm", self.__edge_vectors, self.__edge_vectors)
        self.__edge_valid = self.__valid & (lengths_sq > 0.0)
        self.__edge_lengths_sq = np.where(self.__edge_valid, lengths_sq, 1.0)

        # in-plane frame: the first axis along the longest edge, the second completing a right-handed frame
        self.__origins = self.__vertices[:, 0].copy()
        longest = self.__edge_vectors[np.arange(count),
                                      np.argmax(np.where(self.__edge_valid, lengths_sq, -1.0), axis=1)]
        along = longest - np.einsum("fk,fk->f", longest, self.__normals)[:, None] * self.__normals
        along_length = np.sqrt(np.einsum("fk,fk->f", along, along))
        self.__axis_u = along / np.where(along_length > 0.0, along_length, 1.0)[:, None]
        self.__axis_v = np.cross(self.__normals, self.__axis_u)
        relative = self.__vertices - self.__origins[:, None, :]
        self.__plane_u = np.einsum("fmk,fk->fm", relative, self.__axis_u)
        self.__plane_v = np.einsum("fmk,fk->fm", relative, self.__axis_v)
        self.__plane_end_u = np.take_along_axis(self.__plane_u, following, axis=1)
        self.__plane_end_v = np.take_along_axis(self.__plane_v, following, axis=1)
        self.__shifts = np.zeros((count, 3))

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 3."""
        return 3

    @property
    def count(self) -> int:
        """Number of polygons."""
        return int(self.__counts.size)

    @property
    def measures(self) -> npt.NDArray[np.float64]:
        """Area of each polygon [m^2]."""
        return self.__areas.copy()

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Unit normal of each polygon, oriented by the right-hand rule around its vertices, shape (f, 3)."""
        return self.__normals.copy()

    @property
    def vertex_counts(self) -> npt.NDArray[np.int64]:
        """Number of vertices of each polygon."""
        return self.__counts.copy()

    def moved(self, shifts: npt.NDArray[np.float64]) -> "PolygonFacets":
        """Return the polygons translated by a vector each.

        Args:
            shifts: translation of each polygon, shape (f, 3), or of all of them, shape (3,) [m].

        Returns:
            The translated polygons.
        """
        moved = copy.copy(self)
        moved.__shifts = self.__shifts + np.broadcast_to(shifts, self.__shifts.shape)
        return moved

    def select(self, indices: npt.ArrayLike) -> "PolygonFacets":
        """Return the polygons with the given indices.

        Args:
            indices: indices of the polygons to keep.

        Returns:
            The selected polygons.
        """
        chosen = np.asarray(indices, dtype=np.int64).reshape(-1)
        # the selection keeps only the padding that its own largest polygon needs
        size = int(self.__counts[chosen].max(initial=3))
        selected = copy.copy(self)
        selected.__counts = self.__counts[chosen]
        selected.__vertices = self.__vertices[chosen, :size]
        selected.__valid = self.__valid[chosen, :size]
        selected.__normals = self.__normals[chosen]
        selected.__areas = self.__areas[chosen]
        selected.__edge_starts = self.__edge_starts[chosen, :size]
        selected.__edge_vectors = self.__edge_vectors[chosen, :size]
        selected.__edge_valid = self.__edge_valid[chosen, :size]
        selected.__edge_lengths_sq = self.__edge_lengths_sq[chosen, :size]
        selected.__origins = self.__origins[chosen]
        selected.__axis_u = self.__axis_u[chosen]
        selected.__axis_v = self.__axis_v[chosen]
        selected.__plane_u = self.__plane_u[chosen, :size]
        selected.__plane_v = self.__plane_v[chosen, :size]
        selected.__plane_end_u = self.__plane_end_u[chosen, :size]
        selected.__plane_end_v = self.__plane_end_v[chosen, :size]
        selected.__shifts = self.__shifts[chosen]
        return selected

    def vertex_array(self) -> npt.NDArray[np.float64]:
        """Return the vertices of each polygon, padded with copies of its last vertex, shape (f, m, 3) [m]."""
        return np.asarray(self.__vertices + self.__shifts[:, None, :])

    def distances(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance from every point to every polygon.

        Args:
            points: points, shape (n, 3) [m].

        Returns:
            The distances, shape (n, f) [m].
        """
        n, count = points.shape[0], self.count
        particle = np.repeat(np.arange(n, dtype=np.int64), count)
        facet = np.tile(np.arange(count, dtype=np.int64), n)
        offset = self.pair_offsets(points[particle], facet)
        return np.asarray(np.sqrt(np.einsum("ij,ij->i", offset, offset)).reshape(n, count))

    def pair_offsets(self, points: npt.NDArray[np.float64], facets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Vector from the closest point of a polygon to a point, for pairs of points and polygons.

        Args:
            points: points, shape (k, 3) [m].
            facets: index of the polygon paired with each point, shape (k,).

        Returns:
            The vectors from the closest points to the points, shape (k, 3) [m].
        """
        pairs = facets.size
        result = np.zeros((pairs, 3))
        # the boundary edges of every pair are processed at once, so long batches are cut into chunks
        chunk = max(1, QUERY_CHUNK_SIZE // max(self.__vertices.shape[1], 1))
        for begin in range(0, pairs, chunk):
            end = min(begin + chunk, pairs)
            result[begin:end] = self.__offsets(points[begin:end], facets[begin:end])
        return result

    def touching(self, points: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> Contacts:
        """Find the particles that overlap a polygon.

        Only particles closer to the plane of a polygon than their radius can touch it, so the closest points are
        computed for those pairs alone.

        Args:
            points: particle centres, shape (n, 3) [m].
            radii: particle radii, shape (n,) [m].

        Returns:
            The particle and polygon of each overlapping pair, sorted by particle and then by polygon, the vector
            from the closest point of the polygon to the centre, shape (k, 3) [m], and its squared length [m^2].
        """
        relative = points[:, None, :] - (self.__origins + self.__shifts)[None, :, :]
        heights = np.einsum("nfk,fk->nf", relative, self.__normals)
        particle, facet = np.nonzero(np.abs(heights) < radii[:, None])
        offset = self.pair_offsets(points[particle], facet)
        dist_sq = np.einsum("ij,ij->i", offset, offset)
        hit = dist_sq < radii[particle]**2
        return particle[hit], facet[hit], offset[hit], dist_sq[hit]

    def seat_travel(self, points: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                    direction: npt.NDArray[np.float64]) -> float:
        """How far the polygons can travel along a direction before one of them touches a particle.

        A polygon swept along the direction first touches a sphere with its face, with one of its edges or with one
        of its vertices, which are the planes, cylinders and spheres at a radius around them.

        Args:
            points: particle centres, shape (n, 3) [m].
            radii: particle radii [m].
            direction: unit direction of travel.

        Returns:
            The travel [m]: zero if a particle already touches a polygon on the side it moves to, infinite if no
            particle is in the way.
        """
        n, count = points.shape[0], self.count
        if n == 0 or count == 0:
            return np.inf
        offsets = self.pair_offsets(np.repeat(points, count, axis=0), np.tile(np.arange(count, dtype=np.int64), n))
        gap = np.sqrt(np.einsum("ij,ij->i", offsets, offsets)) - np.repeat(radii, count)
        # a particle already touching a face on the side the body moves to blocks it
        if np.any((gap <= 0.0) & (offsets @ direction > 0.0)):
            return 0.0
        # seen from the polygons, the particles move against the direction of travel
        motion = -direction
        travel = np.full((n, count), np.inf)

        # contact with the interior of a face: the distance to its plane drops to the radius
        origins = self.__origins + self.__shifts
        relative = points[:, None, :] - origins[None, :, :]
        heights = np.einsum("nfk,fk->nf", relative, self.__normals)
        rate = self.__normals @ motion
        moving = np.abs(rate) > 0.0
        safe_rate = np.where(moving, np.abs(rate), 1.0)
        plane_travel = (np.abs(heights) - radii[:, None]) / safe_rate[None, :]
        foot = relative + plane_travel[:, :, None] * motion
        foot_u = np.einsum("nfk,fk->nf", foot, self.__axis_u)
        foot_v = np.einsum("nfk,fk->nf", foot, self.__axis_v)
        inside = self.__inside_plane(foot_u, foot_v)
        hits = moving[None, :] & (heights * rate[None, :] < 0.0) & (plane_travel >= 0.0) & inside
        travel = np.where(hits, plane_travel, travel)

        # contact with an edge: the distance to the line of the edge drops to the radius within the edge
        lengths = np.sqrt(self.__edge_lengths_sq)
        tangents = self.__edge_vectors / lengths[:, :, None]
        starts = self.__edge_starts + self.__shifts[:, None, :]
        w = points[:, None, None, :] - starts[None, :, :, :]
        w_along = np.einsum("nfmk,fmk->nfm", w, tangents)
        w_across = w - w_along[..., None] * tangents[None, ...]
        motion_along = tangents @ motion
        motion_across = motion[None, None, :] - motion_along[..., None] * tangents
        a = np.einsum("fmk,fmk->fm", motion_across, motion_across)
        b = np.einsum("nfmk,fmk->nfm", w_across, motion_across)
        c = np.einsum("nfmk,nfmk->nfm", w_across, w_across) - radii[:, None, None]**2
        discriminant = b**2 - a[None, ...] * c
        oblique = self.__edge_valid & (a > 1.0e-12)
        safe_a = np.where(oblique, a, 1.0)
        first = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / safe_a[None, ...]
        along = w_along + first * motion_along[None, ...]
        hits = oblique[None, ...] & (discriminant >= 0.0) & (first >= 0.0) & (along >= 0.0) & (along <= lengths)
        travel = np.minimum(travel, np.where(hits, first, np.inf).min(axis=2))

        # contact with a vertex
        vertices = self.__vertices + self.__shifts[:, None, :]
        corner = _sphere_travel(points[:, None, None, :] - vertices[None, ...], radii[:, None, None], motion)
        travel = np.minimum(travel, corner.min(axis=2))
        return float(travel.min())

    def __offsets(self, points: npt.NDArray[np.float64], facets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Vector from the closest point of a polygon to a point, for one batch of pairs.

        Args:
            points: points, shape (k, 3) [m].
            facets: index of the polygon paired with each point, shape (k,).

        Returns:
            The vectors from the closest points to the points, shape (k, 3) [m].
        """
        x = points - self.__shifts[facets]
        relative = x - self.__origins[facets]
        normals = self.__normals[facets]
        heights = np.einsum("ij,ij->i", relative, normals)
        u = np.einsum("ij,ij->i", relative, self.__axis_u[facets])
        v = np.einsum("ij,ij->i", relative, self.__axis_v[facets])
        inside = self.__inside_pairs(u, v, facets)

        # closest point of the boundary, from the closest point of every edge
        edges = self.__edge_vectors[facets]
        from_start = x[:, None, :] - self.__edge_starts[facets]
        t = np.clip(np.einsum("kmd,kmd->km", from_start, edges) / self.__edge_lengths_sq[facets], 0.0, 1.0)
        offsets = from_start - t[:, :, None] * edges
        dist_sq = np.where(self.__edge_valid[facets], np.einsum("kmd,kmd->km", offsets, offsets), np.inf)
        nearest = np.argmin(dist_sq, axis=1)
        boundary = offsets[np.arange(facets.size), nearest]
        return np.asarray(np.where(inside[:, None], heights[:, None] * normals, boundary))

    def __inside_pairs(self, u: npt.NDArray[np.float64], v: npt.NDArray[np.float64],
                       facets: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        """Check which in-plane points lie inside their polygon, using the even-odd crossing rule.

        Args:
            u: first in-plane coordinate of each point [m].
            v: second in-plane coordinate of each point [m].
            facets: index of the polygon of each point.

        Returns:
            True for every point inside its polygon.
        """
        start_u, start_v = self.__plane_u[facets], self.__plane_v[facets]
        end_u, end_v = self.__plane_end_u[facets], self.__plane_end_v[facets]
        return _crossings_odd(u[:, None], v[:, None], start_u, start_v, end_u, end_v, self.__valid[facets])

    def __inside_plane(self, u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Check which in-plane points lie inside the polygons, for every point and every polygon.

        Args:
            u: first in-plane coordinate of each point in the frame of each polygon, shape (n, f) [m].
            v: second in-plane coordinate, shape (n, f) [m].

        Returns:
            True for every point inside each polygon, shape (n, f).
        """
        return _crossings_odd(u[..., None], v[..., None], self.__plane_u[None], self.__plane_v[None],
                              self.__plane_end_u[None], self.__plane_end_v[None], self.__valid[None])


def _crossings_odd(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], start_u: npt.NDArray[np.float64],
                   start_v: npt.NDArray[np.float64], end_u: npt.NDArray[np.float64], end_v: npt.NDArray[np.float64],
                   valid: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Count the edges that a ray from each point along +u crosses, and report the points with an odd count.

    Args:
        u: first coordinate of the points, with a trailing axis of length one [m].
        v: second coordinate of the points, of the same shape [m].
        start_u: first coordinate of the start of each edge, the edges along the last axis [m].
        start_v: second coordinate of the start of each edge [m].
        end_u: first coordinate of the end of each edge [m].
        end_v: second coordinate of the end of each edge [m].
        valid: True for the edges that belong to the polygon, False for padding.

    Returns:
        True for every point inside its polygon, with the last axis removed.
    """
    straddles = valid & ((start_v > v) != (end_v > v))
    # edges parallel to the ray never straddle it, so their (zero) denominator is never used
    rise = np.where(np.abs(end_v - start_v) > 0.0, end_v - start_v, 1.0)
    crossing = start_u + (v - start_v) * (end_u - start_u) / rise
    return np.asarray(np.count_nonzero(straddles & (u < crossing), axis=-1) % 2 == 1)
