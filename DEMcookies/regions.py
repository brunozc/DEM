"""Regions of solid behind the facets of a domain, against which polygonal and polyhedral particles press.

A particle that presses on a wall overlaps the solid behind it. That solid is split among the solid facets: the
region of a facet lies behind its plane and between side planes along its edges. Between two facets of the same
rigid body the side plane bisects the angle between them, so that the regions of neighbouring facets meet without a
gap or an overlap, around a convex corner of the domain as well as at the tip of a wedge that juts into it. Along an
edge shared with an open facet, or with a facet of another body, which may move away, the side plane is at right
angles to the facet. A face of a polyhedron, which need not be convex, is split into triangles, each with a region of
its own; the side planes between the triangles of a face are at right angles to it, so their regions add up to that
of the face.

Every region is a list of half-spaces n . x <= h whose first is the plane of the facet, with n its inward normal.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .geometry import Geometry
from .geometry3d import Domain, Geometry3D
from .globals import FOLD_TOLERANCE, GEOMETRY_TOLERANCE

# body code of an open facet, which has no solid behind it
OPEN_BODY = -2


@dataclass(frozen=True, eq=False)
class FacetRegions:
    """Regions of solid behind the solid facets of a domain, at the position the facets have in the geometry.

    Attributes:
        slot: solid facet that each region belongs to, shape (t,).
        normals: outward unit normals of the half-spaces of each region, the first being the inward normal of its
            facet, shape (t, r, d).
        offsets: offsets of those half-spaces, shape (t, r) [m].
        start: first region of each solid facet, with one more entry for the end, shape (s + 1,).
    """

    slot: npt.NDArray[np.int64]
    normals: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.float64]
    start: npt.NDArray[np.int64]

    def moved(self, shifts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Offsets of the half-spaces of every region once each solid facet has been translated.

        Args:
            shifts: translation of each solid facet, shape (s, d) [m].

        Returns:
            The offsets, shape (t, r) [m].
        """
        return np.asarray(self.offsets + np.einsum("trd,td->tr", self.normals, shifts[self.slot]))


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a vector scaled to unit length."""
    return np.asarray(vector / np.linalg.norm(vector))


def _side_normal(own: npt.NDArray[np.float64], other: npt.NDArray[np.float64], bisect: bool) -> npt.NDArray[np.float64]:
    """Outward normal of the side plane of the region of a facet along one of its edges.

    Args:
        own: unit vector in the plane of the facet, at right angles to the edge, pointing away from the facet.
        other: the same vector for the facet across the edge, pointing away from that facet.
        bisect: whether the side plane bisects the angle between the facets, rather than standing at right angles
            to this one.

    Returns:
        The unit normal.
    """
    if bisect:
        difference = own - other
        if float(difference @ difference) > FOLD_TOLERANCE:
            return _unit(difference)
    return own


def facet_regions(geometry: Domain, solid: Sequence[int], body: Sequence[int]) -> FacetRegions:
    """Split the solid behind the solid facets of a domain into regions.

    Args:
        geometry: the domain.
        solid: index in the geometry of each solid facet, in the order of the solid slots.
        body: body of every facet of the geometry: OPEN_BODY for an open facet, -1 for a wall, and the index of the
            loaded body for a facet of one.

    Returns:
        The regions.
    """
    if isinstance(geometry, Geometry):
        return _regions_2d(geometry, solid, body)
    return _regions_3d(geometry, solid, body)


def _assemble(regions: List[Tuple[int, List[npt.NDArray[np.float64]], List[float]]], slots: int,
              dimension: int) -> FacetRegions:
    """Pack the regions of the solid facets into arrays.

    Args:
        regions: solid slot, normals and offsets of each region, in the order of the slots.
        slots: number of solid facets.
        dimension: number of space dimensions.

    Returns:
        The regions.
    """
    planes = dimension + 1
    slot = np.array([r[0] for r in regions], dtype=np.int64)
    normals = np.array([r[1] for r in regions], dtype=np.float64).reshape(-1, planes, dimension)
    offsets = np.array([r[2] for r in regions], dtype=np.float64).reshape(-1, planes)
    start = np.searchsorted(slot, np.arange(slots + 1)).astype(np.int64)
    return FacetRegions(slot, normals, offsets, start)


def _regions_2d(geometry: Geometry, solid: Sequence[int], body: Sequence[int]) -> FacetRegions:
    """Regions behind the solid edges of a polygon, one per edge.

    Args:
        geometry: the polygon.
        solid: index of each solid edge.
        body: body of every edge.

    Returns:
        The regions.
    """
    starts, ends = geometry.edge_starts, geometry.edge_ends
    normals = geometry.inward_normals
    count = geometry.n_edges
    regions: List[Tuple[int, List[npt.NDArray[np.float64]], List[float]]] = []
    for s, edge in enumerate(solid):
        start, end = starts[edge], ends[edge]
        planes = [normals[edge]]
        offsets = [float(normals[edge] @ start)]
        # edge k - 1 ends where edge k starts, and edge k + 1 starts where it ends
        for corner, far, neighbour, neighbour_far in ((start, end, (edge - 1) % count, starts[(edge - 1) % count]),
                                                      (end, start, (edge + 1) % count, ends[(edge + 1) % count])):
            bisect = body[neighbour] != OPEN_BODY and body[neighbour] == body[edge]
            normal = _side_normal(_unit(corner - far), _unit(corner - neighbour_far), bisect)
            planes.append(normal)
            offsets.append(float(normal @ corner))
        regions.append((s, planes, offsets))
    return _assemble(regions, len(solid), 2)


def _triangulate(points: npt.NDArray[np.float64], tolerance: float) -> List[Tuple[int, int, int]]:
    """Split a simple polygon into triangles by clipping ears.

    Args:
        points: vertices of the polygon, counter-clockwise, shape (n, 2) [m].
        tolerance: largest twice-area of a triangle taken as degenerate [m^2].

    Returns:
        The vertex indices of each triangle, counter-clockwise; degenerate triangles of collinear vertices are left
        out.
    """

    def twice_area(a: int, b: int, c: int) -> float:
        return float((points[b, 0] - points[a, 0]) * (points[c, 1] - points[a, 1]) - (points[b, 1] - points[a, 1]) *
                     (points[c, 0] - points[a, 0]))

    def contains(a: int, b: int, c: int, k: int) -> bool:
        return twice_area(a, b, k) >= -tolerance and twice_area(b, c, k) >= -tolerance and twice_area(c, a,
                                                                                                      k) >= -tolerance

    remaining = list(range(points.shape[0]))
    triangles: List[Tuple[int, int, int]] = []
    while len(remaining) > 3:
        corners = [(remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]) for k in range(len(remaining))]
        ears = [
            k for k, (a, b, c) in enumerate(corners) if twice_area(a, b, c) > tolerance and not any(
                contains(a, b, c, other) for other in remaining if other not in (a, b, c))
        ]
        if ears:
            triangles.append(corners[ears[0]])
            remaining.pop(ears[0])
            continue
        # without an ear only straight corners can go: a vertex between two collinear edges spans no triangle
        straight = [k for k, (a, b, c) in enumerate(corners) if abs(twice_area(a, b, c)) <= tolerance]
        if not straight:
            raise ValueError("A face of the domain could not be split into triangles.")
        remaining.pop(straight[0])
    if twice_area(*remaining) > tolerance:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _regions_3d(geometry: Geometry3D, solid: Sequence[int], body: Sequence[int]) -> FacetRegions:
    """Regions behind the solid faces of a polyhedron, one per triangle of each face.

    Args:
        geometry: the polyhedron.
        solid: index of each solid face.
        body: body of every face.

    Returns:
        The regions.
    """
    vertices = geometry.vertices
    normals = geometry.inward_normals
    tolerance = GEOMETRY_TOLERANCE * float(np.max(np.ptp(vertices, axis=0)))
    # every face listed counter-clockwise about its inward normal, so that neighbours run along a shared edge in
    # opposite directions
    loops: List[List[int]] = []
    for outline, inward in zip(geometry.faces, normals):
        corners = vertices[list(outline)]
        area = np.cross(corners - corners[0], np.roll(corners, -1, axis=0) - corners[0]).sum(axis=0)
        loops.append(list(outline) if float(area @ inward) > 0.0 else list(reversed(outline)))
    owner: Dict[Tuple[int, int], int] = {}
    for f, loop in enumerate(loops):
        for a, b in zip(loop, loop[1:] + loop[:1]):
            owner[(a, b)] = f

    regions: List[Tuple[int, List[npt.NDArray[np.float64]], List[float]]] = []
    for s, face in enumerate(solid):
        loop = loops[face]
        normal = normals[face]
        boundary = set(zip(loop, loop[1:] + loop[:1]))
        # a basis (u, v) of the plane with u x v = n, so that counter-clockwise in it is counter-clockwise about n
        u = _unit(vertices[loop[1]] - vertices[loop[0]])
        v = np.cross(normal, u)
        local = np.column_stack((vertices[loop] @ u, vertices[loop] @ v))
        for a, b, c in _triangulate(local, tolerance * float(np.max(np.ptp(local, axis=0)))):
            triangle = [loop[a], loop[b], loop[c]]
            planes = [normal]
            offsets = [float(normal @ vertices[triangle[0]])]
            for p, q in zip(triangle, triangle[1:] + triangle[:1]):
                start, end = vertices[p], vertices[q]
                own = _unit(np.cross(end - start, normal))
                other = own
                bisect = False
                if (p, q) in boundary:
                    neighbour = owner[(q, p)]
                    bisect = body[neighbour] != OPEN_BODY and body[neighbour] == body[face]
                    # the neighbour runs along the edge from q to p, counter-clockwise about its own inward normal
                    other = _unit(np.cross(start - end, normals[neighbour]))
                side = _side_normal(own, other, bisect)
                planes.append(side)
                offsets.append(float(side @ start))
            regions.append((s, planes, offsets))
    return _assemble(regions, len(solid), 3)
