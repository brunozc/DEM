"""Overlap of convex polytopes, the geometry of the contacts of polygonal and polyhedral particles.

Two convex bodies that overlap do so in one convex region, bounded by the parts of the surface of each body that lie
inside the other. Its area (2D) or volume (3D) and its centroid follow from that boundary with the divergence theorem,
without building the region itself: the boundary is cut into pieces, each piece spans a triangle (2D) or a tetrahedron
(3D) with a reference point, and the signed measures and first moments of those simplices add up to those of the
region. The pieces are clipped with the half-spaces of the two bodies, which works on whole batches of contacts:

* in 2D, every edge of each polygon clipped to the other polygon;
* in 3D, every edge of each polyhedron clipped to the other polyhedron, and every line where a face of one meets a
  face of the other, clipped to both. The piece of a face boundary is assigned to that face, whose plane carries the
  apex of its tetrahedra.

The size and direction of the contact come from the surface of body j inside body i: its area vector in 3D, and in
2D the chord between the two points where the outlines cross, turned to point from j to i. A region of space given by
half-spaces alone, such as the solid behind a wall, takes the place of body j: the lines where its planes meet stand in
for its edges, and its first plane is the contact surface.

A face of one body may lie in the plane of a face of the other, as the tops of two aligned boxes do; the boundary of
the overlap on that plane would then be counted from both faces. Body j, or the region, is therefore grown by a
relative amount of the order of the rounding error before its overlap is measured, which leaves the face of body i
alone on that plane and changes nothing else measurably.
"""

from dataclasses import dataclass, replace
from types import ModuleType
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from .backend import array_module, quiet
from .globals import OVERLAP_TIE_BREAK, PARALLEL_PLANES_TOLERANCE
from .vectors import cross_3d, scatter_add

# a piece of the boundary of an overlap: contact of each piece, its start and end point, and the plane it lies on
_Pieces = Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]


@dataclass(frozen=True, eq=False)
class Overlap:
    """Overlap region of k pairs of bodies.

    Attributes:
        measure: area (2D) or volume (3D) of the overlap, zero where the bodies do not overlap, shape (k,) [m^2 or
            m^3].
        centroid: centroid of the overlap, shape (k, d) [m].
        surface: contact surface, pointing from body j towards body i, shape (k, d): in 2D the chord across the
            overlap, whose length is that of the contact line [m], and in 3D the area vector of the surface of body j
            inside body i, whose length is its area [m^2].
    """

    measure: npt.NDArray[np.float64]
    centroid: npt.NDArray[np.float64]
    surface: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Polyhedra:
    """A batch of k convex polyhedra in the space frame, one per contact.

    Attributes:
        vertices: vertices, padded with copies of one of them, shape (k, v, 3) [m].
        normals: outward unit normal of each face, zero for padding, shape (k, f, 3).
        offsets: offset of each face plane, n . x <= offset inside, one for padding, shape (k, f) [m].
        edges: vertices of each edge, in the order in which its left face runs along it, padded with (0, 0), shape
            (k, e, 2).
        edge_faces: face on the left and on the right of each edge, shape (k, e, 2).
        faces: vertices of each face in order, padded with the last; padding faces are all zero, shape (k, f, m).
    """

    vertices: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.float64]
    edges: npt.NDArray[np.int64]
    edge_faces: npt.NDArray[np.int64]
    faces: npt.NDArray[np.int64]


def _dots(normals: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Dot products of sets of plane normals with one point each.

    Stacked matrix products are several times faster than ``einsum`` on the arrays of a contact batch.

    Args:
        normals: normals, shape (..., p, d).
        points: one point per set of normals, shape (..., d).

    Returns:
        The dot product of every normal with its point, shape (..., p).
    """
    xp = array_module(normals)
    return xp.asarray((normals @ points[..., None])[..., 0])


def _heights(vertices: npt.NDArray[np.float64], normals: npt.NDArray[np.float64],
             offsets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Signed distances of the vertices of bodies from the planes of other bodies, positive beyond them.

    Args:
        vertices: vertices of each body, shape (k, v, 3) [m].
        normals: unit normals of the planes of the other body, shape (k, p, 3).
        offsets: offsets of those planes, shape (k, p) [m].

    Returns:
        The distances, shape (k, v, p) [m].
    """
    xp = array_module(vertices)
    return xp.asarray(vertices @ normals.transpose(0, 2, 1) - offsets[:, None, :])


def clip_lines(origin: npt.NDArray[np.float64], direction: npt.NDArray[np.float64], normals: npt.NDArray[np.float64],
               offsets: npt.NDArray[np.float64], low: float,
               high: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Clip lines x = origin + t direction to the intersection of half-spaces n . x <= h (Cyrus-Beck).

    Args:
        origin: a point of each line, shape (..., d) [m].
        direction: direction of each line, shape (..., d).
        normals: normals of the half-spaces that clip each line, shape (..., p, d).
        offsets: offsets of those half-spaces, shape (..., p) [m].
        low: smallest t of every line, 0 for a segment from origin to origin + direction and -inf for a whole line.
        high: largest t of every line, 1 for a segment and inf for a whole line.

    Returns:
        The values of t where the part of each line inside starts and ends, and whether that part is not empty.
    """
    xp = array_module(normals)
    start = _dots(normals, origin) - offsets
    rate = _dots(normals, direction)
    with quiet(xp):
        crossing = -start / rate
    # a line enters a half-space it runs into against its normal, and leaves one it runs out of along it
    t0 = xp.maximum(xp.max(xp.where(rate < 0.0, crossing, -np.inf), axis=-1), low)
    t1 = xp.minimum(xp.min(xp.where(rate > 0.0, crossing, np.inf), axis=-1), high)
    outside = xp.any((rate == 0.0) & (start > 0.0), axis=-1)
    return t0, t1, (t1 > t0) & ~outside


def _grown(vertices: npt.NDArray[np.float64], normals: npt.NDArray[np.float64],
           offsets: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Grow convex bodies by a tiny factor about a point inside them.

    Then no face of another body lies in the plane of one of their faces by accident.

    Args:
        vertices: vertices of each body, shape (k, v, d) [m].
        normals: outward unit normals of the faces, zero for padding, shape (k, f, d).
        offsets: offsets of the face planes, shape (k, f) [m].

    Returns:
        The grown vertices and plane offsets [m].
    """
    # the mean of the vertices lies inside a convex body, also when some are repeated as padding
    centre = vertices.mean(axis=1)
    support = _dots(normals, centre)
    factor = 1.0 + OVERLAP_TIE_BREAK
    return centre[:, None] + factor * (vertices - centre[:, None]), support + factor * (offsets - support)


def _inflated(region_offsets: npt.NDArray[np.float64], vertices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Push the planes of regions out by a tiny distance against the size of the bodies they meet.

    Args:
        region_offsets: offsets of the half-spaces of each region, shape (k, r) [m].
        vertices: vertices of the body each region meets, shape (k, v, d) [m].

    Returns:
        The pushed offsets [m].
    """
    xp = array_module(vertices)
    size = xp.max(xp.linalg.norm(vertices - vertices[:, :1], axis=2), axis=1)
    return xp.asarray(region_offsets + (OVERLAP_TIE_BREAK * size)[:, None])


def _cross_2d(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Out-of-plane component of the cross product of vectors along the last axis."""
    xp = array_module(a)
    return xp.asarray(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])


def _segment_parts(vertices: npt.NDArray[np.float64], normals: npt.NDArray[np.float64],
                   offsets: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Parts of the edges of polygons that lie inside sets of half-planes.

    Args:
        vertices: vertices of each polygon, closed by padding with the first, shape (k, v, 2) [m].
        normals: normals of the half-planes of each polygon, shape (k, p, 2).
        offsets: offsets of those half-planes, shape (k, p) [m].

    Returns:
        The start and the end of the part of each edge inside, both zero for an edge that lies outside, shape
        (k, v, 2) each [m].
    """
    xp = array_module(vertices)
    edge = xp.roll(vertices, -1, axis=1) - vertices
    t0, t1, inside = clip_lines(vertices, edge, normals[:, None], offsets[:, None], 0.0, 1.0)
    start = xp.where(inside[..., None], vertices + t0[..., None] * edge, 0.0)
    end = xp.where(inside[..., None], vertices + t1[..., None] * edge, 0.0)
    return start, end


def _planar_overlap(
    origin: npt.NDArray[np.float64], parts: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Area and centroid of planar regions from the counter-clockwise pieces of their boundaries.

    Args:
        origin: reference point of each region, against which the pieces are given, shape (k, 2) [m].
        parts: start and end points of pieces of the boundaries, shape (k, ..., 2) each [m].

    Returns:
        The area of each region [m^2] and its centroid [m].
    """
    xp = array_module(origin)
    area = xp.zeros(origin.shape[0])
    first = xp.zeros_like(origin)
    for start, end in parts:
        twice = _cross_2d(start, end).reshape(origin.shape[0], -1)
        area += 0.5 * twice.sum(axis=1)
        first += (twice[..., None] * (start + end).reshape(origin.shape[0], -1, 2)).sum(axis=1) / 6.0
    centroid = origin + first / xp.where(area > 0.0, area, 1.0)[:, None]
    return xp.maximum(area, 0.0), centroid


def polygon_overlap(vertices_i: npt.NDArray[np.float64], normals_i: npt.NDArray[np.float64],
                    offsets_i: npt.NDArray[np.float64], vertices_j: npt.NDArray[np.float64],
                    normals_j: npt.NDArray[np.float64], offsets_j: npt.NDArray[np.float64]) -> Overlap:
    """Overlap of k pairs of convex polygons i and j.

    Args:
        vertices_i: vertices of each polygon i, counter-clockwise and closed by padding with the first, shape
            (k, v, 2) [m].
        normals_i: outward unit normal of each edge of polygon i, zero for padding, shape (k, v, 2).
        offsets_i: offset of each edge line of polygon i, one for padding, shape (k, v) [m].
        vertices_j: vertices of each polygon j, shape (k, w, 2) [m].
        normals_j: outward unit normals of polygon j, shape (k, w, 2).
        offsets_j: edge line offsets of polygon j, shape (k, w) [m].

    Returns:
        The overlap of each pair.
    """
    xp = array_module(vertices_i)
    vertices_j, offsets_j = _grown(vertices_j, normals_j, offsets_j)
    # coordinates relative to a point near both polygons keep the products accurate
    origin = 0.5 * (vertices_i[:, 0] + vertices_j[:, 0])
    vi, vj = vertices_i - origin[:, None], vertices_j - origin[:, None]
    hi = offsets_i - _dots(normals_i, origin)
    hj = offsets_j - _dots(normals_j, origin)
    start_i, end_i = _segment_parts(vi, normals_j, hj)
    start_j, end_j = _segment_parts(vj, normals_i, hi)
    area, centroid = _planar_overlap(origin, [(start_i, end_i), (start_j, end_j)])
    # the outward normal of a counter-clockwise edge is the edge turned clockwise; both outlines give the chord
    chord_j, chord_i = (end_j - start_j).sum(axis=1), (end_i - start_i).sum(axis=1)
    surface = 0.5 * xp.column_stack((chord_j[:, 1] - chord_i[:, 1], chord_i[:, 0] - chord_j[:, 0]))
    return Overlap(area, centroid, surface)


def polygon_region_overlap(vertices: npt.NDArray[np.float64], normals: npt.NDArray[np.float64],
                           offsets: npt.NDArray[np.float64], region_normals: npt.NDArray[np.float64],
                           region_offsets: npt.NDArray[np.float64]) -> Overlap:
    """Overlap of k convex polygons with k convex regions of the plane given as half-planes.

    The contact surface is the part of the line of the first half-plane of each region inside its polygon, the face
    of a wall the region lies behind.

    Args:
        vertices: vertices of each polygon, counter-clockwise and closed by padding with the first, shape (k, v, 2)
            [m].
        normals: outward unit normal of each edge, zero for padding, shape (k, v, 2).
        offsets: offset of each edge line, one for padding, shape (k, v) [m].
        region_normals: outward unit normal of each half-plane of each region, shape (k, r, 2).
        region_offsets: offset of each half-plane, shape (k, r) [m].

    Returns:
        The overlap of each polygon with its region.
    """
    xp = array_module(vertices)
    region_offsets = _inflated(region_offsets, vertices)
    origin = vertices[:, 0]
    v = vertices - origin[:, None]
    h = offsets - _dots(normals, origin)
    hr = region_offsets - _dots(region_normals, origin)
    parts = [_segment_parts(v, region_normals, hr)]
    planes = xp.concatenate((region_normals, normals), axis=1)
    planes_offsets = xp.concatenate((hr, h), axis=1)
    cap = xp.zeros(vertices.shape[0])
    for r in range(region_normals.shape[1]):
        normal = region_normals[:, r]
        # the counter-clockwise boundary runs along a line with the outward normal on its right
        direction = xp.column_stack((-normal[:, 1], normal[:, 0]))
        clip_normals, clip_offsets = planes.copy(), planes_offsets.copy()
        clip_normals[:, r], clip_offsets[:, r] = 0.0, 1.0
        t0, t1, inside = clip_lines(hr[:, r, None] * normal, direction, clip_normals, clip_offsets, -np.inf, np.inf)
        inside &= xp.isfinite(t0) & xp.isfinite(t1)
        start = xp.where(inside[:, None], hr[:, r, None] * normal + t0[:, None] * direction, 0.0)
        end = xp.where(inside[:, None], hr[:, r, None] * normal + t1[:, None] * direction, 0.0)
        parts.append((start, end))
        if r == 0:
            cap = xp.where(inside, t1 - t0, 0.0)
    area, centroid = _planar_overlap(origin, parts)
    return Overlap(area, centroid, cap[:, None] * region_normals[:, 0])


def _edge_pieces(vertices: npt.NDArray[np.float64], edges: npt.NDArray[np.int64], edge_faces: npt.NDArray[np.int64],
                 distances: npt.NDArray[np.float64], normals: npt.NDArray[np.float64], offsets: npt.NDArray[np.float64],
                 face_shift: int) -> List[_Pieces]:
    """Parts of the edges of polyhedra that lie inside other convex bodies, as pieces of the faces they bound.

    Args:
        vertices: vertices of each polyhedron, shape (k, v, 3) [m].
        edges: vertices of each edge, shape (k, e, 2).
        edge_faces: left and right face of each edge, shape (k, e, 2).
        distances: signed distance of each vertex from each plane of the other body, shape (k, v, p) [m].
        normals: normals of the planes of the other body, shape (k, p, 3).
        offsets: offsets of those planes, shape (k, p) [m].
        face_shift: index of the first face of the polyhedron among the planes of the pair.

    Returns:
        The pieces on the left faces, and those on the right faces, which run the other way.
    """
    xp = array_module(vertices)
    pair_index = xp.arange(vertices.shape[0])[:, None]
    head = distances[pair_index, edges[..., 0]]
    tail = distances[pair_index, edges[..., 1]]
    # an edge with both ends beyond one plane of the other body lies outside it; padding edges have zero length
    candidate = ~xp.any((head > 0.0) & (tail > 0.0), axis=2) & (edges[..., 0] != edges[..., 1])
    pair, edge = xp.nonzero(candidate)
    start = vertices[pair, edges[pair, edge, 0]]
    step = vertices[pair, edges[pair, edge, 1]] - start
    t0, t1, inside = clip_lines(start, step, normals[pair], offsets[pair], 0.0, 1.0)
    pair, edge, start, step, t0, t1 = pair[inside], edge[inside], start[inside], step[inside], t0[inside], t1[inside]
    p = start + t0[:, None] * step
    q = start + t1[:, None] * step
    faces = edge_faces[pair, edge] + face_shift
    return [(pair, p, q, faces[:, 0]), (pair, q, p, faces[:, 1])]


def _line_pieces(pair: npt.NDArray[np.int64], first: npt.NDArray[np.int64], second: npt.NDArray[np.int64],
                 normals: npt.NDArray[np.float64], offsets: npt.NDArray[np.float64]) -> List[_Pieces]:
    """Edges of overlaps along the lines where two of the planes of a pair meet, clipped to all the others.

    A face with outward normal n_a meets the plane of normal n_b along a line, and the counter-clockwise boundary
    of the part of the face inside that plane runs along n_a x n_b; the part of the other face runs the other way.

    Args:
        pair: contact of each line, shape (t,).
        first: index of the first plane among the planes of the pair, shape (t,).
        second: index of the second plane, shape (t,).
        normals: planes of both bodies of each pair, shape (k, p, 3).
        offsets: offsets of those planes, shape (k, p) [m].

    Returns:
        The pieces on the first planes, and those on the second planes.
    """
    xp = array_module(normals)
    na, nb = normals[pair, first], normals[pair, second]
    ha, hb = offsets[pair, first], offsets[pair, second]
    cosine = xp.einsum("ij,ij->i", na, nb)
    sine_sq = 1.0 - cosine * cosine
    meet = sine_sq > PARALLEL_PLANES_TOLERANCE
    pair, first, second = pair[meet], first[meet], second[meet]
    na, nb, ha, hb, cosine, sine_sq = na[meet], nb[meet], ha[meet], hb[meet], cosine[meet], sine_sq[meet]
    # the point of the line closest to the reference point is a combination of the two normals
    origin = (((ha - hb * cosine) / sine_sq)[:, None] * na + ((hb - ha * cosine) / sine_sq)[:, None] * nb)
    direction = cross_3d(na, nb)
    clip_normals, clip_offsets = normals[pair], offsets[pair]
    line = xp.arange(pair.size)
    # the line lies on its own two planes, which must not clip it
    for index in (first, second):
        clip_normals[line, index] = 0.0
        clip_offsets[line, index] = 1.0
    t0, t1, inside = clip_lines(origin, direction, clip_normals, clip_offsets, -np.inf, np.inf)
    inside &= xp.isfinite(t0) & xp.isfinite(t1)
    p = origin[inside] + t0[inside, None] * direction[inside]
    q = origin[inside] + t1[inside, None] * direction[inside]
    return [(pair[inside], p, q, first[inside]), (pair[inside], q, p, second[inside])]


def _straddling(faces: npt.NDArray[np.int64], distances: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Whether each face of a polyhedron crosses each plane of another body, and can reach that body at all.

    Args:
        faces: vertices of each face, padded with the last, shape (k, f, m).
        distances: signed distance of each vertex from each plane of the other body, shape (k, v, p) [m].

    Returns:
        True where a face has vertices on both sides of a plane and lies beyond no plane of the other body as a
        whole, shape (k, f, p).
    """
    xp = array_module(distances)
    pair_index = xp.arange(faces.shape[0])[:, None]
    low = xp.full((faces.shape[0], faces.shape[1], distances.shape[2]), np.inf)
    high = xp.full_like(low, -np.inf)
    # one corner at a time keeps the memory to that of a face-plane table
    for corner in range(faces.shape[2]):
        value = distances[pair_index, faces[:, :, corner]]
        low = xp.minimum(low, value)
        high = xp.maximum(high, value)
    # a face beyond one plane of the other body as a whole meets none of its faces
    reaches = ~xp.any(low > 0.0, axis=2)
    return xp.asarray((low < 0.0) & (high > 0.0) & reaches[:, :, None])


def _solid_overlap(count: int, origin: npt.NDArray[np.float64], normals: npt.NDArray[np.float64],
                   offsets: npt.NDArray[np.float64], pieces: List[_Pieces],
                   surface_planes: npt.NDArray[np.bool_]) -> Overlap:
    """Volume, centroid and contact surface of overlaps from the pieces of their boundaries.

    The piece from p to q on the plane n . x = h, relative to the reference point, spans with the foot h n of the
    reference point on that plane and the reference point itself a tetrahedron of signed volume h n . (p x q) / 6.

    Args:
        count: number of contacts.
        origin: reference point of each contact, against which everything is given, shape (count, 3) [m].
        normals: planes of both bodies of each contact, shape (count, p, 3).
        offsets: offsets of those planes, shape (count, p) [m].
        pieces: pieces of the boundaries, each counter-clockwise about the outward normal of its plane.
        surface_planes: planes whose pieces make up the contact surface, shape (p,).

    Returns:
        The overlaps.
    """
    xp = array_module(origin)
    pair = xp.concatenate([piece[0] for piece in pieces])
    p = xp.concatenate([piece[1] for piece in pieces])
    q = xp.concatenate([piece[2] for piece in pieces])
    plane = xp.concatenate([piece[3] for piece in pieces])
    normal = normals[pair, plane]
    height = offsets[pair, plane]
    moment = cross_3d(p, q)
    volume = height * xp.einsum("ij,ij->i", normal, moment) / 6.0
    measure = xp.bincount(pair, volume, count)
    first = scatter_add(pair, volume[:, None] * (height[:, None] * normal + p + q) / 4.0, count)
    on_surface = surface_planes[plane]
    surface = scatter_add(pair[on_surface], 0.5 * moment[on_surface], count)
    centroid = origin + first / xp.where(measure > 0.0, measure, 1.0)[:, None]
    return Overlap(xp.maximum(measure, 0.0), centroid, surface)


def _empty_overlap(count: int, dimension: int, xp: ModuleType) -> Overlap:
    """Overlaps of bodies that do not overlap, in the memory of the device of an array module."""
    return Overlap(xp.zeros(count), xp.zeros((count, dimension)), xp.zeros((count, dimension)))


def _expand(overlap: Overlap, live: npt.NDArray[np.int64], count: int) -> Overlap:
    """Place the overlaps of a subset of the contacts among all of them, the others not overlapping.

    Args:
        overlap: overlaps of the subset.
        live: contacts of the subset, shape (l,).
        count: number of contacts.

    Returns:
        The overlaps of all the contacts.
    """
    full = _empty_overlap(count, overlap.centroid.shape[1], array_module(live))
    full.measure[live] = overlap.measure
    full.centroid[live] = overlap.centroid
    full.surface[live] = overlap.surface
    return full


def _relative(bodies: Polyhedra,
              origin: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vertices and plane offsets of polyhedra relative to a reference point each.

    Args:
        bodies: the polyhedra.
        origin: reference point of each, shape (k, 3) [m].

    Returns:
        The relative vertices, shape (k, v, 3), and plane offsets, shape (k, f) [m].
    """
    return bodies.vertices - origin[:, None], bodies.offsets - _dots(bodies.normals, origin)


def polyhedron_overlap(i: Polyhedra, j: Polyhedra) -> Overlap:
    """Overlap of k pairs of convex polyhedra i and j.

    Pairs that a face plane of one body separates from the other are skipped, and so are the edges and the face
    pairs that cannot bound the overlap: an edge with both ends beyond one plane of the other body, and two faces
    that do not each cross the plane of the other.

    Args:
        i: the first body of each pair.
        j: the second body of each pair.

    Returns:
        The overlap of each pair.
    """
    xp = array_module(i.vertices)
    count = i.vertices.shape[0]
    grown_vertices, grown_offsets = _grown(j.vertices, j.normals, j.offsets)
    j = replace(j, vertices=grown_vertices, offsets=grown_offsets)
    origin = 0.5 * (i.vertices[:, 0] + j.vertices[:, 0])
    vi, hi = _relative(i, origin)
    vj, hj = _relative(j, origin)
    beyond_j = _heights(vi, j.normals, hj)
    beyond_i = _heights(vj, i.normals, hi)
    apart = xp.any(xp.all(beyond_j > 0.0, axis=1), axis=1) | xp.any(xp.all(beyond_i > 0.0, axis=1), axis=1)
    live = xp.flatnonzero(~apart)
    if live.size == 0:
        return _empty_overlap(count, 3, xp)

    faces_i = i.normals.shape[1]
    normals = xp.concatenate((i.normals[live], j.normals[live]), axis=1)
    offsets = xp.concatenate((hi[live], hj[live]), axis=1)
    pieces = _edge_pieces(vi[live], i.edges[live], i.edge_faces[live], beyond_j[live], j.normals[live], hj[live], 0)
    pieces += _edge_pieces(vj[live], j.edges[live], j.edge_faces[live], beyond_i[live], i.normals[live], hi[live],
                           faces_i)
    meeting = _straddling(i.faces[live], beyond_j[live]) & _straddling(j.faces[live], beyond_i[live]).transpose(0, 2, 1)
    pair, face_i, face_j = xp.nonzero(meeting)
    pieces += _line_pieces(pair, face_i, faces_i + face_j, normals, offsets)
    surface_planes = xp.arange(normals.shape[1]) >= faces_i
    return _expand(_solid_overlap(live.size, origin[live], normals, offsets, pieces, surface_planes), live, count)


def polyhedron_region_overlap(bodies: Polyhedra, region_normals: npt.NDArray[np.float64],
                              region_offsets: npt.NDArray[np.float64]) -> Overlap:
    """Overlap of k convex polyhedra with k convex regions of space given as half-spaces.

    The contact surface is the part of the plane of the first half-space of each region inside its polyhedron, the
    face of a wall the region lies behind.

    Args:
        bodies: the polyhedra.
        region_normals: outward unit normal of each half-space of each region, shape (k, r, 3).
        region_offsets: offset of each half-space, shape (k, r) [m].

    Returns:
        The overlap of each polyhedron with its region.
    """
    xp = array_module(bodies.vertices)
    count = bodies.vertices.shape[0]
    region_offsets = _inflated(region_offsets, bodies.vertices)
    origin = bodies.vertices[:, 0]
    v, h = _relative(bodies, origin)
    hr = region_offsets - _dots(region_normals, origin)
    beyond = _heights(v, region_normals, hr)
    live = xp.flatnonzero(~xp.any(xp.all(beyond > 0.0, axis=1), axis=1))
    if live.size == 0:
        return _empty_overlap(count, 3, xp)

    faces = bodies.normals.shape[1]
    planes = region_normals.shape[1]
    normals = xp.concatenate((bodies.normals[live], region_normals[live]), axis=1)
    offsets = xp.concatenate((h[live], hr[live]), axis=1)
    pieces = _edge_pieces(v[live], bodies.edges[live], bodies.edge_faces[live], beyond[live], region_normals[live],
                          hr[live], 0)
    # the planes of the region are unbounded, so only the faces of the polyhedron can be tested for a crossing
    pair, face, plane = xp.nonzero(_straddling(bodies.faces[live], beyond[live]))
    pieces += _line_pieces(pair, face, faces + plane, normals, offsets)
    # the region has no edges of its own: they lie where two of its planes meet
    first, second = xp.triu_indices(planes, 1)
    pair = xp.repeat(xp.arange(live.size), first.size)
    pieces += _line_pieces(pair, faces + xp.tile(first, live.size), faces + xp.tile(second, live.size), normals,
                           offsets)
    surface_planes = xp.arange(normals.shape[1]) == faces
    return _expand(_solid_overlap(live.size, origin[live], normals, offsets, pieces, surface_planes), live, count)


def polygon_distance(points: npt.NDArray[np.float64], vertices: npt.NDArray[np.float64],
                     normals: npt.NDArray[np.float64],
                     offsets: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Signed distance from points to convex polygons, and the direction away from them.

    Args:
        points: one point per polygon, shape (k, 2) [m].
        vertices: vertices of each polygon, closed by padding with the first, shape (k, v, 2) [m].
        normals: outward unit normal of each edge, zero for padding, shape (k, v, 2).
        offsets: offset of each edge line, one for padding, shape (k, v) [m].

    Returns:
        The distance from each point to its polygon, negative inside [m], and the unit normal at the closest point
        of the outline, pointing towards the point, shape (k, 2).
    """
    xp = array_module(points)
    count = points.shape[0]
    rows = xp.arange(count)
    height = _dots(normals, points) - offsets
    deepest = xp.argmax(height, axis=1)
    inside = height[rows, deepest] <= 0.0
    edge = xp.roll(vertices, -1, axis=1) - vertices
    relative = points[:, None, :] - vertices
    length_sq = xp.einsum("kvd,kvd->kv", edge, edge)
    along = xp.clip(xp.einsum("kvd,kvd->kv", relative, edge) / xp.where(length_sq > 0.0, length_sq, 1.0), 0.0, 1.0)
    offset = relative - along[..., None] * edge
    gap_sq = xp.einsum("kvd,kvd->kv", offset, offset)
    closest = xp.argmin(gap_sq, axis=1)
    gap = xp.sqrt(gap_sq[rows, closest])
    away = offset[rows, closest] / xp.where(gap > 0.0, gap, 1.0)[:, None]
    distance = xp.where(inside, height[rows, deepest], gap)
    normal = xp.where(inside[:, None], normals[rows, deepest], away)
    return distance, normal


def polyhedron_distance(points: npt.NDArray[np.float64],
                        bodies: Polyhedra) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Signed distance from points to convex polyhedra, and the direction away from them.

    Outside a polyhedron the closest point lies inside a face whose plane the point is beyond, or on an edge.

    Args:
        points: one point per polyhedron, shape (k, 3) [m].
        bodies: the polyhedra.

    Returns:
        The distance from each point to its polyhedron, negative inside [m], and the unit normal at the closest point
        of the surface, pointing towards the point, shape (k, 3).
    """
    xp = array_module(points)
    count = points.shape[0]
    rows = xp.arange(count)
    height = _dots(bodies.normals, points) - bodies.offsets
    deepest = xp.argmax(height, axis=1)
    inside = height[rows, deepest] <= 0.0

    # the foot of the point on a face plane, if it lies inside the face
    foot = points[:, None, :] - height[..., None] * bodies.normals
    corners = bodies.vertices[rows[:, None, None], bodies.faces]
    sides = xp.roll(corners, -1, axis=2) - corners
    turn = xp.einsum("kfmd,kfd->kfm", cross_3d(sides, foot[:, :, None, :] - corners), bodies.normals)
    on_face = (height > 0.0) & xp.all(turn >= 0.0, axis=2)
    face_gap = xp.where(on_face, height, np.inf)
    best_face = xp.argmin(face_gap, axis=1)

    start = bodies.vertices[rows[:, None], bodies.edges[..., 0]]
    edge = bodies.vertices[rows[:, None], bodies.edges[..., 1]] - start
    relative = points[:, None, :] - start
    length_sq = xp.einsum("ked,ked->ke", edge, edge)
    along = xp.clip(xp.einsum("ked,ked->ke", relative, edge) / xp.where(length_sq > 0.0, length_sq, 1.0), 0.0, 1.0)
    offset = relative - along[..., None] * edge
    gap_sq = xp.einsum("ked,ked->ke", offset, offset)
    best_edge = xp.argmin(gap_sq, axis=1)
    edge_gap = xp.sqrt(gap_sq[rows, best_edge])

    use_face = face_gap[rows, best_face] <= edge_gap
    gap = xp.where(use_face, face_gap[rows, best_face], edge_gap)
    away = xp.where(use_face[:, None], bodies.normals[rows, best_face],
                    offset[rows, best_edge] / xp.where(edge_gap > 0.0, edge_gap, 1.0)[:, None])
    distance = xp.where(inside, height[rows, deepest], gap)
    normal = xp.where(inside[:, None], bodies.normals[rows, deepest], away)
    return distance, normal
