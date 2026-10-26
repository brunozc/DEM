"""Bookkeeping of the shapes of the particles of a simulation, and the rotations of rigid bodies.

A ``PolytopeTable`` holds the shape of every particle in its body frame, scaled to its radius, in arrays padded to
common sizes so that a batch of contacts can gather the shapes of its particles at once. A sphere has an empty row.

An orientation is an angle about the out-of-plane axis in 2D, and a unit quaternion (w, x, y, z) in 3D, which turns
the body frame into the space frame: a point at r in the body frame is at x + R r in space.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .globals import SPHERE_INERTIA_FACTOR
from .shapes import ConvexPolyhedron, Shape
from .vectors import cross_3d


def _pad(array: npt.NDArray, width: int, axis: int, mode: str, value: float = 0.0) -> npt.NDArray:
    """Pad an array along one axis to a given width.

    Args:
        array: array to pad.
        width: width of the padded axis, at least its current width.
        axis: axis to pad.
        mode: "edge" repeats the last entry, "first" repeats the first entry and "constant" fills with a value.
        value: value of the "constant" padding.

    Returns:
        The padded array.
    """
    missing = width - array.shape[axis]
    if missing <= 0:
        return array
    if array.shape[axis] == 0 or mode == "constant":
        shape = list(array.shape)
        shape[axis] = missing
        filler = np.full(shape, value, dtype=array.dtype)
    else:
        index = 0 if mode == "first" else -1
        filler = np.repeat(np.take(array, [index], axis=axis), missing, axis=axis)
    return np.concatenate((array, filler), axis=axis)


@dataclass(frozen=True, eq=False)
class PolytopeTable:
    """Shapes of the particles of a simulation, in their body frames and scaled to their radii.

    Row k belongs to particle k. The arrays are never changed: adding or removing particles makes a new table, so a
    recorded frame can keep a table without copying it. A sphere has no vertices, and only padding faces, whose
    planes have a zero normal and a unit offset, so that every point lies inside them and they cut nothing.

    Attributes:
        dimension: number of space dimensions, 2 or 3.
        vertex_count: number of vertices of each particle, zero for a sphere, shape (n,).
        vertices: vertices in the body frame, padded with copies of the first, shape (n, v, d) [m]. Padding a polygon
            with its first vertex closes it: edge k runs from vertex k to vertex k + 1, and the padding edges have
            zero length.
        normals: outward unit normal of each face (of each edge in 2D, edge k being face k), zero for padding, shape
            (n, f, d).
        offsets: distance from the centroid to the plane of each face, one for padding, shape (n, f) [m].
        edges: vertices of each edge of a polyhedron, in the order in which its left face runs along it, padded with
            (0, 0), shape (n, e, 2); empty in 2D.
        edge_faces: face on the left and on the right of each edge of a polyhedron, shape (n, e, 2); empty in 2D.
        faces: vertices of each face of a polyhedron in order, padded with the last; padding faces are all zero, shape
            (n, f, k); empty in 2D.
        inertia_factor: moment of inertia over m R^2 in 2D, shape (n,), and principal moments over m R^2 in 3D,
            shape (n, 3); 2/5 for a sphere.
        reach: distance from the centroid to the farthest point: the radius of a sphere and the circumradius of a
            polytope, shape (n,) [m].
        inradius: distance from the centroid to the closest face plane: the radius of a sphere, shape (n,) [m].
    """

    dimension: int
    vertex_count: npt.NDArray[np.int64]
    vertices: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.float64]
    edges: npt.NDArray[np.int64]
    edge_faces: npt.NDArray[np.int64]
    faces: npt.NDArray[np.int64]
    inertia_factor: npt.NDArray[np.float64]
    reach: npt.NDArray[np.float64]
    inradius: npt.NDArray[np.float64]

    @classmethod
    def build(cls, dimension: int, shapes: Sequence[Optional[Shape]],
              radii: npt.NDArray[np.float64]) -> "PolytopeTable":
        """Make the table of a set of particles.

        Args:
            dimension: number of space dimensions, 2 or 3.
            shapes: shape of each particle, None for a sphere.
            radii: radius of each particle [m].

        Returns:
            The table.
        """
        n = len(shapes)
        polytopes = [(k, shape) for k, shape in enumerate(shapes) if shape is not None]
        v = max([len(shape.vertices) for _, shape in polytopes], default=0)
        f = max([len(shape.normals) for _, shape in polytopes], default=0)
        e, m = 0, 0
        if dimension == 3:
            solids = [shape for _, shape in polytopes if isinstance(shape, ConvexPolyhedron)]
            e = max([len(shape.edges) for shape in solids], default=0)
            m = max([len(face) for shape in solids for face in shape.faces], default=0)
        vertex_count = np.zeros(n, dtype=np.int64)
        vertices = np.zeros((n, v, dimension))
        normals = np.zeros((n, f, dimension))
        offsets = np.ones((n, f))
        edges = np.zeros((n, e, 2), dtype=np.int64)
        edge_faces = np.zeros((n, e, 2), dtype=np.int64)
        faces = np.zeros((n, f if dimension == 3 else 0, m), dtype=np.int64)
        factor_shape = (n, ) if dimension == 2 else (n, 3)
        inertia_factor = np.full(factor_shape, SPHERE_INERTIA_FACTOR)
        reach = np.array(radii, dtype=np.float64)
        inradius = np.array(radii, dtype=np.float64)
        for k, shape in polytopes:
            radius = float(radii[k])
            count = len(shape.vertices)
            vertex_count[k] = count
            vertices[k] = _pad(radius * shape.vertices, v, 0, "first")
            normals[k, :len(shape.normals)] = shape.normals
            offsets[k, :len(shape.offsets)] = radius * shape.offsets
            inertia_factor[k] = shape.inertia_factor
            reach[k] = radius * shape.circumradius
            inradius[k] = radius * shape.inradius
            if isinstance(shape, ConvexPolyhedron):
                edges[k, :len(shape.edges)] = shape.edges
                edge_faces[k, :len(shape.edges)] = shape.edge_faces
                for index, face in enumerate(shape.faces):
                    faces[k, index] = list(face) + [face[-1]] * (m - len(face))
        return cls(dimension, vertex_count, vertices, normals, offsets, edges, edge_faces, faces, inertia_factor, reach,
                   inradius)

    @property
    def polytope(self) -> npt.NDArray[np.bool_]:
        """Whether each particle is a polygon or a polyhedron rather than a sphere, shape (n,)."""
        return np.asarray(self.vertex_count > 0)

    def concatenate(self, other: "PolytopeTable") -> "PolytopeTable":
        """Return the table of the particles of this table followed by those of another.

        Args:
            other: table of the particles to append.

        Returns:
            The joined table, padded to the larger of the two sizes.
        """
        v = max(self.vertices.shape[1], other.vertices.shape[1])
        f = max(self.normals.shape[1], other.normals.shape[1])
        e = max(self.edges.shape[1], other.edges.shape[1])
        m = max(self.faces.shape[2], other.faces.shape[2])
        parts = [self, other]
        return PolytopeTable(self.dimension, np.concatenate([p.vertex_count for p in parts]),
                             np.concatenate([_pad(p.vertices, v, 1, "first") for p in parts]),
                             np.concatenate([_pad(p.normals, f, 1, "constant") for p in parts]),
                             np.concatenate([_pad(p.offsets, f, 1, "constant", 1.0) for p in parts]),
                             np.concatenate([_pad(p.edges, e, 1, "constant") for p in parts]),
                             np.concatenate([_pad(p.edge_faces, e, 1, "constant") for p in parts]),
                             np.concatenate([_pad(_pad(p.faces, f, 1, "constant"), m, 2, "edge") for p in parts]),
                             np.concatenate([p.inertia_factor for p in parts]),
                             np.concatenate([p.reach for p in parts]), np.concatenate([p.inradius for p in parts]))

    def take(self, keep: npt.NDArray[np.bool_]) -> "PolytopeTable":
        """Return the table of a subset of the particles.

        Args:
            keep: True for every particle to keep.

        Returns:
            The table of the kept particles.
        """
        return PolytopeTable(self.dimension, self.vertex_count[keep], self.vertices[keep], self.normals[keep],
                             self.offsets[keep], self.edges[keep], self.edge_faces[keep], self.faces[keep],
                             self.inertia_factor[keep], self.reach[keep], self.inradius[keep])


@dataclass(frozen=True, eq=False)
class PolytopePose:
    """Shapes of a set of polytope particles in the space frame, at their current position and orientation.

    Attributes:
        rows: particle of each entry, shape (p,).
        rotation: rotation matrix of each particle, shape (p, d, d).
        vertices: vertices, shape (p, v, d) [m].
        normals: outward unit normals of the faces, zero for padding, shape (p, f, d).
        offsets: offsets of the face planes, n . x <= offset inside, one for padding, shape (p, f) [m].
    """

    rows: npt.NDArray[np.int64]
    rotation: npt.NDArray[np.float64]
    vertices: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.float64]


def place(table: PolytopeTable, rows: npt.NDArray[np.int64], positions: npt.NDArray[np.float64],
          orientations: npt.NDArray[np.float64]) -> PolytopePose:
    """Move the shapes of some particles into the space frame.

    Args:
        table: shapes of all the particles.
        rows: particles to place, shape (p,).
        positions: centroids of all the particles, shape (n, d) [m].
        orientations: orientations of all the particles, shape (n,) in 2D or (n, 4) in 3D.

    Returns:
        The placed shapes.
    """
    rotation = rotation_matrices(orientations[rows])
    centre = positions[rows]
    vertices = np.einsum("pij,pvj->pvi", rotation, table.vertices[rows]) + centre[:, None, :]
    normals = np.einsum("pij,pfj->pfi", rotation, table.normals[rows])
    # a padding face keeps its zero normal and unit offset, so it still cuts nothing
    offsets = table.offsets[rows] + np.einsum("pfi,pi->pf", normals, centre)
    return PolytopePose(rows, rotation, vertices, normals, offsets)


def rotation_matrices(orientations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rotation matrices of a set of orientations.

    Args:
        orientations: angles, shape (p,), in 2D, or unit quaternions (w, x, y, z), shape (p, 4), in 3D.

    Returns:
        The matrices that turn the body frame into the space frame, shape (p, 2, 2) or (p, 3, 3).
    """
    if orientations.ndim == 1:
        c, s = np.cos(orientations), np.sin(orientations)
        planar = np.empty((orientations.size, 2, 2))
        planar[:, 0, 0], planar[:, 0, 1], planar[:, 1, 0], planar[:, 1, 1] = c, -s, s, c
        return planar
    w, x, y, z = orientations.T
    matrix = np.empty((orientations.shape[0], 3, 3))
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - w * z)
    matrix[:, 0, 2] = 2.0 * (x * z + w * y)
    matrix[:, 1, 0] = 2.0 * (x * y + w * z)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - w * x)
    matrix[:, 2, 0] = 2.0 * (x * z - w * y)
    matrix[:, 2, 1] = 2.0 * (y * z + w * x)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def rotate(orientations: npt.NDArray[np.float64], rotation_vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Turn orientations further by rotations given in the space frame.

    Args:
        orientations: angles, shape (p,), or unit quaternions, shape (p, 4).
        rotation_vectors: rotations, angles about the out-of-plane axis, shape (p,), or rotation vectors (axis times
            angle) in the space frame, shape (p, 3) [rad].

    Returns:
        The new orientations, quaternions normalised against drift.
    """
    if orientations.ndim == 1:
        return np.asarray(orientations + rotation_vectors)
    angle = np.sqrt(np.einsum("ij,ij->i", rotation_vectors, rotation_vectors))
    # sin(a/2)/a tends to 1/2 for small angles, where the quotient would lose its accuracy
    half = 0.5 * angle
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(angle > 1.0e-8, np.sin(half) / np.where(angle > 0.0, angle, 1.0), 0.5 - angle * angle / 48.0)
    w1, v1 = np.cos(half), ratio[:, None] * rotation_vectors
    w2, v2 = orientations[:, 0], orientations[:, 1:]
    # the turn acts after the current orientation, so its quaternion multiplies from the left
    w = w1 * w2 - np.einsum("ij,ij->i", v1, v2)
    v = w1[:, None] * v2 + w2[:, None] * v1 + cross_3d(v1, v2)
    quaternion = np.column_stack((w, v))
    return np.asarray(quaternion / np.linalg.norm(quaternion, axis=1)[:, None])


def apply_inertia(rotation: npt.NDArray[np.float64], principal: npt.NDArray[np.float64],
                  vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiply space-frame vectors by the inertia tensors R diag(I) R^T of bodies, or by its inverse.

    Args:
        rotation: rotation matrices of the bodies, shape (p, 3, 3).
        principal: principal moments of inertia, or their inverses, shape (p, 3).
        vectors: vectors in the space frame, e.g. angular velocities, shape (p, 3).

    Returns:
        The products, e.g. the angular momenta, shape (p, 3).
    """
    body = np.einsum("pji,pj->pi", rotation, vectors)
    return np.asarray(np.einsum("pij,pj->pi", rotation, principal * body))


def random_orientations(rng: np.random.Generator, count: int, dimension: int) -> npt.NDArray[np.float64]:
    """Draw orientations uniformly distributed over all rotations.

    Args:
        rng: random generator.
        count: number of orientations.
        dimension: number of space dimensions, 2 or 3.

    Returns:
        Angles in [0, 2 pi), shape (count,), or unit quaternions, shape (count, 4).
    """
    if dimension == 2:
        return np.asarray(rng.uniform(0.0, 2.0 * math.pi, count))
    # a normalised four-dimensional Gaussian is uniform on the sphere of unit quaternions
    quaternions = rng.normal(size=(count, 4))
    return np.asarray(quaternions / np.linalg.norm(quaternions, axis=1)[:, None])


def identity_orientations(count: int, dimension: int) -> npt.NDArray[np.float64]:
    """Orientations that leave the body frame aligned with the space frame.

    Args:
        count: number of orientations.
        dimension: number of space dimensions, 2 or 3.

    Returns:
        Zero angles, shape (count,), or identity quaternions, shape (count, 4).
    """
    if dimension == 2:
        return np.zeros(count)
    identity = np.zeros((count, 4))
    identity[:, 0] = 1.0
    return identity


def frame_polygons(table: PolytopeTable, positions: npt.NDArray[np.float64],
                   orientations: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Vertices of the polytope particles of a state in the space frame, e.g. to draw them.

    Args:
        table: shapes of the particles.
        positions: centroids, shape (n, d) [m].
        orientations: orientations, shape (n,) or (n, 4).

    Returns:
        The polytope particles, shape (p,), and their vertices, padded with copies of the first, shape (p, v, d) [m].
    """
    rows = np.flatnonzero(table.polytope)
    if rows.size == 0:
        return rows, np.zeros((0, 0, table.dimension))
    return rows, place(table, rows, positions, orientations).vertices
