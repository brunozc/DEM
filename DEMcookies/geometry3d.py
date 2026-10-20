"""Polyhedral simulation domain whose faces each carry a boundary condition, the 3D counterpart of ``Geometry``.

The vertical is the y axis, as in 2D, and z is the depth: a two-dimensional domain is a slice z = const of a
three-dimensional one. Gravity acts along -y, the water level is a height y, and ``Geometry3D.extrude`` turns a
polygonal domain into a prism whose side faces carry the boundary conditions of the edges they are extruded from.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .boundary import BoundaryCondition, LoadedPlaten, Open, Wall, resolve_body_direction
from .facets import PolygonFacets
from .geometry import Geometry
from .globals import GEOMETRY_TOLERANCE, PLANARITY_TOLERANCE, QUERY_CHUNK_SIZE

Point3D = Tuple[float, float, float]


class Geometry3D:
    """Closed polyhedron whose faces carry boundary conditions.

    Face ``k`` is the planar polygon through the vertices ``faces[k]``, listed in order around it, and carries
    ``boundary_conditions[k]``. The faces must close the polyhedron: every edge is shared by exactly two faces, which
    run along it in opposite directions, so that all faces are oriented alike, all clockwise or all counter-clockwise
    seen from outside; either orientation is accepted. A vertex that lies on an edge of a neighbouring face must
    therefore be one of the vertices of that face too. Faces need not be convex, and consecutive edges of a face may
    be collinear. To apply a load to part of a side only, split that side into several coplanar faces and make one
    of them a ``LoadedPlaten``. Faces that carry the same ``LoadedPlaten`` object form one rigid body.
    """

    def __init__(self, vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
                 boundary_conditions: Sequence[BoundaryCondition]) -> None:
        """Create and validate the domain.

        Args:
            vertices: polyhedron vertices (x, y, z) [m].
            faces: indices of the vertices of each face, in order around it.
            boundary_conditions: one boundary condition per face.

        Raises:
            ValueError: if the vertices are not a sequence of (x, y, z) points.
            ValueError: if fewer than four vertices are given.
            ValueError: if a vertex coordinate is not finite.
            ValueError: if fewer than four faces are given.
            ValueError: if the number of boundary conditions differs from the number of faces.
            ValueError: if a face has fewer than three vertices.
            ValueError: if a face refers to a vertex that does not exist.
            ValueError: if a face lists a vertex twice.
            ValueError: if a face has an edge of zero length.
            ValueError: if a face has zero area.
            ValueError: if a face is not planar.
            ValueError: if an edge belongs to one face only.
            ValueError: if an edge belongs to more than two faces.
            ValueError: if two faces run along their shared edge in the same direction.
            ValueError: if the polyhedron has zero volume.
            ValueError: if two adjacent faces fold back onto each other.
            ValueError: if an edge of the polyhedron pierces a face.
            ValueError: if a loaded body has a direction with other than three components.
            ValueError: if a loaded body without an explicit direction has faces whose inward normals cancel out.
        """
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("vertices must be a sequence of (x, y, z) points.")
        n_vertices = points.shape[0]
        if n_vertices < 4:
            raise ValueError(f"A polyhedron needs at least four vertices, got {n_vertices}.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Vertex coordinates must be finite.")
        face_list = [tuple(int(index) for index in face) for face in faces]
        if len(face_list) < 4:
            raise ValueError(f"A polyhedron needs at least four faces, got {len(face_list)}.")
        if len(boundary_conditions) != len(face_list):
            raise ValueError(f"Expected one boundary condition per face ({len(face_list)}), "
                             f"got {len(boundary_conditions)}.")

        size = float(np.max(np.ptp(points, axis=0)))
        tolerance = GEOMETRY_TOLERANCE * size
        areas = np.zeros(len(face_list))
        area_vectors = np.zeros((len(face_list), 3))
        for k, face in enumerate(face_list):
            area_vectors[k] = _check_face(k, face, points, size)
            areas[k] = float(np.sqrt(area_vectors[k] @ area_vectors[k]))

        self.__vertices = points
        self.__faces: Tuple[Tuple[int, ...], ...] = tuple(face_list)
        self.__boundary_conditions: Tuple[BoundaryCondition, ...] = tuple(boundary_conditions)
        neighbours = self.__check_closed()

        # divergence theorem: the area vectors point outwards if the signed volume is positive
        origins = points[[face[0] for face in face_list]]
        signed_volume = float(np.einsum("ij,ij->", origins, area_vectors)) / 3.0
        if abs(signed_volume) <= GEOMETRY_TOLERANCE * size**3:
            raise ValueError("The polyhedron has zero volume.")
        orientation = -1.0 if signed_volume > 0.0 else 1.0
        self.__volume = abs(signed_volume)
        self.__areas = areas
        self.__normals = orientation * area_vectors / areas[:, None]
        self.__facets = PolygonFacets([points[list(face)] for face in face_list])
        self.__triangles = np.array([[points[face[0]], points[face[k]], points[face[k + 1]]] for face in face_list
                                     for k in range(1,
                                                    len(face) - 1)]).reshape(-1, 3, 3)
        self.__check_simple(neighbours, tolerance)

        bodies: Dict[LoadedPlaten, List[int]] = {}
        for k, condition in enumerate(self.__boundary_conditions):
            if isinstance(condition, LoadedPlaten):
                bodies.setdefault(condition, []).append(k)
        self.__platen_groups = tuple((spec, tuple(indices)) for spec, indices in bodies.items())
        for spec, indices in self.__platen_groups:
            resolve_body_direction(spec, self.__areas[list(indices)], self.__normals[list(indices)])

    @classmethod
    def box(cls,
            width: float,
            height: float,
            depth: float,
            bottom: Optional[BoundaryCondition] = None,
            right: Optional[BoundaryCondition] = None,
            top: Optional[BoundaryCondition] = None,
            left: Optional[BoundaryCondition] = None,
            back: Optional[BoundaryCondition] = None,
            front: Optional[BoundaryCondition] = None,
            origin: Point3D = (0.0, 0.0, 0.0)) -> "Geometry3D":
        """Create a rectangular box, open at the top unless stated otherwise.

        The box spans the width along x, the height along y, which is vertical, and the depth along z. The faces
        are numbered bottom (0), right (1), top (2) and left (3), like the edges of ``Geometry.rectangle``, then back
        (4, at the lowest z) and front (5).

        Args:
            width: width of the box, along x [m].
            height: height of the box, along y [m].
            depth: depth of the box, along z [m].
            bottom: boundary condition of the bottom face; None gives a wall.
            right: boundary condition of the right face; None gives a wall.
            top: boundary condition of the top face; None leaves it open.
            left: boundary condition of the left face; None gives a wall.
            back: boundary condition of the back face; None gives a wall.
            front: boundary condition of the front face; None gives a wall.
            origin: corner of the box with the lowest coordinates [m].

        Returns:
            The box.

        Raises:
            ValueError: if the width, height or depth is not positive.
        """
        if width <= 0.0 or height <= 0.0 or depth <= 0.0:
            raise ValueError(f"Box width, height and depth must be positive, got {width}, {height} and {depth}.")
        x0, y0, z0 = origin
        x1, y1, z1 = x0 + width, y0 + height, z0 + depth
        vertices = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1),
                    (x0, y1, z1)]
        faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (0, 3, 2, 1), (4, 5, 6, 7)]
        conditions: Sequence[BoundaryCondition] = (bottom or Wall(), right or Wall(), top or Open(), left
                                                   or Wall(), back or Wall(), front or Wall())
        return cls(vertices, faces, conditions)

    @classmethod
    def extrude(cls,
                geometry: Geometry,
                depth: float,
                back: Optional[BoundaryCondition] = None,
                front: Optional[BoundaryCondition] = None,
                z_back: float = 0.0) -> "Geometry3D":
        """Extrude a polygonal domain along z into a prism.

        Side face ``k`` is swept by edge ``k`` of the polygon and carries its boundary condition, so the faces of
        the prism are numbered like the edges of the polygon, followed by the back face (at ``z_back``) and the front
        face. A ``LoadedPlaten`` with an explicit 2D direction (x, y) is replaced by a copy with the direction
        (x, y, 0); edges that shared one body still share one.

        Args:
            geometry: polygonal domain in the x-y plane.
            depth: extent of the prism along z [m].
            back: boundary condition of the back face; None gives a wall.
            front: boundary condition of the front face; None gives a wall.
            z_back: z coordinate of the back face [m].

        Returns:
            The prism.

        Raises:
            ValueError: if the depth is not positive.
        """
        if depth <= 0.0:
            raise ValueError(f"Extrusion depth must be positive, got {depth}.")
        outline = geometry.vertices
        m = outline.shape[0]
        vertices = np.vstack((np.column_stack(
            (outline, np.full(m, z_back))), np.column_stack((outline, np.full(m, z_back + depth)))))
        faces: List[Tuple[int, ...]] = [(k, (k + 1) % m, m + (k + 1) % m, m + k) for k in range(m)]
        faces.append(tuple(range(m - 1, -1, -1)))
        faces.append(tuple(range(m, 2 * m)))
        bodies: Dict[LoadedPlaten, LoadedPlaten] = {}
        conditions: List[BoundaryCondition] = []
        for condition in geometry.boundary_conditions:
            if isinstance(condition, LoadedPlaten) and condition.direction is not None:
                if condition not in bodies:
                    bodies[condition] = dataclasses.replace(condition, direction=(*condition.direction, 0.0))
                condition = bodies[condition]
            conditions.append(condition)
        conditions.extend((back or Wall(), front or Wall()))
        return cls(vertices.tolist(), faces, conditions)

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 3."""
        return 3

    @property
    def facet_name(self) -> str:
        """Name of the facets of the domain, which are faces."""
        return "face"

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Polyhedron vertices, shape (n, 3) [m]."""
        return self.__vertices.copy()

    @property
    def faces(self) -> Tuple[Tuple[int, ...], ...]:
        """Indices of the vertices of each face, in order around it."""
        return self.__faces

    @property
    def boundary_conditions(self) -> Tuple[BoundaryCondition, ...]:
        """Boundary condition of each face."""
        return self.__boundary_conditions

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.__faces)

    @property
    def n_facets(self) -> int:
        """Number of facets, i.e. of faces."""
        return self.n_faces

    @property
    def facets(self) -> PolygonFacets:
        """The faces as planar polygons, for the contact queries."""
        return self.__facets

    @property
    def face_areas(self) -> npt.NDArray[np.float64]:
        """Area of each face [m^2]."""
        return self.__areas.copy()

    @property
    def inward_normals(self) -> npt.NDArray[np.float64]:
        """Unit normal of each face pointing into the domain, shape (n, 3)."""
        return self.__normals.copy()

    @property
    def platen_groups(self) -> Tuple[Tuple[LoadedPlaten, Tuple[int, ...]], ...]:
        """Each loaded body with the indices of its faces, ordered by their first face."""
        return self.__platen_groups

    @property
    def volume(self) -> float:
        """Volume enclosed by the polyhedron [m^3]."""
        return self.__volume

    @property
    def bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """Bounds (x_min, y_min, z_min, x_max, y_max, z_max) of the polyhedron [m]."""
        low, high = self.bounds
        return (float(low[0]), float(low[1]), float(low[2]), float(high[0]), float(high[1]), float(high[2]))

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lowest and highest coordinates (x, y, z) of the polyhedron [m]."""
        return self.__vertices.min(axis=0), self.__vertices.max(axis=0)

    def contains(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Check which points lie inside the polyhedron, from its winding number about them.

        The winding number is the solid angle the closed surface subtends at a point, over 4 pi: one inside and
        zero outside, whatever the shape of the polyhedron.

        Args:
            points: points to test, shape (n, 3) [m].

        Returns:
            True for every point inside the polyhedron.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = np.zeros(points.shape[0], dtype=bool)
        triangles = self.__triangles
        chunk = max(1, QUERY_CHUNK_SIZE // triangles.shape[0])
        for begin in range(0, points.shape[0], chunk):
            block = points[begin:begin + chunk, None, :]
            a, b, c = triangles[None, :, 0] - block, triangles[None, :, 1] - block, triangles[None, :, 2] - block
            length_a, length_b, length_c = (np.sqrt(np.einsum("ntk,ntk->nt", v, v)) for v in (a, b, c))
            # solid angle of each triangle (Van Oosterom and Strackee, 1983)
            numerator = np.einsum("ntk,ntk->nt", a, np.cross(b, c))
            denominator = (length_a * length_b * length_c + np.einsum("ntk,ntk->nt", a, b) * length_c +
                           np.einsum("ntk,ntk->nt", a, c) * length_b + np.einsum("ntk,ntk->nt", b, c) * length_a)
            winding = np.sum(2.0 * np.arctan2(numerator, denominator), axis=1) / (4.0 * math.pi)
            inside[begin:begin + chunk] = np.abs(winding) > 0.5
        return inside

    def distance_to_faces(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance from every point to every face.

        Args:
            points: points, shape (n, 3) [m].

        Returns:
            The distances, shape (n, faces) [m].
        """
        return self.__facets.distances(np.asarray(points, dtype=np.float64).reshape(-1, 3))

    def distance_to_facets(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Distance from every point to every facet, i.e. to every face.

        Args:
            points: points, shape (n, 3) [m].

        Returns:
            The distances, shape (n, faces) [m].
        """
        return self.distance_to_faces(points)

    def __check_closed(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Check that every edge is shared by exactly two faces that run along it in opposite directions.

        Returns:
            The two faces of every edge, the edge given by its vertex indices in increasing order.

        Raises:
            ValueError: if an edge belongs to one face only.
            ValueError: if an edge belongs to more than two faces.
            ValueError: if two faces run along their shared edge in the same direction.
        """
        uses: Dict[Tuple[int, int], List[Tuple[int, bool]]] = {}
        for k, face in enumerate(self.__faces):
            for a, b in zip(face, face[1:] + face[:1]):
                uses.setdefault((min(a, b), max(a, b)), []).append((k, a < b))
        neighbours: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for (a, b), users in uses.items():
            if len(users) == 1:
                raise ValueError(f"The polyhedron is not closed: edge ({a}, {b}) of face {users[0][0]} belongs to no "
                                 f"other face. A vertex that lies on an edge of a face must be one of its vertices.")
            if len(users) > 2:
                raise ValueError(f"Edge ({a}, {b}) is shared by more than two faces: "
                                 f"{', '.join(str(k) for k, _ in users)}.")
            (first, forward), (second, backward) = users
            if forward == backward:
                raise ValueError(f"Faces {first} and {second} run along their shared edge ({a}, {b}) in the same "
                                 f"direction; reverse the vertices of one of them so that all faces are oriented "
                                 f"alike.")
            neighbours[(a, b)] = (first, second)
        return neighbours

    def __check_simple(self, neighbours: Dict[Tuple[int, int], Tuple[int, int]], tolerance: float) -> None:
        """Check that adjacent faces do not fold back and that no edge of the polyhedron pierces a face.

        Args:
            neighbours: the two faces of every edge.
            tolerance: distance below which two points are taken to touch [m].

        Raises:
            ValueError: if two adjacent faces fold back onto each other.
            ValueError: if an edge of the polyhedron pierces a face.
        """
        for first, second in neighbours.values():
            a, b = self.__normals[first], self.__normals[second]
            if float(np.linalg.norm(np.cross(a, b))) <= GEOMETRY_TOLERANCE and float(a @ b) < 0.0:
                raise ValueError(f"Faces {first} and {second} fold back onto each other.")

        edges = np.array(list(neighbours.keys()), dtype=np.int64)
        incidence = np.zeros((self.__vertices.shape[0], self.n_faces), dtype=bool)
        for k, face in enumerate(self.__faces):
            incidence[list(face), k] = True
        start, end = self.__vertices[edges[:, 0]], self.__vertices[edges[:, 1]]
        origins = self.__vertices[[face[0] for face in self.__faces]]
        height_start = np.einsum("efk,fk->ef", start[:, None, :] - origins[None, :, :], self.__normals)
        height_end = np.einsum("efk,fk->ef", end[:, None, :] - origins[None, :, :], self.__normals)
        crosses = ((height_start > tolerance) & (height_end < -tolerance)) | \
            ((height_start < -tolerance) & (height_end > tolerance))
        # an edge always meets the faces around its own end points
        crosses &= ~(incidence[edges[:, 0]] | incidence[edges[:, 1]])
        edge, face = np.nonzero(crosses)
        if edge.size == 0:
            return
        fraction = height_start[edge, face] / (height_start[edge, face] - height_end[edge, face])
        crossing = start[edge] + fraction[:, None] * (end[edge] - start[edge])
        offsets = self.__facets.pair_offsets(crossing, face)
        pierced = np.flatnonzero(np.einsum("ij,ij->i", offsets, offsets) <= tolerance**2)
        if pierced.size:
            k = pierced[0]
            a, b = edges[edge[k]]
            raise ValueError(f"The edge between vertices {a} and {b} pierces face {face[k]}; the polyhedron must not "
                             f"intersect itself.")


def _check_face(index: int, face: Tuple[int, ...], points: npt.NDArray[np.float64],
                size: float) -> npt.NDArray[np.float64]:
    """Check one face of a polyhedron and return its area vector.

    Args:
        index: index of the face, for the error messages.
        face: indices of the vertices of the face, in order around it.
        points: vertices of the polyhedron, shape (n, 3) [m].
        size: largest extent of the polyhedron along a coordinate axis [m].

    Returns:
        The area vector of the face, normal to it with its area as length, by the right-hand rule [m^2].

    Raises:
        ValueError: if the face has fewer than three vertices.
        ValueError: if the face refers to a vertex that does not exist.
        ValueError: if the face lists a vertex twice.
        ValueError: if the face has an edge of zero length.
        ValueError: if the face has zero area.
        ValueError: if the face is not planar.
    """
    if len(face) < 3:
        raise ValueError(f"Face {index} needs at least three vertices, got {len(face)}.")
    if min(face) < 0 or max(face) >= points.shape[0]:
        raise ValueError(f"Face {index} refers to a vertex that does not exist; the indices must be in "
                         f"[0, {points.shape[0] - 1}].")
    if len(set(face)) != len(face):
        raise ValueError(f"Face {index} lists a vertex twice.")
    polygon = points[list(face)]
    edges = np.roll(polygon, -1, axis=0) - polygon
    if np.any(np.sqrt(np.einsum("ij,ij->i", edges, edges)) <= GEOMETRY_TOLERANCE * size):
        raise ValueError(f"Face {index} has an edge of zero length.")
    # Newell's area vector, exact for any planar polygon whether convex or not
    area_vector = 0.5 * np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
    area = float(np.sqrt(area_vector @ area_vector))
    if area <= GEOMETRY_TOLERANCE * size**2:
        raise ValueError(f"Face {index} has zero area.")
    heights = np.abs((polygon - polygon.mean(axis=0)) @ (area_vector / area))
    if float(heights.max()) > PLANARITY_TOLERANCE * size:
        worst = int(np.argmax(heights))
        raise ValueError(f"Face {index} is not planar: vertex {face[worst]} lies {float(heights[worst]):.3g} m off "
                         f"its plane.")
    return np.asarray(area_vector)


Domain = Union[Geometry, Geometry3D]
