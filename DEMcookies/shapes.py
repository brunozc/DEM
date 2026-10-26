"""Shapes of non-spherical particles: convex polygons in 2D and convex polyhedra in 3D.

A shape sets the form of a particle and its radius the size: a particle of radius R carries its shape scaled so that
its equivalent radius is R. The equivalent radius is that of the sphere of equal volume in 3D, and that of the disc of
equal area in 2D, where a polygon stands for the mid-section of a grain as a disc stands for that of a sphere. A
particle therefore has the mass of the sphere of its radius whatever its shape, so replacing spheres by shapes keeps
the grading of a packing, its buoyancy and the drag of the water on it.

A shape is stored in its body frame, scaled to an equivalent radius of one: the centroid lies at the origin and, in
3D, the principal axes of inertia lie along x, y and z, so that the inertia tensor is diagonal. The vertices of a
polygon run counter-clockwise, and those of every face of a polyhedron counter-clockwise seen from outside.
"""

import itertools
import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .globals import GEOMETRY_TOLERANCE, MIN_CENTRE_CLEARANCE, QUERY_CHUNK_SIZE

# points a shape is made from: a sequence of coordinate tuples, or an array of shape (n, d)
Points = Union[Sequence[Sequence[float]], npt.NDArray[np.float64]]


def _polygon_hull(points: npt.NDArray[np.float64], tolerance: float) -> List[int]:
    """Indices of the vertices of the convex hull of points in a plane, counter-clockwise (Andrew's monotone chain).

    Points on an edge of the hull are left out, so consecutive hull edges are never collinear.

    Args:
        points: points, shape (n, 2) [m].
        tolerance: largest twice-area of a triangle of three points that are taken as collinear [m^2].

    Returns:
        The indices of the hull vertices.
    """
    order = sorted(range(points.shape[0]), key=lambda k: (points[k, 0], points[k, 1]))

    def turn(o: int, a: int, b: int) -> float:
        return float((points[a, 0] - points[o, 0]) * (points[b, 1] - points[o, 1]) - (points[a, 1] - points[o, 1]) *
                     (points[b, 0] - points[o, 0]))

    lower: List[int] = []
    for k in order:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], k) <= tolerance:
            lower.pop()
        lower.append(k)
    upper: List[int] = []
    for k in reversed(order):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], k) <= tolerance:
            upper.pop()
        upper.append(k)
    return lower[:-1] + upper[:-1]


def _polygon_properties(vertices: npt.NDArray[np.float64]) -> Tuple[float, npt.NDArray[np.float64], float]:
    """Area, centroid and polar second moment of area of a counter-clockwise polygon.

    Args:
        vertices: polygon vertices, shape (n, 2) [m].

    Returns:
        The area [m^2], the centroid [m] and the polar second moment of area about the centroid [m^4].
    """
    # coordinates relative to a vertex keep the products small
    origin = vertices[0]
    a = vertices - origin
    b = np.roll(a, -1, axis=0)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    area = 0.5 * float(cross.sum())
    centroid = (cross[:, None] * (a + b)).sum(axis=0) / (6.0 * area)
    # second moments of area about the origin vertex, then moved to the centroid
    xx = float((cross * (a[:, 0]**2 + a[:, 0] * b[:, 0] + b[:, 0]**2)).sum()) / 12.0
    yy = float((cross * (a[:, 1]**2 + a[:, 1] * b[:, 1] + b[:, 1]**2)).sum()) / 12.0
    polar = xx + yy - area * float(centroid @ centroid)
    return area, origin + centroid, polar


def _as_points(vertices: Points, dimension: int) -> npt.NDArray[np.float64]:
    """Check and convert the points a shape is made from.

    Args:
        vertices: points with ``dimension`` coordinates each.
        dimension: number of space dimensions, 2 or 3.

    Returns:
        The points, shape (n, dimension).

    Raises:
        ValueError: if the points do not have ``dimension`` coordinates each.
        ValueError: if fewer than dimension + 1 points are given.
        ValueError: if a coordinate is not finite.
    """
    names = "(x, y)" if dimension == 2 else "(x, y, z)"
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(f"vertices must be a sequence of {names} points.")
    if points.shape[0] < dimension + 1:
        raise ValueError(f"A {'polygon' if dimension == 2 else 'polyhedron'} needs at least {dimension + 1} points, "
                         f"got {points.shape[0]}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Vertex coordinates must be finite.")
    return points


class ConvexPolygon:
    """Convex polygonal shape of a 2D particle.

    A planar particle stands for the mid-section of a grain, as a disc stands for that of a sphere: its mass is that
    of the sphere of its radius, (4/3) pi rho R^3, and its moment of inertia is 4/5 of that of a prism with its
    section and that mass, which gives (2/5) m R^2 for a disc, as for a sphere.
    """

    def __init__(self, vertices: Points) -> None:
        """Make the shape, the convex hull of a set of points.

        Args:
            vertices: points (x, y) whose convex hull is the shape, at any scale and position [m].

        Raises:
            ValueError: if the points are not a sequence of (x, y) pairs.
            ValueError: if fewer than three points are given.
            ValueError: if a coordinate is not finite.
            ValueError: if the points are collinear, so their hull has no area.
        """
        points = _as_points(vertices, 2)
        size = float(np.max(np.ptp(points, axis=0)))
        hull = _polygon_hull(points, GEOMETRY_TOLERANCE * size * size) if size > 0.0 else []
        if len(hull) < 3:
            raise ValueError("The points are collinear; a polygon needs an area.")
        area, centroid, polar = _polygon_properties(points[hull])
        radius = math.sqrt(area / math.pi)
        self.__equivalent_radius = radius
        self.__vertices = (points[hull] - centroid) / radius
        self.__vertices.setflags(write=False)
        edges = np.roll(self.__vertices, -1, axis=0) - self.__vertices
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        # counter-clockwise edges have their outward normal on their right
        self.__normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
        self.__normals.setflags(write=False)
        self.__offsets = np.einsum("ij,ij->i", self.__normals, self.__vertices)
        self.__offsets.setflags(write=False)
        self.__inertia_factor = 0.8 * polar / (area * radius * radius)

    @classmethod
    def regular(cls, sides: int) -> "ConvexPolygon":
        """Make a regular polygon, with a vertex on the positive x axis.

        Args:
            sides: number of sides.

        Returns:
            The shape.

        Raises:
            ValueError: if the polygon has fewer than three sides.
        """
        if sides < 3:
            raise ValueError(f"A polygon needs at least three sides, got {sides}.")
        angles = 2.0 * math.pi * np.arange(sides) / sides
        return cls(np.column_stack((np.cos(angles), np.sin(angles))))

    @classmethod
    def rectangle(cls, width: float, height: float) -> "ConvexPolygon":
        """Make a rectangle, with its sides along the axes.

        Args:
            width: extent along x; only the ratio of width to height matters [m].
            height: extent along y [m].

        Returns:
            The shape.

        Raises:
            ValueError: if the width or the height is not positive.
        """
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Rectangle width and height must be positive, got {width} and {height}.")
        return cls([(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)])

    @classmethod
    def random(cls, vertices: int, seed: Union[int, np.random.Generator] = 0, aspect: float = 1.0) -> "ConvexPolygon":
        """Make an irregular polygon, the hull of random points on an ellipse, e.g. for an angular sand grain.

        Args:
            vertices: number of points, and so of vertices.
            seed: seed of the random generator, or the generator itself.
            aspect: ratio of the x to the y semi-axis of the ellipse; values away from one elongate the grain.

        Returns:
            The shape.

        Raises:
            ValueError: if fewer than three vertices are asked for.
            ValueError: if the aspect ratio is not positive.
        """
        if vertices < 3:
            raise ValueError(f"A polygon needs at least three vertices, got {vertices}.")
        if aspect <= 0.0:
            raise ValueError(f"aspect must be positive, got {aspect}.")
        rng = np.random.default_rng(seed)
        while True:
            angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, vertices))
            # a draw that crowds every point into one half of the ellipse makes a sliver; draw again
            gaps = np.diff(np.append(angles, angles[0] + 2.0 * math.pi))
            if gaps.max() < math.pi:
                return cls(np.column_stack((aspect * np.cos(angles), np.sin(angles))))

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 2."""
        return 2

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Vertices in the body frame at an equivalent radius of one, counter-clockwise, shape (v, 2)."""
        return self.__vertices

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Outward unit normal of the edge from vertex k to vertex k + 1, shape (v, 2)."""
        return self.__normals

    @property
    def offsets(self) -> npt.NDArray[np.float64]:
        """Distance from the centroid to the line of each edge, at an equivalent radius of one, shape (v,)."""
        return self.__offsets

    @property
    def equivalent_radius(self) -> float:
        """Equivalent radius of the points the shape was made from, sqrt(A / pi) [m].

        A particle of this radius has the shape at the size of those points.
        """
        return self.__equivalent_radius

    @property
    def inertia_factor(self) -> float:
        """Moment of inertia of a particle about its centroid over m R^2; 2/5 for a disc."""
        return self.__inertia_factor

    @property
    def circumradius(self) -> float:
        """Distance from the centroid to the farthest vertex, at an equivalent radius of one."""
        return float(np.max(np.hypot(self.__vertices[:, 0], self.__vertices[:, 1])))

    @property
    def inradius(self) -> float:
        """Distance from the centroid to the closest edge line, at an equivalent radius of one."""
        return float(self.__offsets.min())


class ConvexPolyhedron:
    """Convex polyhedral shape of a 3D particle."""

    def __init__(self, vertices: Points) -> None:
        """Make the shape, the convex hull of a set of points.

        Coplanar faces of the hull are merged into one polygon, so a box has six faces, not twelve triangles.

        Args:
            vertices: points (x, y, z) whose convex hull is the shape, at any scale and position [m].

        Raises:
            ValueError: if the points are not a sequence of (x, y, z) points.
            ValueError: if fewer than four points are given.
            ValueError: if a coordinate is not finite.
            ValueError: if the points are coplanar, so their hull has no volume.
        """
        points = _as_points(vertices, 3)
        size = float(np.max(np.ptp(points, axis=0)))
        faces = _polyhedron_hull(points, GEOMETRY_TOLERANCE * size) if size > 0.0 else []
        if len(faces) < 4:
            raise ValueError("The points are coplanar; a polyhedron needs a volume.")
        used = sorted({k for face in faces for k in face})
        renumber = {old: new for new, old in enumerate(used)}
        points = points[used]
        faces = [[renumber[k] for k in face] for face in faces]

        volume, centroid, second = _polyhedron_moments(points, faces)
        moments, axes = np.linalg.eigh(second)
        # a proper rotation, so that the body frame stays right-handed
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] = -axes[:, 2]
        radius = (3.0 * volume / (4.0 * math.pi))**(1.0 / 3.0)
        self.__equivalent_radius = radius
        self.__vertices = ((points - centroid) @ axes) / radius
        self.__faces = tuple(tuple(face) for face in faces)
        # moment of inertia about each principal axis, from the second moments about the other two
        self.__inertia_factor = (np.trace(second) - moments) / (volume * radius * radius)

        normals = np.zeros((len(faces), 3))
        for k, face in enumerate(faces):
            corners = self.__vertices[list(face)]
            area_vector = 0.5 * np.cross(corners - corners[0], np.roll(corners, -1, axis=0) - corners[0]).sum(axis=0)
            normals[k] = area_vector / np.linalg.norm(area_vector)
        self.__normals = normals
        self.__offsets = np.array([normals[k] @ self.__vertices[face[0]] for k, face in enumerate(faces)])
        self.__edges, self.__edge_faces = _edges_of(self.__faces)
        for array in (self.__vertices, self.__inertia_factor, self.__normals, self.__offsets, self.__edges,
                      self.__edge_faces):
            array.setflags(write=False)

    @classmethod
    def box(cls, width: float, height: float, depth: float) -> "ConvexPolyhedron":
        """Make a rectangular box, with its edges along the axes.

        Args:
            width: extent along x; only the ratios of the extents matter [m].
            height: extent along y [m].
            depth: extent along z [m].

        Returns:
            The shape.

        Raises:
            ValueError: if an extent is not positive.
        """
        if min(width, height, depth) <= 0.0:
            raise ValueError(f"Box extents must be positive, got {width}, {height} and {depth}.")
        return cls(list(itertools.product((0.0, width), (0.0, height), (0.0, depth))))

    @classmethod
    def tetrahedron(cls) -> "ConvexPolyhedron":
        """Make a regular tetrahedron.

        Returns:
            The shape.
        """
        return cls([(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)])

    @classmethod
    def octahedron(cls) -> "ConvexPolyhedron":
        """Make a regular octahedron.

        Returns:
            The shape.
        """
        return cls(np.vstack((np.eye(3), -np.eye(3))))

    @classmethod
    def random(
        cls,
        vertices: int,
        seed: Union[int, np.random.Generator] = 0,
        aspect: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> "ConvexPolyhedron":
        """Make an irregular polyhedron, the hull of random points on an ellipsoid, e.g. for a ballast stone.

        Args:
            vertices: number of points, and so of vertices.
            seed: seed of the random generator, or the generator itself.
            aspect: semi-axes of the ellipsoid along x, y and z; only their ratios matter, and unequal ones make
                elongated or flaky grains.

        Returns:
            The shape.

        Raises:
            ValueError: if fewer than four vertices are asked for.
            ValueError: if a semi-axis is not positive.
        """
        if vertices < 4:
            raise ValueError(f"A polyhedron needs at least four vertices, got {vertices}.")
        if min(aspect) <= 0.0:
            raise ValueError(f"aspect must be positive, got {aspect}.")
        rng = np.random.default_rng(seed)
        while True:
            directions = rng.normal(size=(vertices, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            # points crowded to one side of the centre make a sliver; draw again
            if _centre_clearance(directions) > MIN_CENTRE_CLEARANCE:
                return cls(directions * np.asarray(aspect, dtype=np.float64))

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 3."""
        return 3

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Vertices in the body frame at an equivalent radius of one, shape (v, 3)."""
        return self.__vertices

    @property
    def faces(self) -> Tuple[Tuple[int, ...], ...]:
        """Vertex indices of each face, counter-clockwise seen from outside."""
        return self.__faces

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Outward unit normal of each face, shape (f, 3)."""
        return self.__normals

    @property
    def offsets(self) -> npt.NDArray[np.float64]:
        """Distance from the centroid to the plane of each face, at an equivalent radius of one, shape (f,)."""
        return self.__offsets

    @property
    def edges(self) -> npt.NDArray[np.int64]:
        """Vertex indices of each edge, in the order in which its left face runs along it, shape (e, 2)."""
        return self.__edges

    @property
    def edge_faces(self) -> npt.NDArray[np.int64]:
        """Face on the left and face on the right of each edge, seen from outside along it, shape (e, 2)."""
        return self.__edge_faces

    @property
    def equivalent_radius(self) -> float:
        """Equivalent radius of the points the shape was made from, (3 V / (4 pi))^(1/3) [m].

        A particle of this radius has the shape at the size of those points.
        """
        return self.__equivalent_radius

    @property
    def inertia_factor(self) -> npt.NDArray[np.float64]:
        """Principal moments of inertia of a particle about x, y and z over m R^2, shape (3,); 2/5 for a sphere."""
        return self.__inertia_factor

    @property
    def circumradius(self) -> float:
        """Distance from the centroid to the farthest vertex, at an equivalent radius of one."""
        return float(np.max(np.linalg.norm(self.__vertices, axis=1)))

    @property
    def inradius(self) -> float:
        """Distance from the centroid to the closest face plane, at an equivalent radius of one."""
        return float(self.__offsets.min())


Shape = Union[ConvexPolygon, ConvexPolyhedron]


def _centre_clearance(points: npt.NDArray[np.float64]) -> float:
    """Smallest distance from the origin to the plane of a face of the hull of points.

    Args:
        points: points, shape (n, 3) [m].

    Returns:
        The distance, negative if the origin lies outside the hull [m].
    """
    clearance = math.inf
    for face in _polyhedron_hull(points, GEOMETRY_TOLERANCE):
        corners = points[face]
        # consecutive edges of a face are never collinear, and turn counter-clockwise about its outward normal
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        clearance = min(clearance, float(normal @ corners[0]) / float(np.linalg.norm(normal)))
    return clearance


def _polyhedron_hull(points: npt.NDArray[np.float64], tolerance: float) -> List[List[int]]:
    """Faces of the convex hull of points in space, each counter-clockwise seen from outside.

    Every plane through three of the points that has all the points on one side bounds the hull; the points on it
    make one face, whose outline is their planar hull. This visits every triple of points, which suits the few tens
    of vertices of a particle.

    Args:
        points: points, shape (n, 3) [m].
        tolerance: largest distance from a plane of a point that is taken to lie on it [m].

    Returns:
        The vertex indices of each face.
    """
    n = points.shape[0]
    triples = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
    found: Dict[frozenset, npt.NDArray[np.float64]] = {}
    chunk = max(1, QUERY_CHUNK_SIZE // n)
    for start in range(0, triples.shape[0], chunk):
        block = triples[start:start + chunk]
        a, b, c = points[block[:, 0]], points[block[:, 1]], points[block[:, 2]]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal, axis=1)
        good = length > tolerance * max(float(np.max(np.ptp(points, axis=0))), 1.0e-300)
        normal = normal[good] / length[good, None]
        distance = points @ normal.T - np.einsum("ij,ij->i", normal, a[good])
        outward = np.all(distance <= tolerance, axis=0)
        inward = np.all(distance >= -tolerance, axis=0)
        for k in np.flatnonzero(outward | inward):
            key = frozenset(np.flatnonzero(np.abs(distance[:, k]) <= tolerance).tolist())
            if key not in found:
                found[key] = normal[k] if outward[k] else -normal[k]

    faces = []
    for key, normal in found.items():
        members = np.array(sorted(key), dtype=np.int64)
        # a basis (u, v) of the plane with u x v = n, so that counter-clockwise in it is counter-clockwise about n
        helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
        u = np.cross(normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        local = np.column_stack((points[members] @ u, points[members] @ v))
        outline = _polygon_hull(local, tolerance * float(np.max(np.ptp(local, axis=0))))
        if len(outline) >= 3:
            faces.append([int(members[k]) for k in outline])
    return faces


def _polyhedron_moments(
        points: npt.NDArray[np.float64],
        faces: Sequence[Sequence[int]]) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Volume, centroid and second moments of volume of a closed polyhedron.

    The polyhedron is split into tetrahedra between a reference point and the triangles of a fan over each face.

    Args:
        points: vertices, shape (v, 3) [m].
        faces: vertex indices of each face, counter-clockwise seen from outside.

    Returns:
        The volume [m^3], the centroid [m] and the matrix of second moments of volume about the centroid,
        the integral of (x - c)(x - c)^T [m^5].
    """
    origin = points.mean(axis=0)
    relative = points - origin
    volume = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    for face in faces:
        for k in range(1, len(face) - 1):
            a, b, c = relative[face[0]], relative[face[k]], relative[face[k + 1]]
            tetrahedron = float(a @ np.cross(b, c)) / 6.0
            total = a + b + c
            volume += tetrahedron
            first += tetrahedron * total / 4.0
            second += tetrahedron / 20.0 * (np.outer(a, a) + np.outer(b, b) + np.outer(c, c) + np.outer(total, total))
    centroid = first / volume
    return volume, origin + centroid, second - volume * np.outer(centroid, centroid)


def _edges_of(faces: Sequence[Sequence[int]]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Edges of a closed polyhedron, each with the faces on either side of it.

    Args:
        faces: vertex indices of each face, counter-clockwise seen from outside.

    Returns:
        The two vertices of each edge, in the order in which its left face runs along it, shape (e, 2), and its left
        and right face, shape (e, 2).

    Raises:
        ValueError: if an edge does not have exactly one face on either side, so the hull is not closed.
    """
    owner: Dict[Tuple[int, int], int] = {}
    for f, face in enumerate(faces):
        for a, b in zip(face, list(face[1:]) + [face[0]]):
            owner[(a, b)] = f
    edges, sides = [], []
    for (a, b), left in owner.items():
        if (b, a) not in owner:
            raise ValueError("The hull of the points is not closed; perturb nearly coplanar points.")
        if a < b:
            edges.append((a, b))
            sides.append((left, owner[(b, a)]))
    return np.array(edges, dtype=np.int64), np.array(sides, dtype=np.int64)
