"""Particle definitions and a lattice generator for an initial loose packing.

A particle is a sphere, or carries a convex shape, a polygon in 2D and a polyhedron in 3D, scaled to its radius (see
``DEMcookies.shapes``). The generator places particles on a jittered, non-overlapping lattice inside the domain. When
the solver starts they fall under gravity and compact into a dense random packing. For filling a domain completely,
the wave-by-wave raining of ``DEMSimulation.rain_until_full`` is usually more convenient.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

from .geometry import Geometry
from .geometry3d import Domain, Geometry3D
from .polytopes import identity_orientations, random_orientations
from .shapes import ConvexPolygon, ConvexPolyhedron, Shape


@dataclass
class Particle:
    """A single particle of a 2D simulation, used to build an initial configuration.

    Attributes:
        x: x coordinate of the centre, the centroid of a polygon [m].
        y: y coordinate of the centre [m].
        radius: radius [m]; of a polygon, the equivalent radius, that of the disc of equal area.
        vx: x component of the velocity [m/s].
        vy: y component of the velocity [m/s].
        shape: convex polygon that the particle has, scaled to its radius; None for a sphere.
        angle: counter-clockwise rotation of the shape from its body frame [rad].
    """

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    shape: Optional[ConvexPolygon] = None
    angle: float = 0.0

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 2."""
        return 2

    @property
    def position(self) -> Tuple[float, ...]:
        """Coordinates (x, y) of the centre [m]."""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, ...]:
        """Velocity (vx, vy) [m/s]."""
        return (self.vx, self.vy)

    @property
    def orientation(self) -> Tuple[float, ...]:
        """Orientation of the particle, its angle as a one-element tuple [rad]."""
        return (self.angle, )


@dataclass
class Particle3D:
    """A single particle of a 3D simulation, used to build an initial configuration.

    Attributes:
        x: x coordinate of the centre, the centroid of a polyhedron [m].
        y: y coordinate of the centre, along the vertical [m].
        z: z coordinate of the centre [m].
        radius: radius [m]; of a polyhedron, the equivalent radius, that of the sphere of equal volume.
        vx: x component of the velocity [m/s].
        vy: y component of the velocity [m/s].
        vz: z component of the velocity [m/s].
        shape: convex polyhedron that the particle has, scaled to its radius; None for a sphere.
        orientation: rotation of the shape from its body frame, as a quaternion (w, x, y, z); it is normalised, so
            only its direction matters.
    """

    x: float
    y: float
    z: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    shape: Optional[ConvexPolyhedron] = None
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def dimension(self) -> int:
        """Number of space dimensions, 3."""
        return 3

    @property
    def position(self) -> Tuple[float, ...]:
        """Coordinates (x, y, z) of the centre [m]."""
        return (self.x, self.y, self.z)

    @property
    def velocity(self) -> Tuple[float, ...]:
        """Velocity (vx, vy, vz) [m/s]."""
        return (self.vx, self.vy, self.vz)


AnyParticle = Union[Particle, Particle3D]


def check_shapes(shapes: Sequence[Optional[Shape]], dimension: int) -> None:
    """Check that shapes suit the number of dimensions of a simulation.

    Args:
        shapes: shapes, None for a sphere.
        dimension: number of space dimensions, 2 or 3.

    Raises:
        ValueError: if no shape is given.
        ValueError: if a shape is a polyhedron in 2D or a polygon in 3D.
    """
    if len(shapes) == 0:
        raise ValueError("shapes must not be empty; use None for spheres.")
    for k, shape in enumerate(shapes):
        if shape is not None and shape.dimension != dimension:
            kind = "polygons" if dimension == 2 else "polyhedra"
            raise ValueError(f"Shape {k} is a {type(shape).__name__}, but a {dimension}D simulation needs {kind}.")


def bounding_ratio(shapes: Sequence[Optional[Shape]]) -> float:
    """Largest distance from the centre to a point of a particle of any of the shapes, over its radius.

    Args:
        shapes: shapes, None for a sphere.

    Returns:
        The ratio, one for spheres.
    """
    return max(1.0 if shape is None else shape.circumradius for shape in shapes)


@overload
def generate_lattice_particles(geometry: Geometry,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = ...,
                               spacing_factor: float = ...,
                               jitter: float = ...,
                               shapes: Optional[Sequence[Optional[Shape]]] = ...) -> List[Particle]:
    ...


@overload
def generate_lattice_particles(geometry: Geometry3D,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = ...,
                               spacing_factor: float = ...,
                               jitter: float = ...,
                               shapes: Optional[Sequence[Optional[Shape]]] = ...) -> List[Particle3D]:
    ...


def generate_lattice_particles(geometry: Domain,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = 0,
                               spacing_factor: float = 1.15,
                               jitter: float = 0.25,
                               shapes: Optional[Sequence[Optional[Shape]]] = None) -> Sequence[AnyParticle]:
    """Place particles on a loose lattice inside the domain, filling the lowest rows (or layers, in 3D) first.

    Args:
        geometry: domain to fill, a polygon or a polyhedron.
        n_particles: number of particles to create.
        radius_range: smallest and largest radius of the uniformly distributed radii [m].
        seed: seed of the random generator, for reproducible packings.
        spacing_factor: lattice spacing as a multiple of the largest diameter; values above one leave room for the
            jitter.
        jitter: random displacement of each particle as a fraction of the free space between lattice sites, which
            breaks the lattice symmetry.
        shapes: shapes to draw from, polygons in 2D and polyhedra in 3D, each particle taking one at random, in a
            random orientation; None in the list stands for a sphere. None makes every particle a sphere.

    Returns:
        The particles, ``Particle`` in a polygon and ``Particle3D`` in a polyhedron.

    Raises:
        ValueError: if the radius range is not 0 < r_min <= r_max.
        ValueError: if the spacing factor is below one.
        ValueError: if the jitter is not in [0, 1].
        ValueError: if the list of shapes is empty.
        ValueError: if a shape does not suit the number of dimensions of the domain.
        ValueError: if fewer lattice sites fit in the domain than particles are requested.
    """
    r_min, r_max = radius_range
    if not 0.0 < r_min <= r_max:
        raise ValueError(f"radius_range must satisfy 0 < r_min <= r_max, got {radius_range}.")
    if spacing_factor < 1.0:
        raise ValueError(f"spacing_factor must be at least 1, got {spacing_factor}.")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError(f"jitter must be in [0, 1], got {jitter}.")

    dimension = geometry.dimension
    if shapes is not None:
        check_shapes(shapes, dimension)
    rng = np.random.default_rng(seed)
    # a shaped particle reaches farther from its centre than its radius
    reach = r_max * (1.0 if shapes is None else bounding_ratio(shapes))
    spacing = 2.0 * reach * spacing_factor
    # neighbours can each move by this much along a lattice direction and still not overlap
    jitter_amplitude = jitter * 0.5 * (spacing - 2.0 * reach)
    # the jitter acts on every coordinate, so a site can move diagonally by sqrt(d) times the amplitude
    clearance = reach + math.sqrt(dimension) * jitter_amplitude

    low, high = geometry.bounds
    axes = [np.arange(low[k] + clearance, high[k] - clearance + 0.5 * spacing, spacing) for k in range(dimension)]
    # the vertical axis varies slowest, so that the sites are ordered from the bottom up
    order = [1, 0] if dimension == 2 else [1, 2, 0]
    grids = np.meshgrid(*[axes[k] for k in order], indexing="ij")
    sites = np.column_stack([grids[order.index(k)].ravel() for k in range(dimension)])
    fits = geometry.contains(sites) & (geometry.distance_to_facets(sites).min(axis=1) >= clearance)
    sites = sites[fits]
    if sites.shape[0] < n_particles:
        raise ValueError(f"Only {sites.shape[0]} lattice sites fit in the domain, but {n_particles} particles were "
                         f"requested.")

    sites = sites[:n_particles]
    radii = rng.uniform(r_min, r_max, n_particles)
    offsets = rng.uniform(-jitter_amplitude, jitter_amplitude, (n_particles, dimension))
    centres = sites + offsets
    # the shapes are drawn last, so that a packing of spheres is the same as without them
    chosen: List[Optional[Shape]] = [None] * n_particles
    orientations = identity_orientations(n_particles, dimension)
    if shapes is not None:
        chosen = [shapes[k] for k in rng.integers(len(shapes), size=n_particles)]
        orientations = random_orientations(rng, n_particles, dimension)
    if dimension == 2:
        return [
            Particle(float(x), float(y), float(r), shape=_polygon(shape), angle=float(orientations[k]))
            for k, ((x, y), r, shape) in enumerate(zip(centres, radii, chosen))
        ]
    return [
        Particle3D(float(x),
                   float(y),
                   float(z),
                   float(r),
                   shape=_polyhedron(shape),
                   orientation=_quaternion(orientations[k]))
        for k, ((x, y, z), r, shape) in enumerate(zip(centres, radii, chosen))
    ]


def _polygon(shape: Optional[Shape]) -> Optional[ConvexPolygon]:
    """Return a checked shape as the polygon it is, or None for a sphere."""
    return shape if isinstance(shape, ConvexPolygon) else None


def _polyhedron(shape: Optional[Shape]) -> Optional[ConvexPolyhedron]:
    """Return a checked shape as the polyhedron it is, or None for a sphere."""
    return shape if isinstance(shape, ConvexPolyhedron) else None


def _quaternion(values: npt.NDArray[np.float64]) -> Tuple[float, float, float, float]:
    """Return the four components of a quaternion as a tuple."""
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
