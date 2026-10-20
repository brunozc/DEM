"""Particle definitions and a lattice generator for an initial loose packing.

The generator places particles on a jittered, non-overlapping lattice inside the domain. When the solver starts
they fall under gravity and compact into a dense random packing. For filling a domain completely, the wave-by-wave
raining of ``DEMSimulation.rain_until_full`` is usually more convenient.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, overload

import numpy as np

from .geometry import Geometry
from .geometry3d import Domain, Geometry3D


@dataclass
class Particle:
    """A single spherical particle of a 2D simulation, used to build an initial configuration.

    Attributes:
        x: x coordinate of the centre [m].
        y: y coordinate of the centre [m].
        radius: radius [m].
        vx: x component of the velocity [m/s].
        vy: y component of the velocity [m/s].
    """

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0

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


@dataclass
class Particle3D:
    """A single spherical particle of a 3D simulation, used to build an initial configuration.

    Attributes:
        x: x coordinate of the centre [m].
        y: y coordinate of the centre, along the vertical [m].
        z: z coordinate of the centre [m].
        radius: radius [m].
        vx: x component of the velocity [m/s].
        vy: y component of the velocity [m/s].
        vz: z component of the velocity [m/s].
    """

    x: float
    y: float
    z: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

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


@overload
def generate_lattice_particles(geometry: Geometry,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = ...,
                               spacing_factor: float = ...,
                               jitter: float = ...) -> List[Particle]:
    ...


@overload
def generate_lattice_particles(geometry: Geometry3D,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = ...,
                               spacing_factor: float = ...,
                               jitter: float = ...) -> List[Particle3D]:
    ...


def generate_lattice_particles(geometry: Domain,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = 0,
                               spacing_factor: float = 1.15,
                               jitter: float = 0.25) -> Sequence[AnyParticle]:
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

    Returns:
        The particles, ``Particle`` in a polygon and ``Particle3D`` in a polyhedron.

    Raises:
        ValueError: if the radius range is not 0 < r_min <= r_max.
        ValueError: if the spacing factor is below one.
        ValueError: if the jitter is not in [0, 1].
        ValueError: if fewer lattice sites fit in the domain than particles are requested.
    """
    r_min, r_max = radius_range
    if not 0.0 < r_min <= r_max:
        raise ValueError(f"radius_range must satisfy 0 < r_min <= r_max, got {radius_range}.")
    if spacing_factor < 1.0:
        raise ValueError(f"spacing_factor must be at least 1, got {spacing_factor}.")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError(f"jitter must be in [0, 1], got {jitter}.")

    rng = np.random.default_rng(seed)
    dimension = geometry.dimension
    spacing = 2.0 * r_max * spacing_factor
    # neighbours can each move by this much along a lattice direction and still not overlap
    jitter_amplitude = jitter * 0.5 * (spacing - 2.0 * r_max)
    # the jitter acts on every coordinate, so a site can move diagonally by sqrt(d) times the amplitude
    clearance = r_max + math.sqrt(dimension) * jitter_amplitude

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
    if dimension == 2:
        return [Particle(float(x), float(y), float(r)) for (x, y), r in zip(centres, radii)]
    return [Particle3D(float(x), float(y), float(z), float(r)) for (x, y, z), r in zip(centres, radii)]
