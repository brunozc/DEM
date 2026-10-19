"""Particle definition and a lattice generator for an initial loose packing.

The generator places particles on a jittered, non-overlapping lattice inside the domain. When the solver starts
they fall under gravity and compact into a dense random packing. For filling a domain completely, the wave-by-wave
raining of ``DEMSimulation.rain_until_full`` is usually more convenient.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .geometry import Geometry


@dataclass
class Particle:
    """A single circular particle, used to build an initial configuration.

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


def generate_lattice_particles(geometry: Geometry,
                               n_particles: int,
                               radius_range: Tuple[float, float],
                               seed: int = 0,
                               spacing_factor: float = 1.15,
                               jitter: float = 0.25) -> List[Particle]:
    """Place particles on a loose lattice inside the domain, filling the lowest rows first.

    Args:
        geometry: domain to fill.
        n_particles: number of particles to create.
        radius_range: smallest and largest radius of the uniformly distributed radii [m].
        seed: seed of the random generator, for reproducible packings.
        spacing_factor: lattice spacing as a multiple of the largest diameter; values above one leave room for the
            jitter.
        jitter: random displacement of each particle as a fraction of the free space between lattice sites, which
            breaks the lattice symmetry.

    Returns:
        The particles.

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
    spacing = 2.0 * r_max * spacing_factor
    # neighbours can each move by this much along a lattice direction and still not overlap
    jitter_amplitude = jitter * 0.5 * (spacing - 2.0 * r_max)
    # the jitter acts on both coordinates, so a site can move diagonally by sqrt(2) times the amplitude
    clearance = r_max + math.sqrt(2.0) * jitter_amplitude

    x_min, y_min, x_max, y_max = geometry.bounding_box
    xs = np.arange(x_min + clearance, x_max - clearance + 0.5 * spacing, spacing)
    ys = np.arange(y_min + clearance, y_max - clearance + 0.5 * spacing, spacing)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    sites = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    fits = geometry.contains(sites) & (geometry.distance_to_edges(sites).min(axis=1) >= clearance)
    sites = sites[fits]
    if sites.shape[0] < n_particles:
        raise ValueError(f"Only {sites.shape[0]} lattice sites fit in the domain, but {n_particles} particles were "
                         f"requested.")

    sites = sites[:n_particles]
    radii = rng.uniform(r_min, r_max, n_particles)
    offsets = rng.uniform(-jitter_amplitude, jitter_amplitude, (n_particles, 2))
    centres = sites + offsets
    return [Particle(float(x), float(y), float(r)) for (x, y), r in zip(centres, radii)]
