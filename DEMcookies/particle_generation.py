"""Particle definition and gravitational ("raining") deposition.

Particles are generated on a loose, non-overlapping lattice in the upper part
of the box.  When the solver starts they fall under gravity and compact into a
dense random packing - this is the "raining" initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import Box


@dataclass
class Particle:
    """A single circular particle (used to build the initial configuration)."""

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0


def generate_rain_particles(box: Box, n_particles: int,
                            radius_range: tuple[float, float] = (0.006, 0.012),
                            seed: int = 0, spacing_factor: float = 1.15,
                            jitter: float = 0.25) -> list[Particle]:
    """Create ``n_particles`` on a loose lattice ready to rain down.

    Parameters
    ----------
    box:
        The container; particles are kept inside the left/right walls.
    n_particles:
        Number of particles to create.
    radius_range:
        ``(r_min, r_max)`` for the uniformly distributed radii.
    seed:
        Seed for the random generator (reproducible packings).
    spacing_factor:
        Lattice spacing as a multiple of the largest diameter.  Values > 1
        guarantee that the initial configuration has no overlaps.
    jitter:
        Random in-cell displacement as a fraction of the free space, which
        breaks the lattice symmetry and yields a more natural packing.
    """
    rng = np.random.default_rng(seed)
    r_min, r_max = radius_range
    d_max = 2.0 * r_max
    spacing = d_max * spacing_factor

    # how many columns fit between the side walls
    usable_width = box.width - d_max
    n_cols = max(1, int(usable_width // spacing) + 1)
    x0 = r_max
    # maximum jitter that still cannot create an overlap
    jitter_amp = jitter * 0.5 * (spacing - d_max)

    particles: list[Particle] = []
    for k in range(n_particles):
        col = k % n_cols
        row = k // n_cols
        radius = float(rng.uniform(r_min, r_max))
        x = x0 + col * spacing + float(rng.uniform(-jitter_amp, jitter_amp))
        y = r_max + row * spacing + float(rng.uniform(-jitter_amp, jitter_amp))
        # keep the particle between the side walls
        x = min(max(x, radius), box.width - radius)
        particles.append(Particle(x, y, radius))

    return particles

