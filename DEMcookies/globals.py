"""Constants and user-tunable numerical settings shared by the DEMcookies modules."""

import math
from dataclasses import dataclass
from typing import Tuple

# Factor 2 sqrt(5/6) of the Tsuji damping coefficient.
DAMPING_CONSTANT = 2.0 * math.sqrt(5.0 / 6.0)

# Coefficients of the rational approximation of the Rayleigh to shear wave speed ratio, 0.1631 nu + 0.8766.
RAYLEIGH_POISSON_COEFFICIENT = 0.1631
RAYLEIGH_OFFSET = 0.8766

# Smallest restitution coefficient used by the damping model, because ln(e) is singular at e = 0.
MIN_RESTITUTION = 1.0e-4

# Relative tolerance for geometric tests, scaled by the size of the domain.
GEOMETRY_TOLERANCE = 1.0e-9

# Largest displacement [m] that a prescribed displacement history may have when loading starts. A larger one would
# make the body jump into the packing in a single time step.
PRESCRIBED_START_TOLERANCE = 1.0e-12

# Largest distance of a vertex of a face of a polyhedron from the plane of the face, relative to the size of the
# domain. It is looser than the geometric tolerance, so that faces whose vertices were rounded on input are accepted.
PLANARITY_TOLERANCE = 1.0e-6

# Half of the eight neighbouring cells of the cell grid; together with the cell itself every pair is visited once.
HALF_STENCIL_OFFSETS: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

# The same in 3D: half of the 26 neighbouring cells, the layer above the cell and the forward half of its own layer.
HALF_STENCIL_OFFSETS_3D: Tuple[Tuple[int, int, int],
                               ...] = ((0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0), (-1, -1, 1), (0, -1, 1),
                                       (1, -1, 1), (-1, 0, 1), (0, 0, 1), (1, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1))

# Largest number of point-edge pairs that a geometric query of the faces of a polyhedron handles at once, which
# bounds the memory that queries over many points, such as the sampling of the domain, need.
QUERY_CHUNK_SIZE = 200000

# Smallest distance from the centre to a face of the hull of the unit directions drawn for a random polyhedron. A
# regular tetrahedron has 1/3; a draw below this bound is a sliver and is drawn again.
MIN_CENTRE_CLEARANCE = 0.2

# Smallest squared sine of the angle between two planes for the line where they meet to bound the overlap of two
# polytopes. Planes closer to parallel than this meet far away, or not at all, and give no edge.
PARALLEL_PLANES_TOLERANCE = 1.0e-12

# Smallest squared length of the difference of the in-plane directions of two facets that meet at an edge, below which
# the facets fold back onto each other and the plane between their regions is the one at right angles to the first.
FOLD_TOLERANCE = 1.0e-12

# Largest difference, relative to the particle radius, between the unit normals and between the offsets of two facets
# that a polytope touches for them to lie in one plane and share one contact surface.
COPLANAR_TOLERANCE = 1.0e-6

# Relative growth of the second body of a pair of polytopes, or push of the planes of a region against the size of a
# polytope, before their overlap is measured. Faces of the two bodies that lie in one plane, as the tops of aligned
# boxes do, would otherwise both bound the overlap; the growth leaves one of them inside the other body.
OVERLAP_TIE_BREAK = 1.0e-9

# Moment of inertia of a sphere about its centre over m R^2.
SPHERE_INERTIA_FACTOR = 0.4

# Bit shift used to pack a particle pair (i, j) into the single integer key (i << shift) | j.
PAIR_KEY_SHIFT = 32

# Smallest volume porosity used by the drag law. Overlapping particles can make the computed porosity of a dense
# packing unphysically small, and the drag grows without bound as the porosity tends to zero.
MIN_POROSITY = 0.2

# Number of samples per cell side used to measure the area of the domain around each node of the porosity grid.
POROSITY_SAMPLES_PER_CELL = 8

# Largest porosity used for the permeability and for the pore pressure force, which both diverge as the solid vanishes.
MAX_POROSITY = 0.95

# Solid volume fractions between which the skeleton carries the water in its pores along: completely above the dense
# value, where the water is trapped in the pores, and not at all below the dilute value, where it flows past the
# particles of a suspension.
CARRIED_SOLID_FRACTION_DENSE = 0.5
CARRIED_SOLID_FRACTION_DILUTE = 0.2

# Constant c of the permeability k = c R^2 eps^4.7 / (1 - eps) that the drag law implies at low Reynolds number,
# c = 4 / (3 * 5.76), with 5.76 = 4.8^2 / 4 from the Stokes limit of the drag coefficient.
PERMEABILITY_CONSTANT = 4.0 / (3.0 * 5.76)

# Smallest domain area of a grid node, as a fraction of the cell area, for the node to carry a pore pressure.
ACTIVE_NODE_FRACTION = 1.0e-3

# Relative residual at which the conjugate gradient solver of the pore pressure stops, and its iteration budget as a
# multiple of the number of unknowns (in exact arithmetic it converges within that number).
PRESSURE_SOLVER_TOLERANCE = 1.0e-10
PRESSURE_SOLVER_ITERATIONS = 10

# Largest time step as a fraction of the time an undrained pressure wave needs to cross a grid cell.
COUPLING_TIME_STEP_FRACTION = 0.5

# Porosity of a dense packing, at which the pore pressure resists a collective motion of the skeleton most quickly.
DENSE_POROSITY = 0.35

# Number of time steps between two searches for particles that left through an open edge. A particle that has left
# interacts with nothing, so removing it a few steps late is harmless and saves a polygon test every step.
ESCAPE_CHECK_INTERVAL = 50


@dataclass(frozen=True)
class GlobalSettings:
    """User-tunable numerical settings of a simulation.

    Attributes:
        gravity: magnitude of the gravitational acceleration, which acts along -y [m/s^2].
        time_step_safety: fraction of the Rayleigh critical time step used as the integration time step.
        quiet_tolerance: mean translational kinetic energy per particle below which the packing is at rest [J].
        neighbour_skin: distance added to the contact distance when building the neighbour list, as a fraction of
            the smallest particle radius. Larger values rebuild the list less often but test more pairs per step.
        escape_margin: distance beyond an open edge at which a particle is removed, as a multiple of the largest
            particle radius.
    """

    gravity: float = 9.81
    time_step_safety: float = 0.2
    quiet_tolerance: float = 1.0e-6
    neighbour_skin: float = 0.5
    escape_margin: float = 4.0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: if the gravity is negative.
            ValueError: if the time step safety factor is not in (0, 1].
            ValueError: if the quiet tolerance is not positive.
            ValueError: if the neighbour skin is negative.
            ValueError: if the escape margin is negative.
        """
        if self.gravity < 0.0:
            raise ValueError(f"gravity must be non-negative, got {self.gravity}.")
        if not 0.0 < self.time_step_safety <= 1.0:
            raise ValueError(f"time_step_safety must be in (0, 1], got {self.time_step_safety}.")
        if self.quiet_tolerance <= 0.0:
            raise ValueError(f"quiet_tolerance must be positive, got {self.quiet_tolerance}.")
        if self.neighbour_skin < 0.0:
            raise ValueError(f"neighbour_skin must be non-negative, got {self.neighbour_skin}.")
        if self.escape_margin < 0.0:
            raise ValueError(f"escape_margin must be non-negative, got {self.escape_margin}.")
