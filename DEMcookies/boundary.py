"""Boundary conditions that can be assigned to the edges of a Geometry, and the runtime state of loaded platens.

Each edge of the polygonal domain carries one of:

* ``Wall``: a rigid, fixed, frictional wall (``friction=0`` gives a smooth wall);
* ``Open``: no wall; particles that cross the edge leave the simulation;
* ``LoadedPlaten``: a rigid body that applies a time-dependent load. Edges that carry the same ``LoadedPlaten``
  object form one body, so it can be a single plate or a polyline such as a wedge or a curved indenter. It acts as
  a fixed wall until loading starts and then translates along its direction of motion.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .segments import project_onto_segments
from .loads import LoadFunction


def _check_friction(friction: Optional[float]) -> None:
    """Check an optional boundary friction coefficient.

    Args:
        friction: Coulomb friction coefficient, or None to use the particle material's coefficient.

    Raises:
        ValueError: if the friction coefficient is negative.
    """
    if friction is not None and friction < 0.0:
        raise ValueError(f"friction must be non-negative, got {friction}.")


@dataclass(frozen=True)
class Wall:
    """Rigid, fixed wall.

    Attributes:
        friction: Coulomb friction coefficient between the particles and the wall; None uses the particle
            material's coefficient and 0 gives a smooth wall.
    """

    friction: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the wall.

        Raises:
            ValueError: if the friction coefficient is negative.
        """
        _check_friction(self.friction)


@dataclass(frozen=True)
class Open:
    """Open edge: there is no wall, and particles that cross it are removed once they are clear of the domain."""


@dataclass(frozen=True, eq=False)
class LoadedPlaten:
    """Rigid body on one or several edges that applies a time-dependent load to the packing.

    Every edge that carries the *same* ``LoadedPlaten`` object belongs to the same rigid body, so a body can have
    any polyline shape, e.g. a wedge or a curved indenter. Separately created objects are separate bodies, even if
    their fields are equal. The body acts as a fixed wall while the domain is filled. When loading starts it gets
    one degree of freedom, a translation along ``direction``, driven by the applied load and by the contact reaction
    of the particles. To keep the body stable its mass should be at least about ten times the mass of the largest
    particle. A weightless body under zero load is free: when the load is removed, the elastic rebound of the
    packing sends it away at constant speed, so a body that is unloaded should usually be given its self weight.

    Attributes:
        load: applied load as a function of the time since loading started [N]; positive pushes the body along its
            direction of motion.
        mass: mass of the body [kg].
        friction: Coulomb friction coefficient between the particles and the body; None uses the particle
            material's coefficient.
        self_weight: whether gravity also acts on the body mass.
        seat_on_bed: whether the body is first moved along its direction until it touches the packing, so loading
            starts without an impact.
        direction: direction of motion and of the load; None uses the inward normal of a single edge, or the
            length-weighted mean of the inward normals of several edges.
    """

    load: LoadFunction
    mass: float
    friction: Optional[float] = None
    self_weight: bool = False
    seat_on_bed: bool = True
    direction: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        """Validate the platen.

        Raises:
            ValueError: if the mass is not positive.
            ValueError: if the friction coefficient is negative.
            ValueError: if the direction is not a finite, non-zero 2D vector.
        """
        if self.mass <= 0.0:
            raise ValueError(f"LoadedPlaten mass must be positive, got {self.mass}.")
        _check_friction(self.friction)
        if self.direction is not None:
            vector = np.asarray(self.direction, dtype=np.float64)
            if vector.shape != (2, ) or not np.all(np.isfinite(vector)) or np.hypot(*vector) <= 0.0:
                raise ValueError(f"LoadedPlaten direction must be a finite, non-zero (x, y) vector, "
                                 f"got {self.direction}.")


BoundaryCondition = Union[Wall, Open, LoadedPlaten]


def resolve_direction(spec: LoadedPlaten, starts: npt.NDArray[np.float64], ends: npt.NDArray[np.float64],
                      normals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the unit direction of motion of a loaded body.

    Args:
        spec: boundary condition that defines the body.
        starts: start points of the edges of the body, shape (k, 2) [m].
        ends: end points of the edges of the body, shape (k, 2) [m].
        normals: unit inward normals of the edges of the body, shape (k, 2).

    Returns:
        The unit direction.

    Raises:
        ValueError: if no direction is given and the inward normals of the edges cancel out.
    """
    if spec.direction is not None:
        vector = np.asarray(spec.direction, dtype=np.float64)
        return np.asarray(vector / np.hypot(vector[0], vector[1]))
    edges = ends - starts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    resultant = (lengths[:, None] * normals).sum(axis=0)
    size = float(np.hypot(resultant[0], resultant[1]))
    if size <= 1.0e-6 * float(lengths.sum()):
        raise ValueError("The inward normals of the platen edges cancel out; give the LoadedPlaten an explicit "
                         "direction.")
    return np.asarray(resultant / size)


def _time_to_contact(positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64], start: npt.NDArray[np.float64],
                     end: npt.NDArray[np.float64], direction: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Distance a segment can travel along a direction before it touches each particle.

    Args:
        positions: particle centres, shape (n, 2) [m].
        radii: particle radii [m].
        start: start point of the segment [m].
        end: end point of the segment [m].
        direction: unit direction of travel.

    Returns:
        The travel distance for each particle, infinite for particles the segment never touches [m].
    """
    # seen from the segment, the particles move against the direction of travel
    motion = -direction
    edge = end - start
    length = float(np.hypot(edge[0], edge[1]))
    tangent = edge / length
    normal = np.array([-tangent[1], tangent[0]])
    travel = np.full(radii.shape, np.inf)

    # contact with the interior of the segment: the distance to its line drops to the radius
    relative = positions - start
    height = relative @ normal
    rate = float(motion @ normal)
    if abs(rate) > 0.0:
        approaching = height * rate < 0.0
        line_travel = (np.abs(height) - radii) / abs(rate)
        foot = (relative + line_travel[:, None] * motion) @ tangent
        hits = approaching & (line_travel >= 0.0) & (foot >= 0.0) & (foot <= length)
        travel = np.where(hits, line_travel, travel)

    # contact with an end point: the first root of |w + s motion| = r
    for point in (start, end):
        offset = positions - point
        along = offset @ motion
        discriminant = along**2 - (np.einsum("ij,ij->i", offset, offset) - radii**2)
        first = -along - np.sqrt(np.maximum(discriminant, 0.0))
        hits = (discriminant >= 0.0) & (first >= 0.0)
        travel = np.where(hits, np.minimum(travel, first), travel)
    return travel


class Platen:
    """Runtime state of a loaded body during a simulation.

    The body is the set of its edges translated by ``offset`` along ``direction``.
    """

    def __init__(self, edges: Tuple[int, ...], spec: LoadedPlaten, starts: npt.NDArray[np.float64],
                 ends: npt.NDArray[np.float64], normals: npt.NDArray[np.float64]) -> None:
        """Create the body at the position of its edges.

        Args:
            edges: indices of the geometry edges that form the body.
            spec: boundary condition that defines the body.
            starts: start vertices of the edges, shape (k, 2) [m].
            ends: end vertices of the edges, shape (k, 2) [m].
            normals: unit inward normals of the edges, shape (k, 2).
        """
        self.edges = tuple(edges)
        self.spec = spec
        self.__starts = np.array(starts, dtype=np.float64).reshape(-1, 2)
        self.__ends = np.array(ends, dtype=np.float64).reshape(-1, 2)
        self.direction = resolve_direction(spec, self.__starts, self.__ends, np.asarray(normals).reshape(-1, 2))
        self.offset = 0.0
        self.seat_offset = 0.0
        self.velocity = 0.0
        self.reaction = 0.0
        self.active = False

    @property
    def starts(self) -> npt.NDArray[np.float64]:
        """Current start points of the edges of the body, shape (k, 2) [m]."""
        return self.__starts + self.offset * self.direction

    @property
    def ends(self) -> npt.NDArray[np.float64]:
        """Current end points of the edges of the body, shape (k, 2) [m]."""
        return self.__ends + self.offset * self.direction

    @property
    def displacement(self) -> float:
        """Translation along the direction of motion since loading started [m]."""
        return self.offset - self.seat_offset

    def seat(self, positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> None:
        """Move the body along its direction until one of its edges touches the first particle in its way.

        The body is never moved backwards: if a particle ahead of it already overlaps it, it stays where it is.

        Args:
            positions: particle centres [m].
            radii: particle radii [m].
        """
        if radii.size == 0:
            return
        travel = np.inf
        for start, end in zip(self.starts, self.ends):
            _, offset = project_onto_segments(positions, start[None, :], end[None, :])
            gap = np.hypot(offset[:, 0, 0], offset[:, 0, 1]) - radii
            # a particle already touching the edge on the side the body moves to blocks it
            if np.any((gap <= 0.0) & (offset[:, 0, :] @ self.direction > 0.0)):
                return
            travel = min(travel, float(np.min(_time_to_contact(positions, radii, start, end, self.direction))))
        if np.isfinite(travel):
            self.offset += travel

    def activate(self) -> None:
        """Release the body so that it moves under the applied load from its current position."""
        self.seat_offset = self.offset
        self.velocity = 0.0
        self.active = True

    def acceleration(self, load_time: float, gravity: npt.NDArray[np.float64]) -> float:
        """Acceleration of the body along its direction of motion.

        Args:
            load_time: time since loading started [s].
            gravity: gravitational acceleration vector [m/s^2].

        Returns:
            The acceleration along the direction of motion [m/s^2].
        """
        force = self.spec.load(load_time) - self.reaction
        if self.spec.self_weight:
            force += self.spec.mass * float(gravity @ self.direction)
        return force / self.spec.mass
