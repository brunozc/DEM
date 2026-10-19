"""Boundary conditions that can be assigned to the edges of a Geometry, and the runtime state of loaded platens.

Each edge of the polygonal domain carries one of:

* ``Wall``: a rigid, fixed, frictional wall (``friction=0`` gives a smooth wall);
* ``Open``: no wall; particles that cross the edge leave the simulation;
* ``LoadedPlaten``: a rigid plate on the edge that applies a time-dependent load. It acts as a fixed wall until
  loading starts and then translates along the inward normal of its edge.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

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


@dataclass(frozen=True)
class LoadedPlaten:
    """Rigid plate on an edge that applies a time-dependent load to the packing.

    The plate acts as a fixed wall while the domain is filled. When loading starts it becomes a body with one
    degree of freedom, the translation along the inward normal of its edge, driven by the applied load and by the
    contact reaction of the particles. To keep the plate stable its mass should be at least about ten times the
    mass of the largest particle.

    Attributes:
        load: applied load as a function of the time since loading started [N]; positive pushes into the domain.
        mass: mass of the plate [kg].
        friction: Coulomb friction coefficient between the particles and the plate; None uses the particle
            material's coefficient.
        self_weight: whether gravity also acts on the plate mass.
        seat_on_bed: whether the plate is first moved along its normal until it touches the packing, so loading
            starts without an impact.
    """

    load: LoadFunction
    mass: float
    friction: Optional[float] = None
    self_weight: bool = False
    seat_on_bed: bool = True

    def __post_init__(self) -> None:
        """Validate the platen.

        Raises:
            ValueError: if the mass is not positive.
            ValueError: if the friction coefficient is negative.
        """
        if self.mass <= 0.0:
            raise ValueError(f"LoadedPlaten mass must be positive, got {self.mass}.")
        _check_friction(self.friction)


BoundaryCondition = Union[Wall, Open, LoadedPlaten]


class Platen:
    """Runtime state of a loaded platen during a simulation.

    The platen is the segment of its edge translated by ``offset`` along the inward normal.
    """

    def __init__(self, edge: int, spec: LoadedPlaten, start: npt.NDArray[np.float64], end: npt.NDArray[np.float64],
                 normal: npt.NDArray[np.float64]) -> None:
        """Create the platen at the position of its edge.

        Args:
            edge: index of the geometry edge the platen sits on.
            spec: boundary condition that defines the platen.
            start: start vertex of the edge [m].
            end: end vertex of the edge [m].
            normal: unit inward normal of the edge.
        """
        self.edge = edge
        self.spec = spec
        self.normal = np.array(normal, dtype=np.float64)
        self.__start = np.array(start, dtype=np.float64)
        self.__end = np.array(end, dtype=np.float64)
        self.offset = 0.0
        self.seat_offset = 0.0
        self.velocity = 0.0
        self.reaction = 0.0
        self.active = False

    @property
    def start(self) -> npt.NDArray[np.float64]:
        """Current start point of the plate [m]."""
        return self.__start + self.offset * self.normal

    @property
    def end(self) -> npt.NDArray[np.float64]:
        """Current end point of the plate [m]."""
        return self.__end + self.offset * self.normal

    @property
    def displacement(self) -> float:
        """Translation along the inward normal since loading started [m]."""
        return self.offset - self.seat_offset

    def seat(self, positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> None:
        """Move the plate along its normal until it touches the first particle in front of it.

        The plate is never moved backwards, so particles that already touch it keep their contact.

        Args:
            positions: particle centres [m].
            radii: particle radii [m].
        """
        edge = self.end - self.start
        length = float(np.hypot(edge[0], edge[1]))
        tangent = edge / length
        relative = positions - self.start
        depth = relative @ self.normal
        along = relative @ tangent
        # beyond the plate ends the particle can only be touched by an end point
        lateral = np.maximum(0.0, np.maximum(-along, along - length))
        reachable = (depth > 0.0) & (lateral < radii)
        if not np.any(reachable):
            return
        gaps = depth[reachable] - np.sqrt(radii[reachable]**2 - lateral[reachable]**2)
        self.offset += max(0.0, float(np.min(gaps)))

    def activate(self) -> None:
        """Release the plate so that it moves under the applied load from its current position."""
        self.seat_offset = self.offset
        self.velocity = 0.0
        self.active = True

    def acceleration(self, load_time: float, gravity: npt.NDArray[np.float64]) -> float:
        """Acceleration of the plate along its inward normal.

        Args:
            load_time: time since loading started [s].
            gravity: gravitational acceleration vector [m/s^2].

        Returns:
            The acceleration along the inward normal [m/s^2].
        """
        force = self.spec.load(load_time) - self.reaction
        if self.spec.self_weight:
            force += self.spec.mass * float(gravity @ self.normal)
        return force / self.spec.mass
