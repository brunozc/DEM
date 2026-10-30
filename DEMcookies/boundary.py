"""Boundary conditions that can be assigned to the facets of a domain, and the runtime state of loaded platens.

The facets of a domain are the edges of a polygon (``Geometry``) or the faces of a polyhedron (``Geometry3D``). Each
facet carries one of:

* ``Wall``: a rigid, fixed, frictional wall (``friction=0`` gives a smooth wall);
* ``Open``: no wall; particles that cross the facet leave the simulation;
* ``LoadedPlaten``: a rigid body that applies a time-dependent load. Facets that carry the same ``LoadedPlaten``
  object form one body, so it can be a single plate or a polyline or polyhedral surface such as a wedge or a curved
  indenter. It acts as a fixed wall until loading starts and then translates along its direction of motion;
* ``PrescribedPlaten``: a rigid body of the same kind that follows a time-dependent displacement instead, whatever
  the particles push back with.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .facets import FacetSet, SegmentFacets
from .globals import PRESCRIBED_START_TOLERANCE
from .loads import DisplacementFunction, LoadFunction
from .vectors import norm


def _check_friction(friction: Optional[float]) -> None:
    """Check an optional boundary friction coefficient.

    Args:
        friction: Coulomb friction coefficient, or None to use the particle material's coefficient.

    Raises:
        ValueError: if the friction coefficient is negative.
    """
    if friction is not None and friction < 0.0:
        raise ValueError(f"friction must be non-negative, got {friction}.")


def _check_direction(name: str, direction: Optional[Tuple[float, ...]]) -> None:
    """Check an optional direction of motion of a platen.

    Args:
        name: name of the boundary condition, for the error message.
        direction: direction of motion, or None to derive it from the facets.

    Raises:
        ValueError: if the direction is not a finite, non-zero 2D or 3D vector.
    """
    if direction is None:
        return
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape not in ((2, ), (3, )) or not np.all(np.isfinite(vector)) or norm(vector) <= 0.0:
        kind = {(2, ): "(x, y)", (3, ): "(x, y, z)"}.get(vector.shape, "(x, y) or (x, y, z)")
        raise ValueError(f"{name} direction must be a finite, non-zero {kind} vector, got {direction}.")


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
    """Open facet: there is no wall, and particles that cross it are removed once they are clear of the domain."""


@dataclass(frozen=True, eq=False)
class LoadedPlaten:
    """Rigid body on one or several facets that applies a time-dependent load to the packing.

    Every facet that carries the *same* ``LoadedPlaten`` object belongs to the same rigid body, so a body can have
    any polyline shape in 2D or polyhedral shape in 3D, e.g. a wedge or a curved indenter. Separately created objects
    are separate bodies, even if their fields are equal. The body acts as a fixed wall while the domain is filled.
    When loading starts it gets one degree of freedom, a translation along ``direction``, driven by the applied load
    and by the contact reaction of the particles. To keep the body stable its mass should be at least about ten
    times the mass of the largest particle. A weightless body under zero load is free: when the load is removed, the
    elastic rebound of the packing sends it away at constant speed, so a body that is unloaded should usually be
    given its self weight.

    Attributes:
        load: applied load as a function of the time since loading started [N]; positive pushes the body along its
            direction of motion.
        mass: mass of the body [kg].
        friction: Coulomb friction coefficient between the particles and the body; None uses the particle
            material's coefficient.
        self_weight: whether gravity also acts on the body mass.
        seat_on_bed: whether the body is first moved along its direction until it touches the packing, so loading
            starts without an impact.
        direction: direction of motion and of the load, (x, y) in 2D or (x, y, z) in 3D; None uses the inward normal
            of a single facet, or the mean of the inward normals of several facets weighted by their lengths (2D)
            or areas (3D).
    """

    load: LoadFunction
    mass: float
    friction: Optional[float] = None
    self_weight: bool = False
    seat_on_bed: bool = True
    direction: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        """Validate the platen.

        Raises:
            ValueError: if the mass is not positive.
            ValueError: if the friction coefficient is negative.
            ValueError: if the direction is not a finite, non-zero 2D or 3D vector.
        """
        if self.mass <= 0.0:
            raise ValueError(f"LoadedPlaten mass must be positive, got {self.mass}.")
        _check_friction(self.friction)
        _check_direction("LoadedPlaten", self.direction)


@dataclass(frozen=True, eq=False)
class PrescribedPlaten:
    """Rigid body on one or several facets that follows a prescribed displacement history.

    The facets that carry the same ``PrescribedPlaten`` object form one rigid body, as for a ``LoadedPlaten``, and
    the body acts as a fixed wall while the domain is filled. When loading starts it translates along ``direction``
    by the prescribed displacement, whatever the particles push back with: the loading is displacement-controlled
    rather than force-controlled. The reaction of the particles is recorded as for a loaded body, and since the body
    has no mass it is also the force needed to move the body along its path. A body that is drawn back faster than
    the packing rebounds loses contact, and its reaction drops to zero. Nothing limits how quickly the body moves,
    so a fast or large displacement can press it deep into the packing within a few time steps.

    Attributes:
        displacement: translation along the direction of motion as a function of the time since loading started
            [m]; positive moves the body along its direction. It must be zero when loading starts, e.g. a
            ``Sinusoidal`` with ``mean = amplitude`` and ``phase = -pi / 2``.
        friction: Coulomb friction coefficient between the particles and the body; None uses the particle
            material's coefficient.
        seat_on_bed: whether the body is first moved along its direction until it touches the packing, so that the
            displacement is measured from first contact.
        direction: direction of motion, (x, y) in 2D or (x, y, z) in 3D; None uses the inward normal of a single
            facet, or the mean of the inward normals of several facets weighted by their lengths (2D) or areas
            (3D).
    """

    displacement: DisplacementFunction
    friction: Optional[float] = None
    seat_on_bed: bool = True
    direction: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        """Validate the platen.

        Raises:
            ValueError: if the friction coefficient is negative.
            ValueError: if the direction is not a finite, non-zero 2D or 3D vector.
            ValueError: if the displacement is not zero when loading starts.
        """
        _check_friction(self.friction)
        _check_direction("PrescribedPlaten", self.direction)
        start = float(self.displacement(0.0))
        # written so that a NaN start fails as well
        if not abs(start) <= PRESCRIBED_START_TOLERANCE:
            raise ValueError(f"PrescribedPlaten displacement must be zero when loading starts, got {start} m.")


PlatenCondition = Union[LoadedPlaten, PrescribedPlaten]

# The boundary conditions that make their facets a moving rigid body, for isinstance checks.
PLATEN_CONDITIONS = (LoadedPlaten, PrescribedPlaten)

BoundaryCondition = Union[Wall, Open, LoadedPlaten, PrescribedPlaten]


def resolve_body_direction(spec: PlatenCondition, measures: npt.NDArray[np.float64],
                           normals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the unit direction of motion of a moving body on any facets.

    Args:
        spec: boundary condition that defines the body.
        measures: length (2D) or area (3D) of each facet of the body [m or m^2].
        normals: unit inward normals of the facets of the body, shape (k, d).

    Returns:
        The unit direction.

    Raises:
        ValueError: if the direction has a different number of components than the domain has dimensions.
        ValueError: if no direction is given and the inward normals of the facets cancel out.
    """
    dimension = normals.shape[1]
    name = type(spec).__name__
    if spec.direction is not None:
        vector = np.asarray(spec.direction, dtype=np.float64)
        if vector.size != dimension:
            raise ValueError(f"{name} direction {spec.direction} has {vector.size} components, but the domain "
                             f"is {dimension}-dimensional.")
        return np.asarray(vector / norm(vector))
    resultant = (measures[:, None] * normals).sum(axis=0)
    size = float(norm(resultant))
    if size <= 1.0e-6 * float(measures.sum()):
        facets = "edges" if dimension == 2 else "faces"
        raise ValueError(f"The inward normals of the platen {facets} cancel out; give the {name} an explicit "
                         f"direction.")
    return np.asarray(resultant / size)


def resolve_direction(spec: PlatenCondition, starts: npt.NDArray[np.float64], ends: npt.NDArray[np.float64],
                      normals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the unit direction of motion of a moving body on edges of a polygon.

    Args:
        spec: boundary condition that defines the body.
        starts: start points of the edges of the body, shape (k, 2) [m].
        ends: end points of the edges of the body, shape (k, 2) [m].
        normals: unit inward normals of the edges of the body, shape (k, 2).

    Returns:
        The unit direction.

    Raises:
        ValueError: if the direction is not two-dimensional.
        ValueError: if no direction is given and the inward normals of the edges cancel out.
    """
    edges = ends - starts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    return resolve_body_direction(spec, lengths, normals)


class Platen:
    """Runtime state of a moving body, loaded or with a prescribed displacement, during a simulation.

    The body is the set of its facets translated by ``offset`` along ``direction``.

    Attributes:
        edges: indices of the facets of the geometry that form the body, the edges of a polygon or the faces of a
            polyhedron; ``faces`` is the same tuple.
        spec: boundary condition that defines the body.
        direction: unit direction of motion.
        offset: translation along the direction from the position of the facets in the geometry [m].
        seat_offset: translation at which loading started [m].
        velocity: velocity along the direction [m/s].
        reaction: contact force with which the particles push the body back, against its direction [N].
        active: whether the body has been released and moves under its load or along its prescribed path.
    """

    def __init__(self, edges: Tuple[int, ...], spec: PlatenCondition, starts: npt.NDArray[np.float64],
                 ends: npt.NDArray[np.float64], normals: npt.NDArray[np.float64]) -> None:
        """Create a body on edges of a polygon, at the position of its edges.

        Args:
            edges: indices of the geometry edges that form the body.
            spec: boundary condition that defines the body.
            starts: start vertices of the edges, shape (k, 2) [m].
            ends: end vertices of the edges, shape (k, 2) [m].
            normals: unit inward normals of the edges, shape (k, 2).
        """
        facets = SegmentFacets(
            np.array(starts, dtype=np.float64).reshape(-1, 2),
            np.array(ends, dtype=np.float64).reshape(-1, 2))
        self.__setup(edges, spec, facets, np.asarray(normals).reshape(-1, 2))

    @classmethod
    def from_facets(cls, indices: Tuple[int, ...], spec: PlatenCondition, facets: FacetSet,
                    normals: npt.NDArray[np.float64]) -> "Platen":
        """Create a body on any facets, edges of a polygon or faces of a polyhedron, at their position.

        Args:
            indices: indices of the geometry facets that form the body.
            spec: boundary condition that defines the body.
            facets: the facets of the body.
            normals: unit inward normals of the facets, shape (k, d).

        Returns:
            The body.
        """
        platen = cls.__new__(cls)
        platen.__setup(indices, spec, facets, np.asarray(normals, dtype=np.float64))
        return platen

    def __setup(self, indices: Tuple[int, ...], spec: PlatenCondition, facets: FacetSet,
                normals: npt.NDArray[np.float64]) -> None:
        """Set the body up at the position of its facets, at rest and not yet released.

        Args:
            indices: indices of the geometry facets that form the body.
            spec: boundary condition that defines the body.
            facets: the facets of the body.
            normals: unit inward normals of the facets, shape (k, d).
        """
        self.edges = tuple(indices)
        self.spec = spec
        self.__facets = facets
        self.direction = resolve_body_direction(spec, facets.measures, normals)
        self.offset = 0.0
        self.seat_offset = 0.0
        self.velocity = 0.0
        self.reaction = 0.0
        self.active = False

    @property
    def faces(self) -> Tuple[int, ...]:
        """Indices of the facets that form the body, under their 3D name; the same tuple as ``edges``."""
        return self.edges

    @property
    def prescribed(self) -> bool:
        """Whether the body follows a prescribed displacement rather than moving under a load."""
        return isinstance(self.spec, PrescribedPlaten)

    @property
    def shift(self) -> npt.NDArray[np.float64]:
        """Translation of the body from the position of its facets in the geometry [m]."""
        return np.asarray(self.offset * self.direction)

    @property
    def facets(self) -> FacetSet:
        """The facets of the body at its current position."""
        return self.__facets.moved(self.shift)

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Current vertices of each facet of the body, shape (k, vertices per facet, d) [m]."""
        return self.facets.vertex_array()

    @property
    def starts(self) -> npt.NDArray[np.float64]:
        """Current start points of the edges of a body in a polygon, shape (k, 2) [m].

        Raises:
            AttributeError: if the body is on the faces of a polyhedron, which have no start points.
        """
        return self.__segments().starts + self.shift

    @property
    def ends(self) -> npt.NDArray[np.float64]:
        """Current end points of the edges of a body in a polygon, shape (k, 2) [m].

        Raises:
            AttributeError: if the body is on the faces of a polyhedron, which have no end points.
        """
        return self.__segments().ends + self.shift

    @property
    def displacement(self) -> float:
        """Translation along the direction of motion since loading started [m]."""
        return self.offset - self.seat_offset

    def seat(self, positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64]) -> None:
        """Move the body along its direction until one of its facets touches the first particle in its way.

        The body is never moved backwards: if a particle ahead of it already overlaps it, it stays where it is.

        Args:
            positions: particle centres [m].
            radii: particle radii [m].
        """
        if radii.size == 0:
            return
        travel = self.facets.seat_travel(positions, radii, self.direction)
        if np.isfinite(travel):
            self.offset += travel

    def activate(self) -> None:
        """Release the body to move under its load, or along its prescribed path, from its current position."""
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

        Raises:
            TypeError: if the body follows a prescribed displacement.
        """
        spec = self.spec
        if not isinstance(spec, LoadedPlaten):
            raise TypeError("A PrescribedPlaten follows its displacement history and has no equation of motion.")
        force = spec.load(load_time) - self.reaction
        if spec.self_weight:
            force += spec.mass * float(gravity @ self.direction)
        return force / spec.mass

    def drive(self, load_time: float, dt: float) -> None:
        """Move a body with a prescribed displacement to its position at the end of a time step.

        The velocity is set to the mean velocity over the step, which to second order is the velocity half way
        through it, the instant whose velocities the particles have when the contact forces are evaluated.

        Args:
            load_time: time since loading started, at the start of the step [s].
            dt: time step [s].

        Raises:
            TypeError: if the body moves under a load.
        """
        spec = self.spec
        if not isinstance(spec, PrescribedPlaten):
            raise TypeError("A LoadedPlaten moves under its load and cannot be driven along a displacement.")
        before = float(spec.displacement(load_time))
        after = float(spec.displacement(load_time + dt))
        self.offset = self.seat_offset + after
        self.velocity = (after - before) / dt

    def applied_load(self, load_time: float) -> float:
        """Load applied to the body [N].

        A body with a prescribed displacement has no mass, so the force that moves it along its path is the
        reaction of the particles.

        Args:
            load_time: time since loading started [s].

        Returns:
            The applied load [N].
        """
        if isinstance(self.spec, LoadedPlaten):
            return float(self.spec.load(load_time))
        return self.reaction

    def __segments(self) -> SegmentFacets:
        """Return the edges of a body in a polygon, at the position they have in the geometry.

        Raises:
            AttributeError: if the body is on the faces of a polyhedron.
        """
        if not isinstance(self.__facets, SegmentFacets):
            raise AttributeError("A body on the faces of a polyhedron has no edge start and end points; use vertices.")
        return self.__facets
