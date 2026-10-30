"""Time integration and the DEM simulation engine.

``VelocityVerlet`` is a small, self-contained integrator used to verify the scheme. ``DEMSimulation`` is the full
engine: it keeps the particle state in arrays, finds particle contacts with a cell grid and a Verlet neighbour list,
evaluates the Hertz-Mindlin forces between particles and against the solid facets of the domain, moves the loaded
platens under their loads and the prescribed ones along their paths, and advances everything in time with velocity
Verlet. It runs in two or in three dimensions, as set by the geometry: a polygon (``Geometry``) or a polyhedron
(``Geometry3D``).

The particles are spheres, or convex polygons (2D) and polyhedra (3D), in any mix. A contact between two spheres, or
between a sphere and a facet, follows from the overlap of the spheres; a contact that involves a polytope follows
from the region where the two bodies overlap (``DEMcookies.overlap``), or from the closest point of a polytope to a
sphere. The neighbour list works on the spheres that bound the particles.
"""

import math
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .boundary import Open, Platen
from .contact import (ContactForces, HertzMindlinModel, MaterialProperties, RollingMoments, overlap_contact_radius,
                      rayleigh_time_step)
from .facets import FacetSet
from .backend import check_device, gpu_module, to_device, to_host
from .geometry3d import Domain
from .water import PoreWater, Water, evaluate_at
from .globals import COPLANAR_TOLERANCE, ESCAPE_CHECK_INTERVAL, GEOMETRY_TOLERANCE, GlobalSettings
from .neighbours import NeighbourList
from .overlap import (Overlap, Polyhedra, polygon_distance, polygon_overlap, polygon_region_overlap,
                      polyhedron_distance, polyhedron_overlap, polyhedron_region_overlap)
from .particle_generation import AnyParticle, bounding_ratio, check_shapes
from .polytopes import (PolytopePose, PolytopeTable, apply_inertia, frame_polygons, identity_orientations, place,
                        random_orientations, rotate, rotation_matrices)
from .regions import OPEN_BODY, FacetRegions, facet_regions
from .shapes import Shape
from .vectors import cross, norm, rotation_shape, scale, scatter_add, spin_cross


class VelocityVerlet:
    """Canonical velocity-Verlet integrator for an acceleration callback."""

    def __init__(self, dt: float) -> None:
        """Create the integrator.

        Args:
            dt: time step [s].

        Raises:
            ValueError: if the time step is not positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        self.dt = float(dt)

    def step(
        self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64],
        acceleration: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Advance the positions and velocities by one step.

        Args:
            positions: current positions [m].
            velocities: current velocities [m/s].
            acceleration: function of the positions and velocities that returns the accelerations [m/s^2].

        Returns:
            The new positions, velocities and accelerations.
        """
        dt = self.dt
        a0 = acceleration(positions, velocities)
        half_velocities = velocities + 0.5 * a0 * dt
        new_positions = positions + half_velocities * dt
        a1 = acceleration(new_positions, half_velocities)
        new_velocities = half_velocities + 0.5 * a1 * dt
        return new_positions, new_velocities, a1


@dataclass(frozen=True)
class RainSettings:
    """Settings of the wave-by-wave raining deposition.

    Attributes:
        batch_size: number of particles inserted per wave.
        steps_between: time steps between two waves.
        patience: number of consecutive waves without any insertion after which the domain is full.
        attempts: random positions tried per particle and wave.
        rain_speed: initial downward speed of the inserted particles [m/s].
        insertion_band: lowest and highest y of the inserted particle centres [m]; None uses a band one largest
            radius below the top of the domain.
        settle_steps: largest number of time steps of the final settling stage.
        settle_min_steps: number of settling steps before the packing may be declared at rest.
        max_particles: largest number of particles.
        max_steps: largest number of time steps of the raining stage.
        record_every: number of time steps between two recorded frames.
        seed: seed of the random generator.
    """

    batch_size: int = 15
    steps_between: int = 450
    patience: int = 4
    attempts: int = 30
    rain_speed: float = 0.3
    insertion_band: Optional[Tuple[float, float]] = None
    settle_steps: int = 4000
    settle_min_steps: int = 500
    max_particles: int = 2000
    max_steps: int = 300000
    record_every: int = 60
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: if a count that must be positive is not.
            ValueError: if a count that must be non-negative is negative.
            ValueError: if the insertion band is empty.
        """
        for name in ("batch_size", "steps_between", "patience", "attempts", "max_particles", "record_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}.")
        for name in ("settle_steps", "settle_min_steps", "max_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}.")
        if self.insertion_band is not None and self.insertion_band[0] >= self.insertion_band[1]:
            raise ValueError(f"insertion_band must be (y_min, y_max) with y_min < y_max, got {self.insertion_band}.")


@dataclass(frozen=True, eq=False)
class Frame:
    """Snapshot of the simulation for post-processing.

    Attributes:
        time: simulation time [s].
        positions: particle centres, shape (n, d) [m].
        radii: particle radii [m].
        speeds: particle speeds [m/s].
        platens: current facets of every moving body, each of shape (k, m, d) holding the m vertices of its k
            facets: the start and end point of each edge in 2D, and the vertices of each face, padded with copies
            of its last vertex, in 3D [m].
        phase: name of the stage the frame belongs to.
        water_level: height of the free surface of the water [m], or None without water.
        orientations: orientation of every particle, angles, shape (n,), in 2D and quaternions, shape (n, 4), in 3D;
            None if every particle is a sphere.
        shapes: shapes of the particles in their body frames, shared with the simulation and the other frames, which
            never change it; None if every particle is a sphere.
    """

    time: float
    positions: npt.NDArray[np.float64]
    radii: npt.NDArray[np.float64]
    speeds: npt.NDArray[np.float64]
    platens: Tuple[npt.NDArray[np.float64], ...]
    phase: str
    water_level: Optional[float] = None
    orientations: Optional[npt.NDArray[np.float64]] = None
    shapes: Optional[PolytopeTable] = None

    def polytopes(self) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return the polygonal or polyhedral particles of the frame and their vertices in space.

        Returns:
            The indices of those particles, shape (p,), and their vertices, padded with copies of the first, shape
            (p, v, d) [m]; both empty if every particle is a sphere.
        """
        if self.shapes is None or self.orientations is None:
            d = self.positions.shape[1]
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0, d))
        return frame_polygons(self.shapes, self.positions, self.orientations)

    def top(self) -> float:
        """Return the highest point of any particle, or -inf without particles [m]."""
        spheres = np.ones(self.radii.size, dtype=bool)
        tops = [-np.inf]
        rows, vertices = self.polytopes()
        if rows.size:
            spheres[rows] = False
            tops.append(float(vertices[..., 1].max()))
        if np.any(spheres):
            tops.append(float((self.positions[spheres, 1] + self.radii[spheres]).max()))
        return max(tops)


@dataclass
class LoadCurve:
    """Load history recorded for one moving body, loaded or with a prescribed displacement.

    Attributes:
        edges: indices of the geometry facets of the body, edges in 2D and faces in 3D; ``faces`` is the same tuple.
        time: time since loading started [s].
        applied: applied load [N]; for a body with a prescribed displacement, the force that moves it along its path,
            which is the reaction.
        displacement: translation of the body along its direction of motion since loading started [m].
        reaction: contact force with which the particles push the body back, against its direction of motion [N].
    """

    edges: Tuple[int, ...]
    time: List[float] = field(default_factory=list)
    applied: List[float] = field(default_factory=list)
    displacement: List[float] = field(default_factory=list)
    reaction: List[float] = field(default_factory=list)

    @property
    def faces(self) -> Tuple[int, ...]:
        """Indices of the geometry faces of the body, under their 3D name; the same tuple as ``edges``."""
        return self.edges

    def as_array(self) -> npt.NDArray[np.float64]:
        """Return the curve as an array with columns time, applied, displacement and reaction."""
        return np.column_stack((self.time, self.applied, self.displacement, self.reaction)).reshape(-1, 4)


@dataclass(frozen=True, eq=False)
class ContactList:
    """Contacts of the current state, either between two particles or between a particle and a facet of the domain.

    Attributes:
        first: particle i of each contact, shape (k,).
        second: particle j of each contact, or the index in the geometry of the facet that particle i touches,
            shape (k,).
        point: contact point, in the middle of the overlap of two spheres and at the centroid of the overlap of
            polytopes, shape (k, d) [m].
        normal: unit normal pointing from j towards i, shape (k, d).
        overlap: normal overlap, shape (k,) [m]; between polytopes, the mean depth of their overlap.
        normal_force: magnitude of the normal force, shape (k,) [N].
        tangential_force: tangential force on particle i, shape (k, d) [N].
    """

    first: npt.NDArray[np.int64]
    second: npt.NDArray[np.int64]
    point: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]
    overlap: npt.NDArray[np.float64]
    normal_force: npt.NDArray[np.float64]
    tangential_force: npt.NDArray[np.float64]

    @property
    def force(self) -> npt.NDArray[np.float64]:
        """Total contact force on particle i, shape (k, d) [N]; particle j, or the facet, receives the opposite."""
        return np.asarray(self.normal_force[:, None] * self.normal + self.tangential_force)


@dataclass(frozen=True, eq=False)
class _ContactBatch:
    """Contacts found and evaluated together, before their forces are applied and their springs stored.

    Attributes:
        first: particle i of each contact.
        second: particle j of each contact, or the slot of the solid facet that particle i touches.
        normal: unit normal pointing from j towards i, shape (k, d).
        overlap: normal overlap, or mean depth of the overlap of polytopes [m].
        forces: forces of the contact law, with the updated tangential springs.
        rolling: rolling moments, with the updated rolling springs, or None without rolling resistance.
        point: contact point, shape (k, d) [m].
        levers: vectors from the centres of particles i and j to where the force acts, shape (k, d) each, the
            second None for a facet; None for spheres, whose contact force acts at their radius against the normal.
        slot: position of each contact in the neighbour list, for contacts between particles.
    """

    first: npt.NDArray[np.int64]
    second: npt.NDArray[np.int64]
    normal: npt.NDArray[np.float64]
    overlap: npt.NDArray[np.float64]
    forces: ContactForces
    rolling: Optional[RollingMoments]
    point: npt.NDArray[np.float64]
    levers: Optional[Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]] = None
    slot: Optional[npt.NDArray[np.int64]] = None


Observer = Callable[["DEMSimulation"], None]


class DEMSimulation:
    """A discrete element simulation with Hertz-Mindlin contacts, in a polygonal (2D) or polyhedral (3D) domain.

    The geometry sets the number of space dimensions d. Vectors have d components; rotations, angular velocities
    and torques are scalars about the out-of-plane axis in 2D and vectors in 3D. The y axis is vertical in both.
    Particles are spheres, or convex polygons (2D) and polyhedra (3D), whose centre is their centroid and whose
    radius is the equivalent radius of their shape, so that every particle has the mass of the sphere of its radius.

    Attributes:
        dimension: number of space dimensions, 2 or 3.
        pos: particle centres, shape (n, d) [m].
        vel: particle velocities, shape (n, d) [m/s].
        omega: particle angular velocities, shape (n,) in 2D or (n, 3) in 3D [rad/s].
        orientation: particle orientations, angles about the out-of-plane axis, shape (n,), in 2D and unit
            quaternions (w, x, y, z), shape (n, 4), in 3D, which turn the body frame of a shape into space.
        radius: particle radii [m]; the equivalent radius of a polygon or a polyhedron.
        mass: particle masses [kg].
        inertia: moments of inertia about the centre, shape (n,), in 2D and principal moments of inertia about the
            axes of the body frame, shape (n, 3), in 3D [kg m^2].
        force: total force on each particle at the last evaluation, apart from the drag of the water, which is
            integrated implicitly, shape (n, d) [N].
        torque: total torque on each particle at the last evaluation, shape (n,) in 2D or (n, 3) in 3D [N m].
        fluid_force: force of the water on each particle at the last evaluation, including the drag, shape (n, d)
            [N].
        porosity: local volume porosity at each particle, computed only when there is water (NaN otherwise).
        edge_forces: force the particles exert on each facet of the geometry, an edge in 2D or a face in 3D, shape
            (facets, d) [N]; ``face_forces`` is the same array.
        platens: runtime state of every moving body, loaded or prescribed.
        frames: recorded snapshots.
        load_curves: recorded load history of every moving body.
        time: simulation time [s].
        dt: time step [s].
    """

    # a class default, so that a simulation pickled before observers existed still loads, with none
    __observers: Tuple[Observer, ...] = ()

    def __init__(self,
                 geometry: Domain,
                 material: MaterialProperties,
                 radius_range: Optional[Tuple[float, float]] = None,
                 particles: Optional[Sequence[AnyParticle]] = None,
                 settings: Optional[GlobalSettings] = None,
                 dt: Optional[float] = None,
                 water: Optional[Water] = None,
                 shapes: Optional[Sequence[Optional[Shape]]] = None,
                 device: str = "cpu") -> None:
        """Set up the simulation.

        Args:
            geometry: domain, a polygon or a polyhedron with a boundary condition on every facet.
            material: material shared by the particles and boundaries.
            radius_range: smallest and largest radius of rained particles [m]; defaults to the range of the
                initial particles.
            particles: initial particles, ``Particle`` in a polygon and ``Particle3D`` in a polyhedron; the domain
                starts empty when None.
            settings: numerical settings; defaults to ``GlobalSettings()``.
            dt: time step [s]; defaults to a fraction of the Rayleigh time step of the smallest particle, or of the
                drag relaxation time if that is shorter. The Rayleigh time step of a polygon or a polyhedron is that
                of the sphere it holds, whose radius is the distance from its centroid to its closest face.
            water: pore water; the particles are dry when None.
            shapes: shapes of the rained particles, polygons in 2D and polyhedra in 3D: each particle takes one at
                random, in a random orientation, scaled to its radius; None in the list stands for a sphere. None
                rains spheres only.
            device: device that measures the overlaps of polygonal and polyhedral particles, "cpu" or "gpu" (see
                ``device``).

        Raises:
            ValueError: if a particle has a different number of dimensions than the geometry.
            ValueError: if a particle has a shape of the other number of dimensions.
            ValueError: if a particle has a zero quaternion as its orientation.
            ValueError: if the list of shapes is empty.
            ValueError: if a shape does not suit the number of dimensions of the geometry.
            ValueError: if neither a radius range nor particles are given.
            ValueError: if the radius range is not 0 < r_min <= r_max.
            ValueError: if a particle radius is not positive.
            ValueError: if a particle centre lies outside the domain.
            ValueError: if the time step is not positive.
            ValueError: if the time step is too large to resolve the drag of the water on the smallest particle.
            ValueError: if the device is neither "cpu" nor "gpu".
            RuntimeError: if the device is "gpu" and CuPy is not installed or finds no GPU.
        """
        self.__device = "cpu"
        self.__device_table: Optional[PolytopeTable] = None
        self.__device_source: Optional[PolytopeTable] = None
        self.device = device
        self.geometry = geometry
        self.material = material
        self.settings = settings or GlobalSettings()
        self.model = HertzMindlinModel(material)
        self.dimension = geometry.dimension
        d = self.dimension

        initial = list(particles or [])
        for k, particle in enumerate(initial):
            if particle.dimension != d:
                raise ValueError(f"Particle {k} is {_DIMENSION_NAMES[particle.dimension]}, but the geometry is "
                                 f"{_DIMENSION_NAMES[d]}.")
            if particle.shape is not None and particle.shape.dimension != d:
                raise ValueError(f"Particle {k} has a {type(particle.shape).__name__}, but the geometry is "
                                 f"{_DIMENSION_NAMES[d]}.")
        if shapes is not None:
            check_shapes(shapes, d)
        self.__rain_shapes: Optional[Tuple[Optional[Shape], ...]] = None if shapes is None else tuple(shapes)
        self.pos: npt.NDArray[np.float64] = np.array([p.position for p in initial], dtype=np.float64).reshape(-1, d)
        self.vel: npt.NDArray[np.float64] = np.array([p.velocity for p in initial], dtype=np.float64).reshape(-1, d)
        self.radius: npt.NDArray[np.float64] = np.array([p.radius for p in initial], dtype=np.float64)
        if np.any(self.radius <= 0.0):
            raise ValueError("Particle radii must be positive.")
        self.orientation = self.__initial_orientations(initial)
        self.__table = PolytopeTable.build(d, [p.shape for p in initial], self.radius)
        if radius_range is None:
            if not initial:
                raise ValueError("radius_range is required when no initial particles are given.")
            radius_range = (float(self.radius.min()), float(self.radius.max()))
        if not 0.0 < radius_range[0] <= radius_range[1]:
            raise ValueError(f"radius_range must satisfy 0 < r_min <= r_max, got {radius_range}.")
        outside = np.flatnonzero(~geometry.contains(self.pos))
        if outside.size:
            coordinates = ", ".join(f"{c:.4g}" for c in self.pos[outside[0]])
            raise ValueError(f"Particle {outside[0]} at ({coordinates}) lies outside the domain.")
        self.radius_range = (float(radius_range[0]), float(radius_range[1]))

        self.omega: npt.NDArray[np.float64] = np.zeros(rotation_shape(self.n, d))
        self.force: npt.NDArray[np.float64] = np.zeros((self.n, d))
        self.torque: npt.NDArray[np.float64] = np.zeros(rotation_shape(self.n, d))
        self.__update_mass_properties()
        self.__pose: Optional[PolytopePose] = None
        self.__pose_index: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)

        r_min = min([self.radius_range[0]] + ([float(self.radius.min())] if self.n else []))
        r_max = max([self.radius_range[1]] + ([float(self.radius.max())] if self.n else []))
        rain = self.__rain_shapes or (None, )
        # the stiffest contact of a polytope is that of its flattest face, closest to its centroid
        thinnest = [self.radius_range[0] * min(1.0 if s is None else s.inradius for s in rain)]
        thinnest += [float(self.__table.inradius.min())] if self.n else []
        if dt is not None and dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        self.__r_min, self.__r_max = r_min, r_max
        self.__rain_reach = self.radius_range[1] * bounding_ratio(rain)
        self.__user_dt = dt is not None
        self.dt = dt if dt is not None else rayleigh_time_step(material, min(thinnest), self.settings.time_step_safety)
        self.water: Optional[Water] = None
        self.__pore_water: Optional[PoreWater] = None
        self.fluid_force: npt.NDArray[np.float64] = np.zeros((self.n, d))
        self.porosity: npt.NDArray[np.float64] = np.full(self.n, np.nan)
        self.__drag_coefficient: npt.NDArray[np.float64] = np.zeros(self.n)
        self.__water_velocity: npt.NDArray[np.float64] = np.zeros((self.n, d))
        self.set_water(water)
        self.__gravity = np.zeros(d)
        self.__gravity[1] = -self.settings.gravity
        self.__escape_distance = self.settings.escape_margin * r_max
        self.__neighbours = NeighbourList(self.settings.neighbour_skin * r_min, d)
        self.__setup_boundaries()
        self.edge_forces: npt.NDArray[np.float64] = np.zeros((geometry.n_facets, d))
        self.__forces_valid = False

        self.time = 0.0
        self.load_time = 0.0
        self.steps_taken = 0
        self.n_removed = 0
        self.frames: List[Frame] = []
        self.load_curves = [LoadCurve(edges=p.edges) for p in self.platens]

    @property
    def n(self) -> int:
        """Number of particles in the simulation."""
        return int(self.radius.size)

    @property
    def device(self) -> str:
        """Device that measures the overlaps of the polygonal and polyhedral particles: "cpu" or "gpu".

        Those overlaps are most of the work of a simulation with shaped particles, and a GPU measures them many times
        faster, with CuPy; everything else is computed on the CPU in either case, and spheres are not affected. The
        device can be changed at any time, e.g. to carry on on a CPU with a simulation pickled on a machine with a
        GPU; a pickled simulation holds nothing in the memory of a GPU.
        """
        return self.__device

    @device.setter
    def device(self, device: str) -> None:
        """Change the device.

        Args:
            device: "cpu" or "gpu".

        Raises:
            ValueError: if the device is neither "cpu" nor "gpu".
            RuntimeError: if the device is "gpu" and CuPy is not installed or finds no GPU.
        """
        check_device(device)
        if device == "gpu":
            gpu_module()
        self.__device = device
        self.__device_table = None
        self.__device_source = None
        self.__pose = None

    @property
    def bounding_radius(self) -> npt.NDArray[np.float64]:
        """Radius of the sphere about the centre of each particle that holds it: its radius for a sphere [m]."""
        return self.__table.reach

    @property
    def polytope(self) -> npt.NDArray[np.bool_]:
        """Whether each particle is a polygon or a polyhedron rather than a sphere, shape (n,)."""
        return self.__table.polytope

    @property
    def shapes(self) -> PolytopeTable:
        """Shapes of the particles in their body frames, scaled to their radii."""
        return self.__table

    @property
    def face_forces(self) -> npt.NDArray[np.float64]:
        """Force the particles exert on each face of a 3D domain, under its 3D name; the same array as edge_forces."""
        return self.edge_forces

    # ------------------------------------------------------------------ #
    #  Set-up                                                            #
    # ------------------------------------------------------------------ #
    def __setup_boundaries(self) -> None:
        """Build the set of the solid facets (walls and platens) the particles can touch."""
        geometry = self.geometry
        conditions = geometry.boundary_conditions
        normals = geometry.inward_normals

        solid = [(index, condition) for index, condition in enumerate(conditions) if not isinstance(condition, Open)]
        solid_indices = [index for index, _ in solid]
        slot_of_facet = {index: s for s, index in enumerate(solid_indices)}
        friction = np.empty(len(solid))
        for s, (_, condition) in enumerate(solid):
            friction[s] = self.material.friction if condition.friction is None else condition.friction

        # every moving body moves all of its facets, which may be any subset of the solid facets
        self.platens: List[Platen] = []
        self.__platen_slot_groups: List[npt.NDArray[np.int64]] = []
        platen_of_slot = np.full(len(solid), -1, dtype=np.int64)
        for spec, indices in geometry.platen_groups:
            slots = np.array([slot_of_facet[index] for index in indices], dtype=np.int64)
            platen_of_slot[slots] = len(self.platens)
            self.__platen_slot_groups.append(slots)
            body = geometry.facets.select(list(indices))
            self.platens.append(Platen.from_facets(indices, spec, body, normals[list(indices)]))

        self.__has_open_facets = len(solid) < len(conditions)
        self.__solid_index = np.array(solid_indices, dtype=np.int64)
        self.__solid: FacetSet = geometry.facets.select(solid_indices)
        self.__solid_normals = normals[solid_indices].reshape(-1, self.dimension)
        self.__solid_friction = friction
        self.__platen_of_slot = platen_of_slot
        self.__platen_slots = np.flatnonzero(platen_of_slot >= 0)
        self.__regions = self.__facet_regions()
        self.__wall_spring: npt.NDArray[np.float64] = np.zeros((self.n, len(solid), self.dimension))
        self.__wall_rolling: npt.NDArray[np.float64] = np.zeros((self.n, ) + rotation_shape(len(solid), self.dimension))

    def __initial_orientations(self, particles: Sequence[AnyParticle]) -> npt.NDArray[np.float64]:
        """Orientations of the initial particles, with the quaternions normalised.

        Args:
            particles: the initial particles.

        Returns:
            The orientations, shape (n,) in 2D or (n, 4) in 3D.

        Raises:
            ValueError: if a particle has a zero quaternion as its orientation.
        """
        if self.dimension == 2:
            return np.array([p.orientation[0] for p in particles], dtype=np.float64)
        quaternions = np.array([p.orientation for p in particles], dtype=np.float64).reshape(-1, 4)
        lengths = np.linalg.norm(quaternions, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError(f"Particle {int(np.argmin(lengths))} has a zero quaternion as its orientation.")
        return np.asarray(quaternions / lengths[:, None])

    def __facet_regions(self) -> FacetRegions:
        """Split the solid behind the solid facets into the regions that polytopes press on.

        Returns:
            The regions, at the position the facets have in the geometry.
        """
        body_of_facet = [OPEN_BODY] * self.geometry.n_facets
        for index, platen in zip(self.__solid_index, self.__platen_of_slot):
            body_of_facet[int(index)] = int(platen)
        return facet_regions(self.geometry, [int(index) for index in self.__solid_index], body_of_facet)

    def __getstate__(self) -> Dict[str, Any]:
        """Return the attributes to pickle: all but those in the memory of a GPU, which are made again when needed.

        Returns:
            The attributes.
        """
        state = dict(self.__dict__)
        for name in ("_DEMSimulation__device_table", "_DEMSimulation__device_source", "_DEMSimulation__pose"):
            state[name] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled simulation, also one pickled before particles could have shapes, e.g. a checkpoint.

        Args:
            state: attributes of the pickled simulation.
        """
        self.__dict__.update(state)
        # a simulation from before the device existed measured its overlaps on the CPU
        self.__dict__.setdefault("_DEMSimulation__device", "cpu")
        self.__dict__.setdefault("_DEMSimulation__device_table", None)
        self.__dict__.setdefault("_DEMSimulation__device_source", None)
        if "_DEMSimulation__table" in state:
            return
        # a simulation of spheres from before shapes existed carries on as one of spheres of today
        d = self.dimension
        self.__table = PolytopeTable.build(d, [None] * self.n, self.radius)
        self.orientation = identity_orientations(self.n, d)
        self.__rain_shapes = None
        self.__rain_reach = self.radius_range[1]
        self.__pose = None
        self.__pose_index = np.zeros(0, dtype=np.int64)
        self.__regions = self.__facet_regions()
        self.__update_mass_properties()

    def __update_mass_properties(self) -> None:
        """Refresh mass and rotational inertia after the particle set has changed."""
        # every particle has the mass of the sphere of its radius; in a plane too, see the manual for why not discs
        self.mass = self.material.density * (4.0 / 3.0) * math.pi * self.radius**3
        factor = self.__table.inertia_factor
        if self.dimension == 2:
            self.inertia = factor * self.mass * self.radius**2
        else:
            self.inertia = factor * self.mass[:, None] * self.radius[:, None]**2
        self.inv_mass = 1.0 / self.mass
        self.inv_inertia = 1.0 / self.inertia
        self.__solids = np.flatnonzero(self.__table.polytope) if self.dimension == 3 else np.zeros(0, dtype=np.int64)

    def __sphere_inertia(self, particles: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Moment of inertia of spheres, the same about every axis.

        Args:
            particles: indices of spheres.

        Returns:
            Their moments of inertia [kg m^2].
        """
        return self.inertia[particles] if self.dimension == 2 else self.inertia[particles, 0]

    def __append_particles(self,
                           positions: npt.NDArray[np.float64],
                           radii: npt.NDArray[np.float64],
                           velocities: npt.NDArray[np.float64],
                           shapes: Optional[Sequence[Optional[Shape]]] = None,
                           orientations: Optional[npt.NDArray[np.float64]] = None) -> None:
        """Add particles at the end of the state arrays, so existing indices stay valid.

        Args:
            positions: centres of the new particles, shape (m, d) [m].
            radii: radii of the new particles [m].
            velocities: velocities of the new particles, shape (m, d) [m/s].
            shapes: shapes of the new particles, None for a sphere; all spheres if None.
            orientations: orientations of the new particles; the identity if None.
        """
        m = radii.size
        if m == 0:
            return
        new_shapes: Sequence[Optional[Shape]] = shapes if shapes is not None else [None] * m
        new_orientations = orientations if orientations is not None else identity_orientations(m, self.dimension)
        self.__table = self.__table.concatenate(PolytopeTable.build(self.dimension, new_shapes, radii))
        self.orientation = np.concatenate((self.orientation, new_orientations))
        self.pos = np.vstack((self.pos, positions))
        self.vel = np.vstack((self.vel, velocities))
        self.omega = np.concatenate((self.omega, np.zeros(rotation_shape(m, self.dimension))))
        self.radius = np.concatenate((self.radius, radii))
        self.force = np.vstack((self.force, np.zeros((m, self.dimension))))
        self.torque = np.concatenate((self.torque, np.zeros(rotation_shape(m, self.dimension))))
        self.__wall_spring = np.concatenate((self.__wall_spring, np.zeros((m, ) + self.__wall_spring.shape[1:])))
        self.__wall_rolling = np.concatenate((self.__wall_rolling, np.zeros((m, ) + self.__wall_rolling.shape[1:])))
        self.__update_mass_properties()
        self.__neighbours.invalidate()
        self.__forces_valid = False

    def __keep_particles(self, keep: npt.NDArray[np.bool_]) -> None:
        """Remove particles from the state arrays.

        Args:
            keep: True for every particle that stays in the simulation.
        """
        self.__table = self.__table.take(keep)
        self.orientation = self.orientation[keep]
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.omega = self.omega[keep]
        self.radius = self.radius[keep]
        self.force = self.force[keep]
        self.torque = self.torque[keep]
        self.__wall_spring = self.__wall_spring[keep]
        self.__wall_rolling = self.__wall_rolling[keep]
        self.__update_mass_properties()
        self.__neighbours.remove_particles(keep)

    def __remove_escaped_particles(self) -> None:
        """Remove the particles that left the domain and are farther than the escape distance from it."""
        inside = self.geometry.contains(self.pos)
        if np.all(inside):
            return
        outside = np.flatnonzero(~inside)
        distance = self.geometry.distance_to_facets(self.pos[outside]).min(axis=1)
        escaped = outside[distance > self.__escape_distance]
        if escaped.size == 0:
            return
        keep = np.ones(self.n, dtype=bool)
        keep[escaped] = False
        self.__keep_particles(keep)
        self.n_removed += int(escaped.size)

    def set_water(self, water: Optional[Water]) -> None:
        """Add, change or remove the pore water, e.g. to flood a packing that was deposited dry.

        The level and seepage velocity of the water are functions of the simulation time, so a change that should
        start now can be written relative to ``sim.time``. If the drag of the new water needs a shorter time step,
        the default time step is reduced.

        Args:
            water: new pore water; None makes the particles dry.

        Raises:
            ValueError: if the time step was given by the user and is too large to resolve the drag of the water.
        """
        self.water = water
        self.__pore_water = PoreWater(water, self.geometry, self.__r_max) if water is not None else None
        if self.__pore_water is not None:
            # the explicit coupling to the pore pressure needs a step short compared with a pressure wave
            water_dt = self.__pore_water.time_step_limit(self.__r_min, self.material.density)
            if self.dt > water_dt:
                if self.__user_dt:
                    raise ValueError(f"dt = {self.dt:.3e} s is too large for the pore pressure coupling; "
                                     f"use dt <= {water_dt:.3e} s.")
                self.dt = water_dt
        self.__forces_valid = False

    def excess_pore_pressure(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Excess pore pressure, above hydrostatic, at given points; zero without coupled water.

        Args:
            points: points, shape (n, d) [m].

        Returns:
            The excess pore pressure at each point [Pa].
        """
        d = self.dimension
        if self.__pore_water is None:
            return np.zeros(np.asarray(points).reshape(-1, d).shape[0])
        return self.__pore_water.excess_pressure(np.asarray(points, dtype=np.float64).reshape(-1, d))

    # ------------------------------------------------------------------ #
    #  Force evaluation                                                  #
    # ------------------------------------------------------------------ #
    def compute_forces(self) -> None:
        """Evaluate every force and torque for the current state, without advancing the contact history."""
        self.__evaluate_forces(accumulate=False)

    def __evaluate_forces(self, accumulate: bool) -> None:
        """Evaluate every force and torque for the current state.

        Args:
            accumulate: whether the relative sliding of the last time step is added to the tangential springs. Only
                the evaluation that follows a drift may do so; a re-evaluation of the same state must not.
        """
        self.force[:] = 0.0
        self.torque[:] = 0.0
        self.force[:, 1] -= self.mass * self.settings.gravity
        increment = self.dt if accumulate else 0.0
        if self.__pore_water is not None and self.n:
            fluid = self.__pore_water.forces(self.pos, self.vel, self.radius, self.time, self.settings.gravity,
                                             increment)
            # the drag is kept apart from the other forces and integrated implicitly in the kicks
            self.force += fluid.force
            self.__drag_coefficient = fluid.drag_coefficient
            self.__water_velocity = fluid.water_velocity
            self.fluid_force = fluid.force + fluid.drag_coefficient[:, None] * (fluid.water_velocity - self.vel)
            self.porosity = fluid.porosity
        else:
            self.__drag_coefficient = np.zeros(self.n)
            self.__water_velocity = np.zeros((self.n, self.dimension))
            self.fluid_force = np.zeros((self.n, self.dimension))
            self.porosity = np.full(self.n, np.nan)
        self.edge_forces[:] = 0.0
        for platen in self.platens:
            platen.reaction = 0.0
        if self.n:
            self.__place_polytopes()
            self.__particle_contacts(increment)
            self.__boundary_contacts(increment)
        self.__forces_valid = True

    def __particle_contacts(self, increment: float) -> None:
        """Add the particle-particle contact forces.

        Args:
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].
        """
        neighbours = self.__neighbours
        neighbours.update(self.pos, self.bounding_radius)
        touching, batches = self.__pair_contacts(increment)
        # a contact that opened loses its history, so a later re-closure starts from zero
        neighbours.spring[~touching] = 0.0
        neighbours.rolling[~touching] = 0.0
        n = self.n
        for batch in batches:
            neighbours.spring[batch.slot] = batch.forces.tangential_spring
            i, j = batch.first, batch.second
            force = batch.forces.force
            self.force += scatter_add(i, force, n) - scatter_add(j, force, n)
            if batch.levers is None:
                # both bodies receive a torque of the same sign, like two meshing gears
                lever = cross(batch.normal, batch.forces.tangential_force)
                self.torque -= scatter_add(i, scale(self.radius[i], lever), n) + scatter_add(
                    j, scale(self.radius[j], lever), n)
            else:
                lever_i, lever_j = batch.levers
                assert lever_j is not None
                self.torque += scatter_add(i, cross(lever_i, force), n) - scatter_add(j, cross(lever_j, force), n)
            if batch.rolling is not None:
                neighbours.rolling[batch.slot] = batch.rolling.rolling_spring
                self.torque += scatter_add(i, batch.rolling.moment, n) - scatter_add(j, batch.rolling.moment, n)

    def __pair_contacts(self, increment: float) -> Tuple[npt.NDArray[np.bool_], List[_ContactBatch]]:
        """Find the candidate pairs of the neighbour list that touch, and evaluate their contact law.

        Nothing is stored: the springs of the contacts are read, and their updated values returned. The pairs of two
        spheres, of a sphere and a polytope and of two polytopes are evaluated apart, each in a batch of its own.

        Args:
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].

        Returns:
            Which candidate pairs touch, and the batches of their contacts.
        """
        neighbours = self.__neighbours
        touching = np.zeros(neighbours.first.size, dtype=bool)
        batches: List[_ContactBatch] = []
        if neighbours.first.size == 0:
            return touching, batches
        polytope = self.polytope
        if not np.any(polytope):
            kinds = [np.arange(neighbours.first.size)]
            evaluators = [self.__sphere_pairs]
        else:
            # 0 for two spheres, 1 for a sphere and a polytope, 2 for two polytopes
            kind = polytope[neighbours.first].astype(np.int64) + polytope[neighbours.second]
            kinds = [np.flatnonzero(kind == k) for k in range(3)]
            evaluators = [self.__sphere_pairs, self.__mixed_pairs, self.__polytope_pairs]
        for slots, evaluate in zip(kinds, evaluators):
            if slots.size == 0:
                continue
            batch = evaluate(slots, increment)
            if batch is not None:
                assert batch.slot is not None
                touching[batch.slot] = True
                batches.append(batch)
        return touching, batches

    def __sphere_pairs(self, slots: npt.NDArray[np.int64], increment: float) -> Optional[_ContactBatch]:
        """Evaluate the candidate pairs of two spheres.

        Args:
            slots: positions of the pairs in the neighbour list.
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].

        Returns:
            The contacts of the pairs that touch, or None if none does.
        """
        neighbours = self.__neighbours
        first, second = neighbours.first[slots], neighbours.second[slots]
        delta = self.pos[first] - self.pos[second]
        dist_sq = np.einsum("ij,ij->i", delta, delta)
        reach = self.radius[first] + self.radius[second]
        touching = dist_sq < reach * reach
        contacts = np.flatnonzero(touching)
        if contacts.size == 0:
            return None

        i = first[contacts]
        j = second[contacts]
        dist = np.sqrt(dist_sq[contacts])
        normal = np.zeros((contacts.size, self.dimension))
        normal[:, 0] = 1.0
        # coincident centres have no defined normal, so they keep the arbitrary +x direction
        separated = dist > GEOMETRY_TOLERANCE * reach[contacts]
        normal[separated] = delta[contacts][separated] / dist[separated, None]
        radius_i, radius_j = self.radius[i], self.radius[j]
        # the contact point of i moves with omega_i x (-R_i n), that of j with omega_j x (R_j n)
        spin = scale(radius_i, self.omega[i]) + scale(radius_j, self.omega[j])
        relative_velocity = self.vel[i] - self.vel[j] - spin_cross(spin, normal)
        mass_i, mass_j = self.mass[i], self.mass[j]
        effective_radius = radius_i * radius_j / (radius_i + radius_j)
        overlap = reach[contacts] - dist
        forces = self.model.evaluate(normal, overlap, relative_velocity, neighbours.spring[slots[contacts]], increment,
                                     effective_radius, mass_i * mass_j / (mass_i + mass_j))
        rolling = None
        if self.model.rolling_friction > 0.0:
            # a particle rolls about its contact point, so its inertia there is I + m R^2
            inertia_i = self.__sphere_inertia(i) + mass_i * radius_i**2
            inertia_j = self.__sphere_inertia(j) + mass_j * radius_j**2
            rolling = self.model.rolling(forces, self.omega[i] - self.omega[j], neighbours.rolling[slots[contacts]],
                                         increment, effective_radius, inertia_i * inertia_j / (inertia_i + inertia_j))
        # the middle of the overlap lies inside particle i, R_i - delta / 2 from its centre against the normal
        point = self.pos[i] - scale(radius_i - 0.5 * overlap, normal)
        return _ContactBatch(i, j, normal, overlap, forces, rolling, point, slot=slots[contacts])

    def __mixed_pairs(self, slots: npt.NDArray[np.int64], increment: float) -> Optional[_ContactBatch]:
        """Evaluate the candidate pairs of a sphere and a polytope.

        The sphere presses on the polytope where the polytope comes closest to its centre, as on a wall: the Hertz
        law applies with the radius of the sphere as R*, and the contact force acts on the polytope in the middle of
        the overlap.

        Args:
            slots: positions of the pairs in the neighbour list.
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].

        Returns:
            The contacts of the pairs that touch, or None if none does.
        """
        neighbours = self.__neighbours
        first, second = neighbours.first[slots], neighbours.second[slots]
        sphere_first = ~self.polytope[first]
        sphere = np.where(sphere_first, first, second)
        body = np.where(sphere_first, second, first)
        distance, away = self.__distance_to_polytopes(self.pos[sphere], body)
        overlap = self.radius[sphere] - distance
        contacts = np.flatnonzero(overlap > 0.0)
        if contacts.size == 0:
            return None

        slots, overlap, away = slots[contacts], overlap[contacts], away[contacts]
        sphere, body, sphere_first = sphere[contacts], body[contacts], sphere_first[contacts]
        i, j = first[contacts], second[contacts]
        radius = self.radius[sphere]
        point = self.pos[sphere] - scale(radius - 0.5 * overlap, away)
        sphere_lever = -scale(radius, away)
        body_lever = point - self.pos[body]
        lever_i = np.where(sphere_first[:, None], sphere_lever, body_lever)
        lever_j = np.where(sphere_first[:, None], body_lever, sphere_lever)
        normal = np.where(sphere_first[:, None], away, -away)
        relative_velocity = (self.vel[i] + spin_cross(self.omega[i], lever_i) - self.vel[j] -
                             spin_cross(self.omega[j], lever_j))
        mass_i, mass_j = self.mass[i], self.mass[j]
        # a flat face of the polytope leaves the curvature of the sphere as that of the contact
        forces = self.model.evaluate(normal, overlap, relative_velocity, neighbours.spring[slots], increment, radius,
                                     mass_i * mass_j / (mass_i + mass_j))
        return _ContactBatch(i, j, normal, overlap, forces, None, point, (lever_i, lever_j), slots)

    def __polytope_pairs(self, slots: npt.NDArray[np.int64], increment: float) -> Optional[_ContactBatch]:
        """Evaluate the candidate pairs of two polytopes, from the region where they overlap.

        Args:
            slots: positions of the pairs in the neighbour list.
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].

        Returns:
            The contacts of the pairs that touch, or None if none does.
        """
        neighbours = self.__neighbours
        first, second = neighbours.first[slots], neighbours.second[slots]
        # the neighbour list holds pairs within the skin; only those whose bounding spheres overlap can touch
        delta = self.pos[first] - self.pos[second]
        reach = self.bounding_radius[first] + self.bounding_radius[second]
        near = np.einsum("ij,ij->i", delta, delta) < reach * reach
        slots, first, second = slots[near], first[near], second[near]
        if slots.size == 0:
            return None
        overlap = self.__polytope_overlap(first, second)
        size = norm(overlap.surface)
        contacts = np.flatnonzero((overlap.measure > 0.0) & (size > 0.0))
        if contacts.size == 0:
            return None

        slots, i, j, size = slots[contacts], first[contacts], second[contacts], size[contacts]
        point = overlap.centroid[contacts]
        normal = overlap.surface[contacts] / size[:, None]
        depth = overlap.measure[contacts] / size
        lever_i, lever_j = point - self.pos[i], point - self.pos[j]
        relative_velocity = (self.vel[i] + spin_cross(self.omega[i], lever_i) - self.vel[j] -
                             spin_cross(self.omega[j], lever_j))
        mass_i, mass_j = self.mass[i], self.mass[j]
        radius = overlap_contact_radius(size, self.dimension)
        forces = self.model.evaluate_overlap(normal, depth, radius, relative_velocity, neighbours.spring[slots],
                                             increment, mass_i * mass_j / (mass_i + mass_j))
        twisting = None
        if self.dimension == 3:
            inertia_i, inertia_j = self.__inertia_about(i, normal), self.__inertia_about(j, normal)
            twisting = self.model.twisting(forces, self.omega[i] - self.omega[j], neighbours.rolling[slots], increment,
                                           radius, inertia_i * inertia_j / (inertia_i + inertia_j))
        return _ContactBatch(i, j, normal, depth, forces, twisting, point, (lever_i, lever_j), slots)

    def __inertia_about(self, particles: npt.NDArray[np.int64],
                        axes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Moment of inertia of 3D polytope particles about axes through their centres.

        Args:
            particles: indices of polyhedral particles.
            axes: unit axis of each particle in space, shape (k, 3).

        Returns:
            The moments of inertia, n . R diag(I) R^T n [kg m^2].
        """
        pose = self.__pose
        assert pose is not None
        rotation = to_host(pose.rotation[self.__on_device(self.__pose_index[particles])])
        body = np.einsum("kji,kj->ki", rotation, axes)
        return np.asarray(np.einsum("ki,ki->k", self.inertia[particles], body * body))

    def __on_device(self, array: npt.NDArray[Any]) -> Any:
        """Return an array in the memory of the device that measures the overlaps of the polytopes."""
        return to_device(array, self.__array_module())

    def __array_module(self) -> ModuleType:
        """Return the array module of the device that measures the overlaps of the polytopes."""
        return gpu_module() if self.__device == "gpu" else np

    def __device_shapes(self) -> PolytopeTable:
        """Return the table of shapes in the memory of the device, copied there once for every new table."""
        if self.__device == "cpu":
            return self.__table
        if self.__device_table is None or self.__device_source is not self.__table:
            self.__device_table = self.__table.to_device(gpu_module())
            self.__device_source = self.__table
        return self.__device_table

    def __place_polytopes(self) -> None:
        """Move the shapes of the polytope particles to their current position and orientation, on the device."""
        rows = np.flatnonzero(self.polytope)
        self.__pose_index = np.full(self.n, -1, dtype=np.int64)
        self.__pose_index[rows] = np.arange(rows.size)
        self.__pose = None
        if rows.size:
            self.__pose = place(self.__device_shapes(), self.__on_device(rows), self.__on_device(self.pos),
                                self.__on_device(self.orientation))

    def __placed(self, particles: npt.NDArray[np.int64]) -> Tuple[Any, Any, Any]:
        """Return the vertices, face normals and face offsets of polytope particles at their current pose.

        Args:
            particles: indices of polytope particles.

        Returns:
            Their vertices, shape (k, v, d) [m], face normals, shape (k, f, d), and face offsets, shape (k, f) [m],
            on the device.
        """
        pose = self.__pose
        assert pose is not None
        index = self.__on_device(self.__pose_index[particles])
        return pose.vertices[index], pose.normals[index], pose.offsets[index]

    def __polyhedra(self, particles: npt.NDArray[np.int64]) -> Polyhedra:
        """Return polyhedral particles at their current pose, with their topology, for the overlap queries.

        Args:
            particles: indices of polyhedral particles.

        Returns:
            The polyhedra, on the device.
        """
        vertices, normals, offsets = self.__placed(particles)
        table = self.__device_shapes()
        rows = self.__on_device(particles)
        return Polyhedra(vertices, normals, offsets, table.edges[rows], table.edge_faces[rows], table.faces[rows])

    def __polytope_overlap(self, first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]) -> Overlap:
        """Return the overlaps of pairs of polytope particles.

        Args:
            first: particle i of each pair.
            second: particle j of each pair.

        Returns:
            The overlaps, with the contact surface pointing from j towards i, in the memory of the CPU.
        """
        if self.dimension == 2:
            return _on_host(polygon_overlap(*self.__placed(first), *self.__placed(second)))
        return _on_host(polyhedron_overlap(self.__polyhedra(first), self.__polyhedra(second)))

    def __region_overlap(self, particles: npt.NDArray[np.int64], normals: npt.NDArray[np.float64],
                         offsets: npt.NDArray[np.float64]) -> Overlap:
        """Return the overlaps of polytope particles with regions of solid behind the facets.

        Args:
            particles: polytope of each region.
            normals: outward unit normals of the half-spaces of each region, shape (k, r, d).
            offsets: offsets of those half-spaces, shape (k, r) [m].

        Returns:
            The overlaps, with the contact surface on the plane of the facet, in the memory of the CPU.
        """
        normals, offsets = self.__on_device(normals), self.__on_device(offsets)
        if self.dimension == 2:
            return _on_host(polygon_region_overlap(*self.__placed(particles), normals, offsets))
        return _on_host(polyhedron_region_overlap(self.__polyhedra(particles), normals, offsets))

    def __distance_to_polytopes(
            self, points: npt.NDArray[np.float64],
            particles: npt.NDArray[np.int64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the signed distance from points to polytope particles, and the direction away from them.

        Args:
            points: one point per particle, shape (k, d) [m].
            particles: indices of polytope particles.

        Returns:
            The distances, negative inside [m], and the unit normals at the closest points of the particles,
            pointing towards the points, in the memory of the CPU.
        """
        on_device = self.__on_device(points)
        if self.dimension == 2:
            distance, normal = polygon_distance(on_device, *self.__placed(particles))
        else:
            distance, normal = polyhedron_distance(on_device, self.__polyhedra(particles))
        return to_host(distance), to_host(normal)

    def __solid_state(self) -> Tuple[FacetSet, npt.NDArray[np.float64]]:
        """Return the solid facets at their current position and their velocities, shape (facets, d) [m/s]."""
        shifts = np.zeros((self.__solid.count, self.dimension))
        velocities = np.zeros_like(shifts)
        for slots, platen in zip(self.__platen_slot_groups, self.platens):
            shifts[slots] = platen.shift
            if platen.active:
                velocities[slots] = platen.velocity * platen.direction
        return self.__solid.moved(shifts), velocities

    def __redundant_contacts(self, particle: npt.NDArray[np.int64], facet: npt.NDArray[np.int64],
                             offset: npt.NDArray[np.float64], facets: FacetSet) -> npt.NDArray[np.bool_]:
        """Flag contacts at a point that another solid facet touching the same particle already accounts for.

        The closest point of a facet to a particle may lie on another solid facet as well, at a vertex or along an
        edge they share. That other facet then has a closest point of its own that is at least as close: either a
        different, closer point, which is where the particle really touches the boundary, or the same point, which
        of all the facets that report it only the one with the lowest index keeps. Without this rule a particle
        resting on a re-entrant corner, or on a floor made of several facets, would feel two contacts.

        Args:
            particle: particle of each contact, sorted.
            facet: solid facet of each contact.
            offset: vector from the closest point of the facet to the particle centre, shape (k, d) [m].
            facets: the solid facets at their current position.

        Returns:
            True for every contact that must be ignored.
        """
        redundant = np.zeros(particle.size, dtype=bool)
        firsts, seconds = [], []
        # contacts are sorted by particle, so the contacts of one particle are adjacent
        for lag in range(1, particle.size):
            index = np.arange(particle.size - lag)
            same = particle[index] == particle[index + lag]
            if not np.any(same):
                break
            firsts.append(index[same])
            seconds.append(index[same] + lag)
        if not firsts:
            return redundant
        a = np.concatenate(firsts + seconds)
        b = np.concatenate(seconds + firsts)
        points = self.pos[particle] - offset
        tolerance = GEOMETRY_TOLERANCE * self.radius_range[1]
        on_other = norm(facets.pair_offsets(points[a], facet[b])) <= tolerance
        distinct = norm(points[b] - points[a]) > tolerance
        covered = on_other & (distinct | (facet[b] < facet[a]))
        redundant[a[covered]] = True
        return redundant

    def __boundary_contacts(self, increment: float) -> None:
        """Add the contact forces between the particles and the solid facets, and the reactions on the platens.

        Args:
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].
        """
        if self.__solid.count == 0:
            return
        batches = self.__facet_contacts(increment)
        touching = np.zeros((self.n, self.__solid.count), dtype=bool)
        for batch in batches:
            touching[batch.first, batch.second] = True
        self.__wall_spring[~touching] = 0.0
        self.__wall_rolling[~touching] = 0.0
        if not batches:
            return

        n = self.n
        for batch in batches:
            particle, facet = batch.first, batch.second
            self.__wall_spring[particle, facet] = batch.forces.tangential_spring
            force = batch.forces.force
            self.force += scatter_add(particle, force, n)
            if batch.levers is None:
                lever = cross(batch.normal, batch.forces.tangential_force)
                self.torque -= scatter_add(particle, scale(self.radius[particle], lever), n)
            else:
                self.torque += scatter_add(particle, cross(batch.levers[0], force), n)
            if batch.rolling is not None:
                self.__wall_rolling[particle, facet] = batch.rolling.rolling_spring
                self.torque += scatter_add(particle, batch.rolling.moment, n)

        facet = np.concatenate([batch.second for batch in batches])
        force = np.concatenate([batch.forces.force for batch in batches])
        # the particles push each facet with the opposite of the force it exerts on them
        self.edge_forces[self.__solid_index] = -scatter_add(facet, force, self.__solid_index.size)
        if self.platens:
            platen = self.__platen_of_slot[facet]
            on_platen = platen >= 0
            if np.any(on_platen):
                directions = np.array([p.direction for p in self.platens])
                # the particles push the platen back with the opposite of the force it exerts on them
                along = np.einsum("ij,ij->i", force[on_platen], directions[platen[on_platen]])
                reactions = np.bincount(platen[on_platen], along, len(self.platens))
                for p, reaction in zip(self.platens, reactions):
                    p.reaction = float(reaction)

    def __facet_contacts(self, increment: float) -> List[_ContactBatch]:
        """Find the particles that touch a solid facet, and evaluate their contact law.

        Nothing is stored: the springs of the contacts are read, and their updated values returned.

        Args:
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].

        Returns:
            The batches of the contacts of the spheres and of the polytopes, whose second member is the slot of the
            solid facet; empty if no particle touches one.
        """
        facets, velocities = self.__solid_state()
        particle, facet, offset, dist_sq = facets.touching(self.pos, self.bounding_radius)
        batches: List[_ContactBatch] = []
        polytope = self.polytope[particle]
        if np.any(polytope):
            batch = self.__polytope_facet_contacts(particle[polytope], facet[polytope], velocities, increment)
            if batch is not None:
                batches.append(batch)
            sphere = ~polytope
            particle, facet, offset, dist_sq = particle[sphere], facet[sphere], offset[sphere], dist_sq[sphere]
        keep = ~self.__redundant_contacts(particle, facet, offset, facets)
        particle, facet, offset, dist_sq = particle[keep], facet[keep], offset[keep], dist_sq[keep]
        if particle.size == 0:
            return batches

        dist = np.sqrt(dist_sq)
        radius = self.radius[particle]
        # a centre exactly on the facet has no defined normal, so it is pushed along the inward normal
        normal = self.__solid_normals[facet].copy()
        separated = dist > GEOMETRY_TOLERANCE * radius
        normal[separated] = offset[separated] / dist[separated, None]
        relative_velocity = (self.vel[particle] - velocities[facet] -
                             spin_cross(scale(radius, self.omega[particle]), normal))
        overlap = radius - dist
        # a wall is a body of infinite radius and mass, so R* and m* reduce to the particle's
        forces = self.model.evaluate(normal, overlap, relative_velocity, self.__wall_spring[particle, facet], increment,
                                     radius, self.mass[particle], self.__solid_friction[facet])
        rolling = None
        if self.model.rolling_friction > 0.0:
            # facets never rotate, so the relative spin is the particle's own
            rolling = self.model.rolling(forces, self.omega[particle], self.__wall_rolling[particle, facet], increment,
                                         radius,
                                         self.__sphere_inertia(particle) + self.mass[particle] * radius**2)
        point = self.pos[particle] - scale(radius - 0.5 * overlap, normal)
        # the polytopes come first, so put the spheres in front to keep their contacts in the order of before
        return [_ContactBatch(particle, facet, normal, overlap, forces, rolling, point)] + batches

    def __polytope_facet_contacts(self, particle: npt.NDArray[np.int64], facet: npt.NDArray[np.int64],
                                  velocities: npt.NDArray[np.float64], increment: float) -> Optional[_ContactBatch]:
        """Evaluate the contacts of polytopes with the solid facets they may touch, from their overlap with the solid.

        Each particle overlaps the regions of solid behind the facets (``DEMcookies.regions``); the regions of one
        facet add up to its contact. The contacts of a particle with coplanar facets share the contact radius of
        their whole contact surface, as one contact would have.

        Args:
            particle: polytope of each candidate pair, sorted.
            facet: solid facet of each candidate pair.
            velocities: velocity of each solid facet, shape (s, d) [m/s].
            increment: time over which the sliding velocity is accumulated into the springs [s].

        Returns:
            The contacts of the pairs that touch, or None if none does.
        """
        regions = self.__regions
        shifts = np.zeros((self.__solid.count, self.dimension))
        for slots, platen in zip(self.__platen_slot_groups, self.platens):
            shifts[slots] = platen.shift
        offsets = regions.moved(shifts)
        # every region of the facet of every candidate, each counted for its candidate
        counts = regions.start[facet + 1] - regions.start[facet]
        owner = np.repeat(np.arange(particle.size), counts)
        region = np.repeat(regions.start[facet], counts) + np.arange(owner.size) - np.repeat(
            np.cumsum(counts) - counts, counts)
        rows = particle[owner]
        overlap = self.__region_overlap(rows, regions.normals[region], offsets[region])
        measure = np.bincount(owner, overlap.measure, particle.size)
        first = scatter_add(owner, overlap.measure[:, None] * overlap.centroid, particle.size)
        size = np.bincount(owner, np.einsum("ij,ij->i", overlap.surface, regions.normals[region, 0]), particle.size)
        contacts = np.flatnonzero((measure > 0.0) & (size > 0.0))
        if contacts.size == 0:
            return None

        particle, facet, measure, size = particle[contacts], facet[contacts], measure[contacts], size[contacts]
        point = first[contacts] / measure[:, None]
        normal = self.__solid_normals[facet]
        plane = offsets[regions.start[facet], 0]
        total = _coplanar_totals(particle, normal, plane, size, self.radius[particle])
        lever = point - self.pos[particle]
        relative_velocity = self.vel[particle] + spin_cross(self.omega[particle], lever) - velocities[facet]
        radius = overlap_contact_radius(total, self.dimension)
        friction = self.__solid_friction[facet]
        # a wall is a body of infinite mass, so m* reduces to the particle's
        forces = self.model.evaluate_overlap(normal, measure / size, radius, relative_velocity,
                                             self.__wall_spring[particle, facet], increment, self.mass[particle],
                                             size / total, friction)
        twisting = None
        if self.dimension == 3:
            # facets never rotate, so the relative spin is the particle's own
            twisting = self.model.twisting(forces, self.omega[particle], self.__wall_rolling[particle, facet],
                                           increment, radius, self.__inertia_about(particle, normal), friction)
        return _ContactBatch(particle, facet, normal, measure / size, forces, twisting, point, (lever, None))

    def __seat_points(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the points that a moving body is seated on: the spheres, and the vertices of the polytopes.

        Returns:
            The points, shape (k, d) [m], and their radii, zero for a vertex [m].
        """
        sphere = ~self.polytope
        points, radii = [self.pos[sphere]], [self.radius[sphere]]
        rows, vertices = frame_polygons(self.__table, self.pos, self.orientation)
        if rows.size:
            real = np.arange(vertices.shape[1])[None, :] < self.__table.vertex_count[rows, None]
            points.append(vertices[real])
            radii.append(np.zeros(int(real.sum())))
        return np.concatenate(points), np.concatenate(radii)

    def contacts(self) -> Tuple[ContactList, ContactList]:
        """Return the contacts of the current state, between particles and between particles and the solid facets.

        The contact law is evaluated as ``compute_forces`` evaluates it, without advancing the contact history, but
        nothing that the motion depends on is changed, so the contacts can be inspected or exported in the middle of a
        run without affecting it.

        Returns:
            The particle-particle contacts, and the particle-facet contacts, whose ``second`` is the index of the
            facet in the geometry.
        """
        pairs: List[_ContactBatch] = []
        walls: List[_ContactBatch] = []
        if self.n:
            # the list is already up to date after a step; it only changes after particles were added or removed
            self.__neighbours.update(self.pos, self.bounding_radius)
            self.__place_polytopes()
            _, pairs = self.__pair_contacts(0.0)
            if self.__solid.count:
                walls = self.__facet_contacts(0.0)
        return self.__contact_list(pairs), self.__contact_list(walls, self.__solid_index)

    def __contact_list(self,
                       batches: List[_ContactBatch],
                       facet_index: Optional[npt.NDArray[np.int64]] = None) -> ContactList:
        """Turn batches of evaluated contacts into a contact list.

        Args:
            batches: the contacts.
            facet_index: index in the geometry of each solid facet slot, for contacts with the facets; None for
                contacts between particles.

        Returns:
            The contacts.
        """
        d = self.dimension
        if not batches:
            none = np.zeros(0, dtype=np.int64)
            return ContactList(none, none.copy(), np.zeros((0, d)), np.zeros((0, d)), np.zeros(0), np.zeros(0),
                               np.zeros((0, d)))
        second = np.concatenate([batch.second for batch in batches])
        if facet_index is not None:
            second = facet_index[second]
        return ContactList(np.concatenate([batch.first for batch in batches]), second,
                           np.concatenate([batch.point for batch in batches]),
                           np.concatenate([batch.normal for batch in batches]),
                           np.concatenate([batch.overlap for batch in batches]),
                           np.concatenate([batch.forces.normal_force for batch in batches]),
                           np.concatenate([batch.forces.tangential_force for batch in batches]))

    # ------------------------------------------------------------------ #
    #  Time integration                                                  #
    # ------------------------------------------------------------------ #
    def __kick(self, half: float) -> None:
        """Update the particle velocities over half a time step, with the drag of the water taken implicitly.

        The drag ``beta (u - v)`` is evaluated with the velocity at the end of the kick, which keeps it stable however
        strong it is, e.g. in viscous water or for fine grains.

        Args:
            half: half the time step [s].
        """
        rate = half * self.inv_mass
        pushed = self.vel + rate[:, None] * (self.force + self.__drag_coefficient[:, None] * self.__water_velocity)
        self.vel = pushed / (1.0 + rate * self.__drag_coefficient)[:, None]

    def __spin(self, half: float) -> None:
        """Update the angular velocities over half a time step.

        A polyhedron spins about an axis that need not be one of its principal axes, so its angular momentum
        R diag(I) R^T omega, not its angular velocity, is what the torque changes.

        Args:
            half: half the time step [s].
        """
        if self.dimension == 2:
            self.omega += scale(self.inv_inertia, half * self.torque)
            return
        solids = self.__solids
        if solids.size:
            rotation = rotation_matrices(self.orientation[solids])
            momentum = apply_inertia(rotation, self.inertia[solids], self.omega[solids]) + half * self.torque[solids]
        self.omega += self.inv_inertia * (half * self.torque)
        if solids.size:
            self.omega[solids] = apply_inertia(rotation, self.inv_inertia[solids], momentum)

    def __turn(self, dt: float) -> None:
        """Turn the particles over a time step.

        The angular momentum of a polyhedron stays the same during the drift, while its angular velocity follows its
        orientation: the turn uses the angular velocity of the orientation half way, which makes it accurate to
        second order, and leaves the angular velocity that of the new orientation.

        Args:
            dt: time step [s].
        """
        if self.dimension == 2:
            self.orientation = self.orientation + dt * self.omega
            return
        solids = self.__solids
        turned = rotate(self.orientation, dt * self.omega)
        if solids.size:
            start = self.orientation[solids]
            momentum = apply_inertia(rotation_matrices(start), self.inertia[solids], self.omega[solids])
            middle = rotate(start, 0.5 * dt * self.omega[solids])
            spin = apply_inertia(rotation_matrices(middle), self.inv_inertia[solids], momentum)
            turned[solids] = rotate(start, dt * spin)
            self.omega[solids] = apply_inertia(rotation_matrices(turned[solids]), self.inv_inertia[solids], momentum)
        self.orientation = turned

    def step(self) -> None:
        """Advance the whole system by one velocity-Verlet step."""
        if not self.__forces_valid:
            self.__evaluate_forces(accumulate=False)
        dt = self.dt
        half = 0.5 * dt
        active = [p for p in self.platens if p.active]
        loaded = [p for p in active if not p.prescribed]

        # first half kick, with the forces of the previous evaluation
        self.__kick(half)
        self.__spin(half)
        for platen in loaded:
            platen.velocity += half * platen.acceleration(self.load_time, self.__gravity)

        # drift; a prescribed body goes straight to its position at the end of the step
        self.pos += dt * self.vel
        self.__turn(dt)
        for platen in active:
            if platen.prescribed:
                platen.drive(self.load_time, dt)
            else:
                platen.offset += dt * platen.velocity
        self.time += dt
        self.steps_taken += 1
        if active:
            self.load_time += dt
        if self.__has_open_facets and self.steps_taken % ESCAPE_CHECK_INTERVAL == 0:
            self.__remove_escaped_particles()

        self.__evaluate_forces(accumulate=True)

        # second half kick, with the new forces
        self.__kick(half)
        self.__spin(half)
        for platen in loaded:
            platen.velocity += half * platen.acceleration(self.load_time, self.__gravity)
        for observer in self.__observers:
            observer(self)

    def add_observer(self, observer: Observer) -> None:
        """Call a function with the simulation after every time step, e.g. a ``VTKWriter`` that exports it.

        The observers belong to the simulation: a simulation that is pickled, e.g. as a checkpoint, takes its
        observers along, so they must be picklable themselves, and they carry on when it is resumed.

        Args:
            observer: function of the simulation.
        """
        self.__observers = self.__observers + (observer, )

    def remove_observer(self, observer: Observer) -> None:
        """Stop calling an observer, e.g. before the simulation is stored for runs that export elsewhere.

        Args:
            observer: an observer of the simulation.

        Raises:
            ValueError: if the function does not observe the simulation.
        """
        if not any(o is observer for o in self.__observers):
            raise ValueError(f"{observer!r} does not observe this simulation.")
        self.__observers = tuple(o for o in self.__observers if o is not observer)

    # ------------------------------------------------------------------ #
    #  Diagnostics and recording                                         #
    # ------------------------------------------------------------------ #
    def mean_kinetic_energy(self) -> float:
        """Mean translational kinetic energy per particle [J]."""
        if self.n == 0:
            return 0.0
        speed_sq = np.einsum("ij,ij->i", self.vel, self.vel)
        return float(0.5 * np.sum(self.mass * speed_sq) / self.n)

    def bed_top(self) -> float:
        """Highest point of any particle [m]."""
        if self.n == 0:
            return float(self.geometry.bounds[0][1])
        points, radii = self.__seat_points()
        return float(np.max(points[:, 1] + radii))

    def __record_frame(self, phase: str) -> None:
        """Store a snapshot of the current state.

        Args:
            phase: name of the current stage.
        """
        speeds = norm(self.vel)
        platens = tuple(p.vertices for p in self.platens)
        level = evaluate_at(self.water.level, self.time) if self.water is not None else None
        # the table of shapes is never changed, only replaced, so a frame can share it
        shaped = bool(np.any(self.polytope))
        self.frames.append(
            Frame(self.time, self.pos.copy(), self.radius.copy(), speeds, platens, phase, level,
                  self.orientation.copy() if shaped else None, self.__table if shaped else None))

    def __record_loads(self) -> None:
        """Append the current state of every platen to its load curve."""
        for platen, curve in zip(self.platens, self.load_curves):
            curve.time.append(self.load_time)
            curve.applied.append(platen.applied_load(self.load_time))
            curve.displacement.append(platen.displacement)
            curve.reaction.append(platen.reaction)

    # ------------------------------------------------------------------ #
    #  High level drivers                                                #
    # ------------------------------------------------------------------ #
    def deposit(self, max_steps: int, min_steps: int = 500, record_every: int = 40) -> None:
        """Let the initial particles fall until the packing is at rest, or until the step limit.

        Args:
            max_steps: largest number of time steps.
            min_steps: number of time steps before the packing may be declared at rest.
            record_every: number of time steps between two recorded frames.

        Raises:
            ValueError: if record_every is not positive.
        """
        _check_record_every(record_every)
        for s in range(max_steps):
            self.step()
            if s % record_every == 0:
                self.__record_frame("deposition")
            if s >= min_steps and self.mean_kinetic_energy() < self.settings.quiet_tolerance:
                break

    def rain_until_full(self, settings: Optional[RainSettings] = None) -> int:
        """Rain particles into the domain in waves until it is full, then let the packing settle.

        Each wave inserts up to ``batch_size`` particles in a band near the top of the domain. A radius that cannot
        be placed is retried in the next waves, largest first, so the size distribution is not biased towards small
        particles. Once the packing reaches the band nothing can be inserted, and after ``patience`` empty waves in
        a row the domain is considered full. With ``shapes`` given to the simulation, each particle takes one of them
        at random, in a random orientation, and is placed so that the sphere that holds it overlaps nothing.

        Args:
            settings: deposition settings; defaults to ``RainSettings()``.

        Returns:
            The number of particles in the domain.
        """
        rain = settings or RainSettings()
        rng = np.random.default_rng(rain.seed)
        r_min, r_max = self.radius_range
        top = float(self.geometry.bounds[1][1])
        reach = self.__rain_reach
        band = rain.insertion_band or (top - 2.0 * reach, top - reach)
        pending: List[Tuple[float, Optional[Shape]]] = []
        empty_waves = 0
        steps = 0
        while empty_waves < rain.patience and self.n < rain.max_particles and steps < rain.max_steps:
            room = min(rain.batch_size, rain.max_particles - self.n)
            # only top the queue up, so radii that could not be placed are retried instead of replaced
            radii = rng.uniform(r_min, r_max, max(0, room - len(pending)))
            pending.extend(zip((float(r) for r in radii), self.__draw_shapes(rng, radii.size)))
            positions, placed, unplaced = self.__place_wave(pending[:room], band, rain.attempts, rng)
            pending = unplaced + pending[room:]
            velocities = np.zeros_like(positions)
            velocities[:, 1] = -rain.rain_speed
            radii = np.array([radius for radius, _ in placed], dtype=np.float64)
            shapes = [shape for _, shape in placed]
            # only shaped rain turns the particles, so that a rain of spheres draws what it did without shapes
            orientations = None
            if self.__rain_shapes is not None:
                orientations = random_orientations(rng, radii.size, self.dimension)
            self.__append_particles(positions, radii, velocities, shapes, orientations)
            empty_waves = 0 if radii.size else empty_waves + 1
            for _ in range(rain.steps_between):
                self.step()
                steps += 1
                if steps % rain.record_every == 0:
                    self.__record_frame("deposition")

        for s in range(rain.settle_steps):
            self.step()
            steps += 1
            if steps % rain.record_every == 0:
                self.__record_frame("deposition")
            if s >= rain.settle_min_steps and self.mean_kinetic_energy() < self.settings.quiet_tolerance:
                break
        return self.n

    def __draw_shapes(self, rng: np.random.Generator, count: int) -> List[Optional[Shape]]:
        """Draw the shapes of rained particles.

        Args:
            rng: random generator, which is not used when the simulation rains spheres only.
            count: number of particles.

        Returns:
            The shape of each particle, None for a sphere.
        """
        shapes = self.__rain_shapes
        if shapes is None:
            return [None] * count
        return [shapes[k] for k in rng.integers(len(shapes), size=count)]

    def __place_wave(
        self, items: Sequence[Tuple[float,
                                    Optional[Shape]]], band: Tuple[float,
                                                                   float], attempts: int, rng: np.random.Generator
    ) -> Tuple[npt.NDArray[np.float64], List[Tuple[float, Optional[Shape]]], List[Tuple[float, Optional[Shape]]]]:
        """Find non-overlapping positions in the insertion band for a wave of particles.

        A shaped particle is placed so that the sphere that holds it, whatever its orientation, overlaps nothing.

        Args:
            items: radius [m] and shape of each particle to place, None for a sphere.
            band: lowest and highest y of the particle centres [m].
            attempts: random positions tried per particle.
            rng: random generator.

        Returns:
            The centres of the placed particles, their radii and shapes, and those of the particles that could not be
            placed.
        """
        y_low, y_high = band
        low, high = self.geometry.bounds
        largest = max([self.__rain_reach] + ([float(self.bounding_radius.max())] if self.n else []))
        # the band is horizontal: x, and z in 3D, are drawn across the whole domain
        horizontal = [0] if self.dimension == 2 else [0, 2]
        # only particles whose centre is this close to the band can overlap a new particle
        nearby = self.pos[:, 1] > y_low - 2.0 * largest
        others = self.pos[nearby]
        other_reach = self.bounding_radius[nearby]
        facets, _ = self.__solid_state()
        moved = facets.select(self.__platen_slots)

        centres: List[npt.NDArray[np.float64]] = []
        placed: List[Tuple[float, Optional[Shape]]] = []
        unplaced: List[Tuple[float, Optional[Shape]]] = []
        for item in sorted(items, key=lambda entry: entry[0], reverse=True):
            radius, shape = item
            reach = radius if shape is None else radius * shape.circumradius
            if any(high[k] - low[k] <= 2.0 * reach for k in horizontal):
                unplaced.append(item)
                continue
            columns = [rng.uniform(low[0] + reach, high[0] - reach, attempts), rng.uniform(y_low, y_high, attempts)]
            if self.dimension == 3:
                columns.append(rng.uniform(low[2] + reach, high[2] - reach, attempts))
            candidates = np.column_stack(columns)
            fits = self.geometry.contains(candidates)
            fits &= self.geometry.distance_to_facets(candidates).min(axis=1) >= reach
            if moved.count:
                fits &= moved.distances(candidates).min(axis=1) >= reach
            if others.shape[0]:
                delta = candidates[:, None, :] - others[None, :, :]
                fits &= ~np.any(np.einsum("ijk,ijk->ij", delta, delta) < (other_reach + reach)**2, axis=1)
            found = np.flatnonzero(fits)
            if found.size == 0:
                unplaced.append(item)
                continue
            centre = candidates[found[0]]
            centres.append(centre)
            placed.append(item)
            others = np.vstack((others, centre[None, :]))
            other_reach = np.append(other_reach, reach)
        return np.array(centres, dtype=np.float64).reshape(-1, self.dimension), placed, unplaced

    def rest(self, duration: float, record_every: int = 60, phase: str = "settling") -> None:
        """Advance the simulation for a given time without changing the load, e.g. to let the packing relax.

        Args:
            duration: time to simulate [s].
            record_every: number of time steps between two recorded frames.
            phase: name of the stage, stored with the recorded frames.

        Raises:
            ValueError: if the duration is negative.
            ValueError: if record_every is not positive.
        """
        steps = self.__steps_for(duration)
        _check_record_every(record_every)
        for s in range(steps):
            self.step()
            if s % record_every == 0:
                self.__record_frame(phase)

    def seat_platens(self) -> None:
        """Lower every moving body with ``seat_on_bed`` onto the packing, and hold it there without releasing it.

        Seating a body before the domain is full lets particles be rained around it afterwards, e.g. into the cribs
        between railway sleepers, which would otherwise block it on its way down. The body stays fixed, and the load
        clock stopped, until ``compress`` releases it.

        Raises:
            ValueError: if the geometry has no loaded or prescribed platen.
            ValueError: if loading has already started.
        """
        if not self.platens:
            raise ValueError(f"The geometry has no LoadedPlaten or PrescribedPlaten {self.geometry.facet_name} to "
                             f"seat.")
        if any(p.active for p in self.platens):
            raise ValueError("Loading has already started; a released platen cannot be seated again.")
        for platen in self.platens:
            if platen.spec.seat_on_bed:
                platen.seat(*self.__seat_points())
        self.__forces_valid = False

    def compress(self, duration: float, record_every: int = 40) -> None:
        """Apply the loads of the loaded platens, and the displacements of the prescribed ones, for a given time.

        On the first call every platen is seated on the packing (if its ``seat_on_bed`` is set) and released, and
        the load clock starts from zero. Later calls continue the same load and displacement histories.

        Args:
            duration: time to simulate [s].
            record_every: number of time steps between two recorded frames and load curve points.

        Raises:
            ValueError: if the geometry has no loaded or prescribed platen.
            ValueError: if the duration is negative.
            ValueError: if record_every is not positive.
        """
        if not self.platens:
            raise ValueError(f"The geometry has no LoadedPlaten or PrescribedPlaten {self.geometry.facet_name} to "
                             f"apply a load with.")
        steps = self.__steps_for(duration)
        _check_record_every(record_every)
        if not any(p.active for p in self.platens):
            for platen in self.platens:
                if platen.spec.seat_on_bed and self.n:
                    platen.seat(*self.__seat_points())
                platen.activate()
            self.__forces_valid = False
            self.compute_forces()
            self.__record_loads()
        for s in range(steps):
            self.step()
            if s % record_every == 0:
                self.__record_frame("loading")
                self.__record_loads()

    def __steps_for(self, duration: float) -> int:
        """Convert a duration into a number of time steps.

        Args:
            duration: time to simulate [s].

        Returns:
            The number of time steps.

        Raises:
            ValueError: if the duration is negative.
        """
        if duration < 0.0:
            raise ValueError(f"duration must be non-negative, got {duration}.")
        return int(round(duration / self.dt))


_DIMENSION_NAMES = {2: "two-dimensional", 3: "three-dimensional"}


def _on_host(overlap: Overlap) -> Overlap:
    """Return overlaps in the memory of the CPU, copied from the GPU if they were measured there."""
    return Overlap(to_host(overlap.measure), to_host(overlap.centroid), to_host(overlap.surface))


def _coplanar_totals(particle: npt.NDArray[np.int64], normal: npt.NDArray[np.float64], plane: npt.NDArray[np.float64],
                     size: npt.NDArray[np.float64], radius: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Size of the whole contact surface that each contact of a particle with a facet is part of.

    The contacts of one particle with facets in the same plane, e.g. a floor made of several faces, are parts of one
    contact surface; every other contact is a surface of its own.

    Args:
        particle: particle of each contact, sorted.
        normal: unit normal of the facet of each contact, shape (k, d).
        plane: offset of the plane of that facet [m].
        size: size of the contact surface of each contact, an area in 3D and a length in 2D [m^2 or m].
        radius: radius of the particle of each contact, which scales the tolerance on the offsets [m].

    Returns:
        The total size of the coplanar contact surfaces of each contact, its own included.
    """
    total = size.copy()
    # contacts are sorted by particle, so the contacts of one particle are adjacent
    for lag in range(1, particle.size):
        index = np.arange(particle.size - lag)
        same = particle[index] == particle[index + lag]
        if not np.any(same):
            break
        a, b = index[same], index[same] + lag
        coplanar = ((norm(normal[a] - normal[b]) <= COPLANAR_TOLERANCE) &
                    (np.abs(plane[a] - plane[b]) <= COPLANAR_TOLERANCE * radius[a]))
        total[a[coplanar]] += size[b[coplanar]]
        total[b[coplanar]] += size[a[coplanar]]
    return total


def _check_record_every(record_every: int) -> None:
    """Check a recording interval.

    Args:
        record_every: number of time steps between two recorded frames.

    Raises:
        ValueError: if record_every is not positive.
    """
    if record_every < 1:
        raise ValueError(f"record_every must be at least 1, got {record_every}.")
