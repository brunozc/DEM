"""Time integration and the DEM simulation engine.

``VelocityVerlet`` is a small, self-contained integrator used to verify the scheme. ``DEMSimulation`` is the full
engine: it keeps the particle state in arrays, finds particle contacts with a cell grid and a Verlet neighbour list,
evaluates the Hertz-Mindlin forces between particles and against the solid facets of the domain, moves the loaded
platens, and advances everything in time with velocity Verlet. It runs in two or in three dimensions, as set by the
geometry: a polygon (``Geometry``) or a polyhedron (``Geometry3D``).
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .boundary import Open, Platen
from .contact import HertzMindlinModel, MaterialProperties, rayleigh_time_step
from .facets import FacetSet
from .geometry3d import Domain
from .water import PoreWater, Water, evaluate_at
from .globals import ESCAPE_CHECK_INTERVAL, GEOMETRY_TOLERANCE, GlobalSettings
from .neighbours import NeighbourList
from .particle_generation import AnyParticle
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
        platens: current facets of every loaded body, each of shape (k, m, d) holding the m vertices of its k
            facets: the start and end point of each edge in 2D, and the vertices of each face, padded with copies
            of its last vertex, in 3D [m].
        phase: name of the stage the frame belongs to.
        water_level: height of the free surface of the water [m], or None without water.
    """

    time: float
    positions: npt.NDArray[np.float64]
    radii: npt.NDArray[np.float64]
    speeds: npt.NDArray[np.float64]
    platens: Tuple[npt.NDArray[np.float64], ...]
    phase: str
    water_level: Optional[float] = None


@dataclass
class LoadCurve:
    """Load history recorded for one loaded body.

    Attributes:
        edges: indices of the geometry facets of the body, edges in 2D and faces in 3D; ``faces`` is the same tuple.
        time: time since loading started [s].
        applied: applied load [N].
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


class DEMSimulation:
    """A discrete element simulation with Hertz-Mindlin contacts, in a polygonal (2D) or polyhedral (3D) domain.

    The geometry sets the number of space dimensions d. Vectors have d components; rotations, angular velocities
    and torques are scalars about the out-of-plane axis in 2D and vectors in 3D. The y axis is vertical in both.

    Attributes:
        dimension: number of space dimensions, 2 or 3.
        pos: particle centres, shape (n, d) [m].
        vel: particle velocities, shape (n, d) [m/s].
        omega: particle angular velocities, shape (n,) in 2D or (n, 3) in 3D [rad/s].
        radius: particle radii [m].
        force: total force on each particle at the last evaluation, apart from the drag of the water, which is
            integrated implicitly, shape (n, d) [N].
        torque: total torque on each particle at the last evaluation, shape (n,) in 2D or (n, 3) in 3D [N m].
        fluid_force: force of the water on each particle at the last evaluation, including the drag, shape (n, d)
            [N].
        porosity: local volume porosity at each particle, computed only when there is water (NaN otherwise).
        edge_forces: force the particles exert on each facet of the geometry, an edge in 2D or a face in 3D, shape
            (facets, d) [N]; ``face_forces`` is the same array.
        platens: runtime state of every loaded body.
        frames: recorded snapshots.
        load_curves: recorded load history of every loaded body.
        time: simulation time [s].
        dt: time step [s].
    """

    def __init__(self,
                 geometry: Domain,
                 material: MaterialProperties,
                 radius_range: Optional[Tuple[float, float]] = None,
                 particles: Optional[Sequence[AnyParticle]] = None,
                 settings: Optional[GlobalSettings] = None,
                 dt: Optional[float] = None,
                 water: Optional[Water] = None) -> None:
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
                drag relaxation time if that is shorter.
            water: pore water; the particles are dry when None.

        Raises:
            ValueError: if a particle has a different number of dimensions than the geometry.
            ValueError: if neither a radius range nor particles are given.
            ValueError: if the radius range is not 0 < r_min <= r_max.
            ValueError: if a particle radius is not positive.
            ValueError: if a particle centre lies outside the domain.
            ValueError: if the time step is not positive.
            ValueError: if the time step is too large to resolve the drag of the water on the smallest particle.
        """
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
        self.pos: npt.NDArray[np.float64] = np.array([p.position for p in initial], dtype=np.float64).reshape(-1, d)
        self.vel: npt.NDArray[np.float64] = np.array([p.velocity for p in initial], dtype=np.float64).reshape(-1, d)
        self.radius: npt.NDArray[np.float64] = np.array([p.radius for p in initial], dtype=np.float64)
        if np.any(self.radius <= 0.0):
            raise ValueError("Particle radii must be positive.")
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

        r_min = min([self.radius_range[0]] + ([float(self.radius.min())] if self.n else []))
        r_max = max([self.radius_range[1]] + ([float(self.radius.max())] if self.n else []))
        if dt is not None and dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        self.__r_min, self.__r_max = r_min, r_max
        self.__user_dt = dt is not None
        self.dt = dt if dt is not None else rayleigh_time_step(material, r_min, self.settings.time_step_safety)
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

        # every loaded body moves all of its facets, which may be any subset of the solid facets
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
        self.__wall_spring: npt.NDArray[np.float64] = np.zeros((self.n, len(solid), self.dimension))
        self.__wall_rolling: npt.NDArray[np.float64] = np.zeros((self.n, ) + rotation_shape(len(solid), self.dimension))

    def __update_mass_properties(self) -> None:
        """Refresh mass and rotational inertia after the particle set has changed."""
        # particles are spheres in a plane, see the manual for why not discs
        self.mass = self.material.density * (4.0 / 3.0) * math.pi * self.radius**3
        self.inertia = 0.4 * self.mass * self.radius**2
        self.inv_mass = 1.0 / self.mass
        self.inv_inertia = 1.0 / self.inertia

    def __append_particles(self, positions: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                           velocities: npt.NDArray[np.float64]) -> None:
        """Add particles at the end of the state arrays, so existing indices stay valid.

        Args:
            positions: centres of the new particles, shape (m, d) [m].
            radii: radii of the new particles [m].
            velocities: velocities of the new particles, shape (m, d) [m/s].
        """
        m = radii.size
        if m == 0:
            return
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
            self.__particle_contacts(increment)
            self.__boundary_contacts(increment)
        self.__forces_valid = True

    def __particle_contacts(self, increment: float) -> None:
        """Add the particle-particle contact forces.

        Args:
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].
        """
        neighbours = self.__neighbours
        neighbours.update(self.pos, self.radius)
        first, second = neighbours.first, neighbours.second
        if first.size == 0:
            return
        delta = self.pos[first] - self.pos[second]
        dist_sq = np.einsum("ij,ij->i", delta, delta)
        reach = self.radius[first] + self.radius[second]
        touching = dist_sq < reach * reach
        # a contact that opened loses its history, so a later re-closure starts from zero
        neighbours.spring[~touching] = 0.0
        neighbours.rolling[~touching] = 0.0
        contacts = np.flatnonzero(touching)
        if contacts.size == 0:
            return

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
        result = self.model.evaluate(normal, reach[contacts] - dist, relative_velocity, neighbours.spring[contacts],
                                     increment, effective_radius, mass_i * mass_j / (mass_i + mass_j))
        neighbours.spring[contacts] = result.tangential_spring

        n = self.n
        force = result.force
        self.force += scatter_add(i, force, n) - scatter_add(j, force, n)
        # both bodies receive a torque of the same sign, like two meshing gears
        lever = cross(normal, result.tangential_force)
        self.torque -= scatter_add(i, scale(radius_i, lever), n) + scatter_add(j, scale(radius_j, lever), n)

        if self.model.rolling_friction > 0.0:
            # a particle rolls about its contact point, so its inertia there is I + m R^2
            inertia_i = self.inertia[i] + mass_i * radius_i**2
            inertia_j = self.inertia[j] + mass_j * radius_j**2
            rolling = self.model.rolling(result, self.omega[i] - self.omega[j], neighbours.rolling[contacts], increment,
                                         effective_radius, inertia_i * inertia_j / (inertia_i + inertia_j))
            neighbours.rolling[contacts] = rolling.rolling_spring
            self.torque += scatter_add(i, rolling.moment, n) - scatter_add(j, rolling.moment, n)

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
        facets, velocities = self.__solid_state()
        particle, facet, offset, dist_sq = facets.touching(self.pos, self.radius)
        keep = ~self.__redundant_contacts(particle, facet, offset, facets)
        particle, facet, offset, dist_sq = particle[keep], facet[keep], offset[keep], dist_sq[keep]
        touching = np.zeros((self.n, facets.count), dtype=bool)
        touching[particle, facet] = True
        self.__wall_spring[~touching] = 0.0
        self.__wall_rolling[~touching] = 0.0
        if particle.size == 0:
            return

        dist = np.sqrt(dist_sq)
        radius = self.radius[particle]
        # a centre exactly on the facet has no defined normal, so it is pushed along the inward normal
        normal = self.__solid_normals[facet].copy()
        separated = dist > GEOMETRY_TOLERANCE * radius
        normal[separated] = offset[separated] / dist[separated, None]
        relative_velocity = (self.vel[particle] - velocities[facet] -
                             spin_cross(scale(radius, self.omega[particle]), normal))
        # a wall is a body of infinite radius and mass, so R* and m* reduce to the particle's
        result = self.model.evaluate(normal, radius - dist, relative_velocity, self.__wall_spring[particle, facet],
                                     increment, radius, self.mass[particle], self.__solid_friction[facet])
        self.__wall_spring[particle, facet] = result.tangential_spring

        n = self.n
        force = result.force
        self.force += scatter_add(particle, force, n)
        self.torque -= scatter_add(particle, scale(radius, cross(normal, result.tangential_force)), n)
        if self.model.rolling_friction > 0.0:
            # facets never rotate, so the relative spin is the particle's own
            rolling = self.model.rolling(result, self.omega[particle], self.__wall_rolling[particle, facet], increment,
                                         radius, self.inertia[particle] + self.mass[particle] * radius**2)
            self.__wall_rolling[particle, facet] = rolling.rolling_spring
            self.torque += scatter_add(particle, rolling.moment, n)
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

    def step(self) -> None:
        """Advance the whole system by one velocity-Verlet step."""
        if not self.__forces_valid:
            self.__evaluate_forces(accumulate=False)
        dt = self.dt
        half = 0.5 * dt
        active = [p for p in self.platens if p.active]

        # first half kick, with the forces of the previous evaluation
        self.__kick(half)
        self.omega += scale(self.inv_inertia, half * self.torque)
        for platen in active:
            platen.velocity += half * platen.acceleration(self.load_time, self.__gravity)

        # drift
        self.pos += dt * self.vel
        for platen in active:
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
        self.omega += scale(self.inv_inertia, half * self.torque)
        for platen in active:
            platen.velocity += half * platen.acceleration(self.load_time, self.__gravity)

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
        return float(np.max(self.pos[:, 1] + self.radius))

    def __record_frame(self, phase: str) -> None:
        """Store a snapshot of the current state.

        Args:
            phase: name of the current stage.
        """
        speeds = norm(self.vel)
        platens = tuple(p.vertices for p in self.platens)
        level = evaluate_at(self.water.level, self.time) if self.water is not None else None
        self.frames.append(Frame(self.time, self.pos.copy(), self.radius.copy(), speeds, platens, phase, level))

    def __record_loads(self) -> None:
        """Append the current state of every platen to its load curve."""
        for platen, curve in zip(self.platens, self.load_curves):
            curve.time.append(self.load_time)
            curve.applied.append(platen.spec.load(self.load_time))
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
        a row the domain is considered full.

        Args:
            settings: deposition settings; defaults to ``RainSettings()``.

        Returns:
            The number of particles in the domain.
        """
        rain = settings or RainSettings()
        rng = np.random.default_rng(rain.seed)
        r_min, r_max = self.radius_range
        top = float(self.geometry.bounds[1][1])
        band = rain.insertion_band or (top - 2.0 * r_max, top - r_max)
        pending: List[float] = []
        empty_waves = 0
        steps = 0
        while empty_waves < rain.patience and self.n < rain.max_particles and steps < rain.max_steps:
            room = min(rain.batch_size, rain.max_particles - self.n)
            # only top the queue up, so radii that could not be placed are retried instead of replaced
            pending.extend(float(r) for r in rng.uniform(r_min, r_max, max(0, room - len(pending))))
            positions, radii, unplaced = self.__place_wave(pending[:room], band, rain.attempts, rng)
            pending = unplaced + pending[room:]
            velocities = np.zeros_like(positions)
            velocities[:, 1] = -rain.rain_speed
            self.__append_particles(positions, radii, velocities)
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

    def __place_wave(self, radii: Sequence[float], band: Tuple[float, float], attempts: int,
                     rng: np.random.Generator) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], List[float]]:
        """Find non-overlapping positions in the insertion band for a wave of particles.

        Args:
            radii: radii to place [m].
            band: lowest and highest y of the particle centres [m].
            attempts: random positions tried per particle.
            rng: random generator.

        Returns:
            The centres and radii of the placed particles, and the radii that could not be placed.
        """
        y_low, y_high = band
        low, high = self.geometry.bounds
        r_max = self.radius_range[1]
        # the band is horizontal: x, and z in 3D, are drawn across the whole domain
        horizontal = [0] if self.dimension == 2 else [0, 2]
        # only particles whose centre is this close to the band can overlap a new particle
        nearby = self.pos[:, 1] > y_low - 2.0 * r_max
        others = self.pos[nearby]
        other_radii = self.radius[nearby]
        facets, _ = self.__solid_state()
        moved = facets.select(self.__platen_slots)

        placed: List[Tuple[float, ...]] = []
        unplaced: List[float] = []
        for radius in sorted(radii, reverse=True):
            if any(high[k] - low[k] <= 2.0 * radius for k in horizontal):
                unplaced.append(radius)
                continue
            columns = [rng.uniform(low[0] + radius, high[0] - radius, attempts), rng.uniform(y_low, y_high, attempts)]
            if self.dimension == 3:
                columns.append(rng.uniform(low[2] + radius, high[2] - radius, attempts))
            candidates = np.column_stack(columns)
            fits = self.geometry.contains(candidates)
            fits &= self.geometry.distance_to_facets(candidates).min(axis=1) >= radius
            if moved.count:
                fits &= moved.distances(candidates).min(axis=1) >= radius
            if others.shape[0]:
                delta = candidates[:, None, :] - others[None, :, :]
                fits &= ~np.any(np.einsum("ijk,ijk->ij", delta, delta) < (other_radii + radius)**2, axis=1)
            found = np.flatnonzero(fits)
            if found.size == 0:
                unplaced.append(radius)
                continue
            centre = candidates[found[0]]
            placed.append((*(float(c) for c in centre), radius))
            others = np.vstack((others, centre[None, :]))
            other_radii = np.append(other_radii, radius)

        table = np.array(placed, dtype=np.float64).reshape(-1, self.dimension + 1)
        return table[:, :-1].copy(), table[:, -1].copy(), unplaced

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
        """Lower every loaded body with ``seat_on_bed`` onto the packing, and hold it there without releasing it.

        Seating a body before the domain is full lets particles be rained around it afterwards, e.g. into the cribs
        between railway sleepers, which would otherwise block it on its way down. The body stays fixed, and the load
        clock stopped, until ``compress`` releases it.

        Raises:
            ValueError: if the geometry has no loaded platen.
            ValueError: if loading has already started.
        """
        if not self.platens:
            raise ValueError(f"The geometry has no LoadedPlaten {self.geometry.facet_name} to seat.")
        if any(p.active for p in self.platens):
            raise ValueError("Loading has already started; a released platen cannot be seated again.")
        for platen in self.platens:
            if platen.spec.seat_on_bed:
                platen.seat(self.pos, self.radius)
        self.__forces_valid = False

    def compress(self, duration: float, record_every: int = 40) -> None:
        """Apply the loads of the loaded platens for a given time.

        On the first call every platen is seated on the packing (if its ``seat_on_bed`` is set) and released, and
        the load clock starts from zero. Later calls continue the same load history.

        Args:
            duration: time to simulate [s].
            record_every: number of time steps between two recorded frames and load curve points.

        Raises:
            ValueError: if the geometry has no loaded platen.
            ValueError: if the duration is negative.
            ValueError: if record_every is not positive.
        """
        if not self.platens:
            raise ValueError(f"The geometry has no LoadedPlaten {self.geometry.facet_name} to apply a load with.")
        steps = self.__steps_for(duration)
        _check_record_every(record_every)
        if not any(p.active for p in self.platens):
            for platen in self.platens:
                if platen.spec.seat_on_bed and self.n:
                    platen.seat(self.pos, self.radius)
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


def _check_record_every(record_every: int) -> None:
    """Check a recording interval.

    Args:
        record_every: number of time steps between two recorded frames.

    Raises:
        ValueError: if record_every is not positive.
    """
    if record_every < 1:
        raise ValueError(f"record_every must be at least 1, got {record_every}.")
