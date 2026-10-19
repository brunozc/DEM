"""Time integration and the DEM simulation engine.

``VelocityVerlet`` is a small, self-contained integrator used to verify the scheme. ``DEMSimulation`` is the full
engine: it keeps the particle state in arrays, finds particle contacts with a cell grid and a Verlet neighbour list,
evaluates the Hertz-Mindlin forces between particles and against the solid edges of the domain, moves the loaded
platens, and advances everything in time with velocity Verlet.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .boundary import Open, Platen
from .contact import HertzMindlinModel, MaterialProperties, rayleigh_time_step
from .geometry import Geometry
from .segments import project_onto_segments
from .water import PoreWater, Water, evaluate_at
from .globals import ESCAPE_CHECK_INTERVAL, GEOMETRY_TOLERANCE, VERTEX_TOLERANCE, GlobalSettings
from .neighbours import NeighbourList
from .particle_generation import Particle


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
        positions: particle centres, shape (n, 2) [m].
        radii: particle radii [m].
        speeds: particle speeds [m/s].
        platens: current edges of every loaded body, each of shape (k, 2, 2) holding the start and end point of its
            k edges [m].
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
        edges: indices of the geometry edges of the body.
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

    def as_array(self) -> npt.NDArray[np.float64]:
        """Return the curve as an array with columns time, applied, displacement and reaction."""
        return np.column_stack((self.time, self.applied, self.displacement, self.reaction)).reshape(-1, 4)


class DEMSimulation:
    """A 2D discrete element simulation with Hertz-Mindlin contacts in a polygonal domain.

    Attributes:
        pos: particle centres, shape (n, 2) [m].
        vel: particle velocities, shape (n, 2) [m/s].
        omega: particle angular velocities [rad/s].
        radius: particle radii [m].
        force: total force on each particle at the last evaluation, apart from the drag of the water, which is
            integrated implicitly, shape (n, 2) [N].
        fluid_force: force of the water on each particle at the last evaluation, including the drag, shape (n, 2)
            [N].
        porosity: local volume porosity at each particle, computed only when there is water (NaN otherwise).
        edge_forces: force the particles exert on each edge of the geometry, shape (edges, 2) [N].
        platens: runtime state of every loaded body.
        frames: recorded snapshots.
        load_curves: recorded load history of every loaded body.
        time: simulation time [s].
        dt: time step [s].
    """

    def __init__(self,
                 geometry: Geometry,
                 material: MaterialProperties,
                 radius_range: Optional[Tuple[float, float]] = None,
                 particles: Optional[Sequence[Particle]] = None,
                 settings: Optional[GlobalSettings] = None,
                 dt: Optional[float] = None,
                 water: Optional[Water] = None) -> None:
        """Set up the simulation.

        Args:
            geometry: domain, with a boundary condition on every edge.
            material: material shared by the particles and boundaries.
            radius_range: smallest and largest radius of rained particles [m]; defaults to the range of the
                initial particles.
            particles: initial particles; the domain starts empty when None.
            settings: numerical settings; defaults to ``GlobalSettings()``.
            dt: time step [s]; defaults to a fraction of the Rayleigh time step of the smallest particle, or of the
                drag relaxation time if that is shorter.
            water: pore water; the particles are dry when None.

        Raises:
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

        initial = list(particles or [])
        self.pos: npt.NDArray[np.float64] = np.array([[p.x, p.y] for p in initial], dtype=np.float64).reshape(-1, 2)
        self.vel: npt.NDArray[np.float64] = np.array([[p.vx, p.vy] for p in initial], dtype=np.float64).reshape(-1, 2)
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
            x, y = self.pos[outside[0]]
            raise ValueError(f"Particle {outside[0]} at ({x:.4g}, {y:.4g}) lies outside the domain.")
        self.radius_range = (float(radius_range[0]), float(radius_range[1]))

        self.omega: npt.NDArray[np.float64] = np.zeros(self.n)
        self.force: npt.NDArray[np.float64] = np.zeros((self.n, 2))
        self.torque: npt.NDArray[np.float64] = np.zeros(self.n)
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
        self.fluid_force: npt.NDArray[np.float64] = np.zeros((self.n, 2))
        self.porosity: npt.NDArray[np.float64] = np.full(self.n, np.nan)
        self.__drag_coefficient: npt.NDArray[np.float64] = np.zeros(self.n)
        self.__water_velocity: npt.NDArray[np.float64] = np.zeros((self.n, 2))
        self.set_water(water)
        self.__gravity = np.array([0.0, -self.settings.gravity])
        self.__escape_distance = self.settings.escape_margin * r_max
        self.__neighbours = NeighbourList(self.settings.neighbour_skin * r_min)
        self.__setup_boundaries()
        self.edge_forces: npt.NDArray[np.float64] = np.zeros((geometry.n_edges, 2))
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

    # ------------------------------------------------------------------ #
    #  Set-up                                                            #
    # ------------------------------------------------------------------ #
    def __setup_boundaries(self) -> None:
        """Build the arrays of the solid edges (walls and platens) the particles can touch."""
        conditions = self.geometry.boundary_conditions
        n_edges = len(conditions)
        starts = self.geometry.edge_starts
        ends = self.geometry.edge_ends
        normals = self.geometry.inward_normals

        solid = [(edge, condition) for edge, condition in enumerate(conditions) if not isinstance(condition, Open)]
        solid_edges = [edge for edge, _ in solid]
        segment_of_edge = {edge: s for s, edge in enumerate(solid_edges)}
        friction = np.empty(len(solid))
        for s, (edge, condition) in enumerate(solid):
            friction[s] = self.material.friction if condition.friction is None else condition.friction

        # every loaded body moves all of its edges, which may be any subset of the solid edges
        self.platens: List[Platen] = []
        self.__platen_segment_groups: List[npt.NDArray[np.int64]] = []
        platen_of_segment = np.full(len(solid), -1, dtype=np.int64)
        for spec, edges in self.geometry.platen_groups:
            segments = np.array([segment_of_edge[edge] for edge in edges], dtype=np.int64)
            platen_of_segment[segments] = len(self.platens)
            self.__platen_segment_groups.append(segments)
            self.platens.append(Platen(edges, spec, starts[list(edges)], ends[list(edges)], normals[list(edges)]))

        self.__has_open_edges = len(solid) < n_edges
        self.__segment_edge = np.array(solid_edges, dtype=np.int64)
        self.__segment_starts = starts[solid_edges].reshape(-1, 2)
        self.__segment_ends = ends[solid_edges].reshape(-1, 2)
        self.__segment_normals = normals[solid_edges].reshape(-1, 2)
        self.__segment_friction = friction
        self.__platen_of_segment = platen_of_segment
        self.__platen_segments = np.flatnonzero(platen_of_segment >= 0)
        self.__previous_segment = np.array([segment_of_edge.get((edge - 1) % n_edges, -1) for edge in solid_edges],
                                           dtype=np.int64)
        self.__next_segment = np.array([segment_of_edge.get((edge + 1) % n_edges, -1) for edge in solid_edges],
                                       dtype=np.int64)
        self.__wall_spring: npt.NDArray[np.float64] = np.zeros((self.n, len(solid), 2))
        self.__wall_rolling: npt.NDArray[np.float64] = np.zeros((self.n, len(solid)))

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
            positions: centres of the new particles, shape (m, 2) [m].
            radii: radii of the new particles [m].
            velocities: velocities of the new particles, shape (m, 2) [m/s].
        """
        m = radii.size
        if m == 0:
            return
        self.pos = np.vstack((self.pos, positions))
        self.vel = np.vstack((self.vel, velocities))
        self.omega = np.concatenate((self.omega, np.zeros(m)))
        self.radius = np.concatenate((self.radius, radii))
        self.force = np.vstack((self.force, np.zeros((m, 2))))
        self.torque = np.concatenate((self.torque, np.zeros(m)))
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
        distance = self.geometry.distance_to_edges(self.pos[outside]).min(axis=1)
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
            points: points, shape (n, 2) [m].

        Returns:
            The excess pore pressure at each point [Pa].
        """
        if self.__pore_water is None:
            return np.zeros(np.asarray(points).reshape(-1, 2).shape[0])
        return self.__pore_water.excess_pressure(np.asarray(points, dtype=np.float64).reshape(-1, 2))

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
            self.__water_velocity = np.zeros((self.n, 2))
            self.fluid_force = np.zeros((self.n, 2))
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
        normal = np.tile([1.0, 0.0], (contacts.size, 1))
        # coincident centres have no defined normal, so they keep the arbitrary +x direction
        separated = dist > GEOMETRY_TOLERANCE * reach[contacts]
        normal[separated] = delta[contacts][separated] / dist[separated, None]
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        spin = self.radius[i] * self.omega[i] + self.radius[j] * self.omega[j]
        relative_velocity = self.vel[i] - self.vel[j] - spin[:, None] * tangent
        radius_i, radius_j = self.radius[i], self.radius[j]
        mass_i, mass_j = self.mass[i], self.mass[j]
        effective_radius = radius_i * radius_j / (radius_i + radius_j)
        result = self.model.evaluate(normal, reach[contacts] - dist, relative_velocity, neighbours.spring[contacts],
                                     increment, effective_radius, mass_i * mass_j / (mass_i + mass_j))
        neighbours.spring[contacts] = result.tangential_spring

        n = self.n
        force = result.force
        self.force[:, 0] += np.bincount(i, force[:, 0], n) - np.bincount(j, force[:, 0], n)
        self.force[:, 1] += np.bincount(i, force[:, 1], n) - np.bincount(j, force[:, 1], n)
        # both bodies receive a torque of the same sign, like two meshing gears
        cross = normal[:, 0] * result.tangential_force[:, 1] - normal[:, 1] * result.tangential_force[:, 0]
        self.torque -= np.bincount(i, radius_i * cross, n) + np.bincount(j, radius_j * cross, n)

        if self.model.rolling_friction > 0.0:
            # a particle rolls about its contact point, so its inertia there is I + m R^2
            inertia_i = self.inertia[i] + mass_i * radius_i**2
            inertia_j = self.inertia[j] + mass_j * radius_j**2
            rolling = self.model.rolling(result, self.omega[i] - self.omega[j], neighbours.rolling[contacts], increment,
                                         effective_radius, inertia_i * inertia_j / (inertia_i + inertia_j))
            neighbours.rolling[contacts] = rolling.rolling_spring
            self.torque += np.bincount(i, rolling.moment, n) - np.bincount(j, rolling.moment, n)

    def __segment_state(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the current start points, end points and velocities of the solid edges."""
        starts = self.__segment_starts.copy()
        ends = self.__segment_ends.copy()
        velocities = np.zeros_like(starts)
        for segments, platen in zip(self.__platen_segment_groups, self.platens):
            starts[segments] = platen.starts
            ends[segments] = platen.ends
            if platen.active:
                velocities[segments] = platen.velocity * platen.direction
        return starts, ends, velocities

    def __redundant_vertex_contacts(self, t: npt.NDArray[np.float64], starts: npt.NDArray[np.float64],
                                    ends: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Flag contacts with a vertex that another solid edge already accounts for.

        A vertex shared by two solid edges belongs to both. If the closest point of one edge to a particle is the
        shared vertex, the other edge either has a closer point of its own or reports the same vertex, so in both
        cases the contact is already counted once by the other edge.

        Args:
            t: segment parameter of the closest point of every edge to every particle, shape (n, s).
            starts: current start points of the solid edges, shape (s, 2) [m].
            ends: current end points of the solid edges, shape (s, 2) [m].

        Returns:
            True for every (particle, edge) contact that must be ignored, shape (n, s).
        """
        previous = self.__previous_segment
        has_previous = previous >= 0
        shares_start = np.zeros(starts.shape[0], dtype=bool)
        gap = ends[previous[has_previous]] - starts[has_previous]
        shares_start[has_previous] = np.hypot(gap[:, 0], gap[:, 1]) <= GEOMETRY_TOLERANCE * self.radius_range[1]
        following = self.__next_segment
        shares_end = np.zeros_like(shares_start)
        shares_end[following >= 0] = shares_start[following[following >= 0]]

        # a start vertex is always left to the previous edge, which reports it too or has a closer point
        redundant = (t <= VERTEX_TOLERANCE) & shares_start[None, :]
        # an end vertex is kept only if the next edge reports that same vertex as its closest point
        next_t = t[:, np.maximum(following, 0)]
        redundant |= (t >= 1.0 - VERTEX_TOLERANCE) & shares_end[None, :] & (next_t > VERTEX_TOLERANCE)
        return redundant

    def __boundary_contacts(self, increment: float) -> None:
        """Add the contact forces between the particles and the solid edges, and the reactions on the platens.

        Args:
            increment: time over which the sliding and rolling velocities are accumulated into the springs [s].
        """
        if self.__segment_starts.shape[0] == 0:
            return
        starts, ends, velocities = self.__segment_state()
        t, offset = project_onto_segments(self.pos, starts, ends)
        dist_sq = offset[..., 0]**2 + offset[..., 1]**2
        touching = dist_sq < (self.radius**2)[:, None]
        touching &= ~self.__redundant_vertex_contacts(t, starts, ends)
        self.__wall_spring[~touching] = 0.0
        self.__wall_rolling[~touching] = 0.0
        if not np.any(touching):
            return

        particle, segment = np.nonzero(touching)
        dist = np.sqrt(dist_sq[particle, segment])
        radius = self.radius[particle]
        # a centre exactly on the edge has no defined normal, so it is pushed along the inward normal
        normal = self.__segment_normals[segment].copy()
        separated = dist > GEOMETRY_TOLERANCE * radius
        normal[separated] = offset[particle[separated], segment[separated]] / dist[separated, None]
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        relative_velocity = (self.vel[particle] - velocities[segment] -
                             (radius * self.omega[particle])[:, None] * tangent)
        # a wall is a body of infinite radius and mass, so R* and m* reduce to the particle's
        result = self.model.evaluate(normal, radius - dist, relative_velocity, self.__wall_spring[particle, segment],
                                     increment, radius, self.mass[particle], self.__segment_friction[segment])
        self.__wall_spring[particle, segment] = result.tangential_spring

        n = self.n
        force = result.force
        self.force[:, 0] += np.bincount(particle, force[:, 0], n)
        self.force[:, 1] += np.bincount(particle, force[:, 1], n)
        cross = normal[:, 0] * result.tangential_force[:, 1] - normal[:, 1] * result.tangential_force[:, 0]
        self.torque -= np.bincount(particle, radius * cross, n)
        if self.model.rolling_friction > 0.0:
            # edges never rotate, so the relative spin is the particle's own
            rolling = self.model.rolling(result, self.omega[particle], self.__wall_rolling[particle, segment],
                                         increment, radius, self.inertia[particle] + self.mass[particle] * radius**2)
            self.__wall_rolling[particle, segment] = rolling.rolling_spring
            self.torque += np.bincount(particle, rolling.moment, n)
        # the particles push each edge with the opposite of the force it exerts on them
        segments = self.__segment_edge.size
        self.edge_forces[self.__segment_edge, 0] = -np.bincount(segment, force[:, 0], segments)
        self.edge_forces[self.__segment_edge, 1] = -np.bincount(segment, force[:, 1], segments)

        if self.platens:
            platen = self.__platen_of_segment[segment]
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
        self.omega += half * self.torque * self.inv_inertia
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
        if self.__has_open_edges and self.steps_taken % ESCAPE_CHECK_INTERVAL == 0:
            self.__remove_escaped_particles()

        self.__evaluate_forces(accumulate=True)

        # second half kick, with the new forces
        self.__kick(half)
        self.omega += half * self.torque * self.inv_inertia
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
            return self.geometry.bounding_box[1]
        return float(np.max(self.pos[:, 1] + self.radius))

    def __record_frame(self, phase: str) -> None:
        """Store a snapshot of the current state.

        Args:
            phase: name of the current stage.
        """
        speeds = np.hypot(self.vel[:, 0], self.vel[:, 1])
        platens = tuple(np.stack((p.starts, p.ends), axis=1) for p in self.platens)
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
        band = rain.insertion_band or (self.geometry.bounding_box[3] - 2.0 * r_max,
                                       self.geometry.bounding_box[3] - r_max)
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
        x_min, _, x_max, _ = self.geometry.bounding_box
        r_max = self.radius_range[1]
        # only particles whose centre is this close to the band can overlap a new particle
        nearby = self.pos[:, 1] > y_low - 2.0 * r_max
        others = self.pos[nearby]
        other_radii = self.radius[nearby]
        starts, ends, _ = self.__segment_state()
        moved = self.__platen_segments

        placed: List[Tuple[float, float, float]] = []
        unplaced: List[float] = []
        for radius in sorted(radii, reverse=True):
            if x_max - x_min <= 2.0 * radius:
                unplaced.append(radius)
                continue
            xs = rng.uniform(x_min + radius, x_max - radius, attempts)
            ys = rng.uniform(y_low, y_high, attempts)
            candidates = np.column_stack((xs, ys))
            fits = self.geometry.contains(candidates)
            fits &= self.geometry.distance_to_edges(candidates).min(axis=1) >= radius
            if moved.size:
                _, offset = project_onto_segments(candidates, starts[moved], ends[moved])
                fits &= np.hypot(offset[..., 0], offset[..., 1]).min(axis=1) >= radius
            if others.shape[0]:
                delta = candidates[:, None, :] - others[None, :, :]
                fits &= ~np.any(np.einsum("ijk,ijk->ij", delta, delta) < (other_radii + radius)**2, axis=1)
            found = np.flatnonzero(fits)
            if found.size == 0:
                unplaced.append(radius)
                continue
            x, y = candidates[found[0]]
            placed.append((float(x), float(y), radius))
            others = np.vstack((others, [[x, y]]))
            other_radii = np.append(other_radii, radius)

        table = np.array(placed, dtype=np.float64).reshape(-1, 3)
        return table[:, :2].copy(), table[:, 2].copy(), unplaced

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
            raise ValueError("The geometry has no LoadedPlaten edge to seat.")
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
            raise ValueError("The geometry has no LoadedPlaten edge to apply a load with.")
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


def _check_record_every(record_every: int) -> None:
    """Check a recording interval.

    Args:
        record_every: number of time steps between two recorded frames.

    Raises:
        ValueError: if record_every is not positive.
    """
    if record_every < 1:
        raise ValueError(f"record_every must be at least 1, got {record_every}.")
