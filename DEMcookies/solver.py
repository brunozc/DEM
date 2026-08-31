"""Time integration and the DEM simulation engine.

``VelocityVerlet`` is a small, self-contained integrator used in the tests.
``DEMSimulation`` is the full engine: it builds the particle state arrays,
detects contacts with a uniform-grid (cell list) broad phase, evaluates the
Hertz-Mindlin forces against neighbouring particles, the three fixed walls and
the loading platen, and advances everything in time.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .boundary import Box, LoadingPlaten
from .contact import HertzMindlinModel, MaterialProperties, rayleigh_time_step
from .particle_generation import Particle

# half-stencil of neighbouring cells so every pair is visited exactly once
_NEIGHBOUR_OFFSETS = ((1, 0), (0, 1), (1, 1), (-1, 1))


class VelocityVerlet:
    """Canonical velocity-Verlet integrator for an acceleration callback."""

    def __init__(self, dt: float):
        self.dt = float(dt)

    def step(self, pos: np.ndarray, vel: np.ndarray,
             acceleration: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Advance ``pos``/``vel`` by one step.

        ``acceleration(pos, vel)`` must return the acceleration array.
        Returns the new position, velocity and acceleration.
        """
        dt = self.dt
        a0 = acceleration(pos, vel)
        vel_half = vel + 0.5 * a0 * dt
        pos_new = pos + vel_half * dt
        a1 = acceleration(pos_new, vel_half)
        vel_new = vel_half + 0.5 * a1 * dt
        return pos_new, vel_new, a1


class DEMSimulation:
    """A 2D discrete element simulation with Hertz-Mindlin contacts."""

    def __init__(self, box: Box, material: MaterialProperties,
                 particles: list[Particle], platen: LoadingPlaten,
                 load_fn: Callable[[float], float], gravity: float = 9.81,
                 dt: Optional[float] = None, quiet_tol: float = 1e-6,
                 radius_range: Optional[tuple[float, float]] = None):
        self.box = box
        self.material = material
        self.model = HertzMindlinModel(material)
        self.platen = platen
        self.load_fn = load_fn
        self.g = gravity
        self.quiet_tol = quiet_tol
        self._rain_speed = 0.3          # initial downward speed of rained particles

        self.n = len(particles)
        if particles:
            self.pos = np.array([[p.x, p.y] for p in particles], dtype=float)
            self.vel = np.array([[p.vx, p.vy] for p in particles], dtype=float)
            self.radius = np.array([p.radius for p in particles], dtype=float)
        else:
            self.pos = np.zeros((0, 2))
            self.vel = np.zeros((0, 2))
            self.radius = np.zeros(0)
        self.omega = np.zeros(self.n)

        if radius_range is not None:
            self.r_min, self.r_max = float(radius_range[0]), float(radius_range[1])
        else:
            self.r_min = float(self.radius.min())
            self.r_max = float(self.radius.max())

        self.force = np.zeros((self.n, 2))
        self.torque = np.zeros(self.n)
        self._recompute_derived()

        # tangential displacement history, keyed by contact
        self.tangential: dict = {}

        self.cell_size = 2.0 * self.r_max
        self.dt = dt if dt is not None else rayleigh_time_step(material, self.r_min)

        self.time = 0.0
        self.time_load = 0.0

        # outputs
        self.frames: list = []
        self.load_curve: list = []

    def _recompute_derived(self) -> None:
        """Refresh mass/inertia after the particle set has changed."""
        rho = self.material.density
        self.mass = rho * (4.0 / 3.0) * math.pi * self.radius ** 3
        self.inertia = 0.4 * self.mass * self.radius ** 2
        self.inv_mass = 1.0 / self.mass
        self.inv_inertia = 1.0 / self.inertia

    def _insert_rain_batch(self, batch_size: int, rng, attempts: int = 30) -> list:
        """Try to drop ``batch_size`` new particles in a band near the top.

        Returns the ``(x, y, radius)`` tuples that could be placed without
        overlapping existing particles - an empty list means the top of the
        box is blocked (i.e. the box is full).
        """
        r_min, r_max = self.r_min, self.r_max
        y_lo = self.box.height - 2.0 * r_max
        y_hi = self.box.height - r_max
        if self.n:
            mask = self.pos[:, 1] > (y_lo - 2.0 * r_max)
            top_pos = self.pos[mask]
            top_rad = self.radius[mask]
        else:
            top_pos = np.zeros((0, 2))
            top_rad = np.zeros(0)

        placed: list = []
        for _ in range(batch_size):
            for _attempt in range(attempts):
                r = float(rng.uniform(r_min, r_max))
                x = float(rng.uniform(r, self.box.width - r))
                y = float(rng.uniform(y_lo, y_hi))
                ok = True
                if top_pos.shape[0]:
                    dx = top_pos[:, 0] - x
                    dy = top_pos[:, 1] - y
                    sr = top_rad + r
                    if np.any(dx * dx + dy * dy < sr * sr):
                        ok = False
                if ok:
                    for px, py, pr in placed:
                        if (px - x) ** 2 + (py - y) ** 2 < (pr + r) ** 2:
                            ok = False
                            break
                if ok:
                    placed.append((x, y, r))
                    break
        return placed

    def _append_particles(self, placed: list) -> int:
        """Add the placed particles to the state arrays (indices grow at end)."""
        if not placed:
            return 0
        m = len(placed)
        new_pos = np.array([[x, y] for x, y, _ in placed], dtype=float)
        new_rad = np.array([r for _, _, r in placed], dtype=float)
        new_vel = np.zeros((m, 2))
        new_vel[:, 1] = -self._rain_speed
        self.pos = np.vstack([self.pos, new_pos])
        self.vel = np.vstack([self.vel, new_vel])
        self.omega = np.concatenate([self.omega, np.zeros(m)])
        self.radius = np.concatenate([self.radius, new_rad])
        self.force = np.vstack([self.force, np.zeros((m, 2))])
        self.torque = np.concatenate([self.torque, np.zeros(m)])
        self.n += m
        self._recompute_derived()
        return m

    # ------------------------------------------------------------------ #
    #  Broad phase                                                       #
    # ------------------------------------------------------------------ #
    def _candidate_pairs(self) -> list[tuple[int, int]]:
        """Return candidate contact pairs using a uniform grid."""
        inv = 1.0 / self.cell_size
        pos = self.pos
        cells: dict[tuple[int, int], list[int]] = {}
        for idx in range(self.n):
            key = (int(math.floor(pos[idx, 0] * inv)),
                   int(math.floor(pos[idx, 1] * inv)))
            cells.setdefault(key, []).append(idx)

        pairs: list[tuple[int, int]] = []
        for (cx, cy), members in cells.items():
            m = len(members)
            # within the same cell
            for a in range(m):
                ia = members[a]
                for b in range(a + 1, m):
                    pairs.append((ia, members[b]))
            # towards half of the neighbouring cells
            for ox, oy in _NEIGHBOUR_OFFSETS:
                neighbour = cells.get((cx + ox, cy + oy))
                if neighbour:
                    for ia in members:
                        for ib in neighbour:
                            pairs.append((ia, ib))
        return pairs

    # ------------------------------------------------------------------ #
    #  Force evaluation                                                  #
    # ------------------------------------------------------------------ #
    def compute_forces(self) -> None:
        """Evaluate every force and torque for the current configuration."""
        force = self.force
        torque = self.torque
        force[:] = 0.0
        torque[:] = 0.0
        force[:, 1] -= self.mass * self.g           # gravity

        pos = self.pos
        vel = self.vel
        rad = self.radius
        om = self.omega
        mass = self.mass
        model = self.model
        dt = self.dt
        old_tan = self.tangential
        new_tan: dict = {}

        # --- particle / particle ----------------------------------------
        for a, b in self._candidate_pairs():
            i, j = (a, b) if a < b else (b, a)
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = math.hypot(dx, dy)
            delta = rad[i] + rad[j] - dist
            if delta <= 0.0 or dist == 0.0:
                continue
            nx = dx / dist
            ny = dy / dist
            tpx = -ny
            tpy = nx
            spin = rad[i] * om[i] + rad[j] * om[j]
            vrx = (vel[i, 0] - vel[j, 0]) - spin * tpx
            vry = (vel[i, 1] - vel[j, 1]) - spin * tpy
            r_eff = rad[i] * rad[j] / (rad[i] + rad[j])
            m_eff = mass[i] * mass[j] / (mass[i] + mass[j])
            key = ("pp", i, j)
            xi = old_tan.get(key, (0.0, 0.0))
            fx, fy, ftx, fty, xix, xiy = model.contact_force(
                nx, ny, delta, vrx, vry, xi[0], xi[1], dt, r_eff, m_eff)
            new_tan[key] = (xix, xiy)
            force[i, 0] += fx
            force[i, 1] += fy
            force[j, 0] -= fx
            force[j, 1] -= fy
            cross = nx * fty - ny * ftx
            torque[i] -= rad[i] * cross
            torque[j] -= rad[j] * cross

        # --- particle / walls + platen ----------------------------------
        b_width = self.box.width
        platen = self.platen
        platen_fy = 0.0
        for i in range(self.n):
            xi_, yi_ = pos[i, 0], pos[i, 1]
            ri = rad[i]
            # left wall (x = 0), outward normal +x
            self._boundary_contact(i, 1.0, 0.0, ri - xi_, 0.0, 0.0,
                                   ("wL", i), old_tan, new_tan)
            # right wall (x = B), outward normal -x
            self._boundary_contact(i, -1.0, 0.0, xi_ + ri - b_width, 0.0, 0.0,
                                   ("wR", i), old_tan, new_tan)
            # bottom wall (y = 0), outward normal +y
            self._boundary_contact(i, 0.0, 1.0, ri - yi_, 0.0, 0.0,
                                   ("wB", i), old_tan, new_tan)
            # loading platen, normal pointing down onto the particle
            if platen.active:
                delta_p = yi_ + ri - platen.y
                if delta_p > 0.0:
                    _, fyp = self._boundary_contact(
                        i, 0.0, -1.0, delta_p, 0.0, platen.vy,
                        ("pl", i), old_tan, new_tan)
                    platen_fy -= fyp        # reaction on the platen (+y)

        platen.fy = platen_fy
        self.tangential = new_tan

    def _boundary_contact(self, i: int, nx: float, ny: float, delta: float,
                          vjx: float, vjy: float, key, old_tan: dict,
                          new_tan: dict):
        """Apply a Hertz-Mindlin contact between particle ``i`` and a boundary."""
        if delta <= 0.0:
            return 0.0, 0.0
        rad = self.radius
        om = self.omega
        vel = self.vel
        tpx = -ny
        tpy = nx
        spin = rad[i] * om[i]
        vrx = (vel[i, 0] - vjx) - spin * tpx
        vry = (vel[i, 1] - vjy) - spin * tpy
        xi = old_tan.get(key, (0.0, 0.0))
        fx, fy, ftx, fty, xix, xiy = self.model.contact_force(
            nx, ny, delta, vrx, vry, xi[0], xi[1], self.dt, rad[i], self.mass[i])
        new_tan[key] = (xix, xiy)
        self.force[i, 0] += fx
        self.force[i, 1] += fy
        cross = nx * fty - ny * ftx
        self.torque[i] -= rad[i] * cross
        return fx, fy

    # ------------------------------------------------------------------ #
    #  Time integration                                                  #
    # ------------------------------------------------------------------ #
    def _platen_acceleration(self) -> float:
        p = self.platen
        applied = self.load_fn(self.time_load)
        return (p.fy - applied - p.mass * p.gravity) / p.mass

    def step(self) -> None:
        """Advance the whole system by one velocity-Verlet step."""
        dt = self.dt
        half = 0.5 * dt
        platen = self.platen

        # first half kick (uses forces from the previous evaluation)
        self.vel[:, 0] += half * self.force[:, 0] * self.inv_mass
        self.vel[:, 1] += half * self.force[:, 1] * self.inv_mass
        self.omega += half * self.torque * self.inv_inertia
        if platen.active:
            platen.vy += half * self._platen_acceleration()

        # drift
        self.pos += self.vel * dt
        if platen.active:
            platen.y += platen.vy * dt
        self.time += dt
        if platen.active:
            self.time_load += dt

        # new forces
        self.compute_forces()

        # second half kick
        self.vel[:, 0] += half * self.force[:, 0] * self.inv_mass
        self.vel[:, 1] += half * self.force[:, 1] * self.inv_mass
        self.omega += half * self.torque * self.inv_inertia
        if platen.active:
            platen.vy += half * self._platen_acceleration()

    # ------------------------------------------------------------------ #
    #  Diagnostics and recording                                         #
    # ------------------------------------------------------------------ #
    def mean_kinetic_energy(self) -> float:
        if self.n == 0:
            return 0.0
        v2 = self.vel[:, 0] ** 2 + self.vel[:, 1] ** 2
        return float(0.5 * np.sum(self.mass * v2) / self.n)

    def _record_frame(self, phase: str) -> None:
        speed = np.hypot(self.vel[:, 0], self.vel[:, 1])
        platen_y = self.platen.y if self.platen.active else None
        self.frames.append((self.time, self.pos.copy(), self.radius.copy(),
                            speed, platen_y, phase))

    def _record_load(self) -> None:
        self.load_curve.append((self.time_load,
                                self.load_fn(self.time_load),
                                self.platen.displacement,
                                self.platen.fy))

    # ------------------------------------------------------------------ #
    #  High level driver                                                 #
    # ------------------------------------------------------------------ #
    def deposit(self, max_steps: int, min_steps: int = 500,
                record_every: int = 40) -> None:
        """Rain the particles down until the packing is quiet (or step cap)."""
        self.compute_forces()
        for s in range(max_steps):
            self.step()
            if s % record_every == 0:
                self._record_frame("deposition")
            if s >= min_steps and self.mean_kinetic_energy() < self.quiet_tol:
                break

    def bed_top(self) -> float:
        if self.n == 0:
            return 0.0
        return float(np.max(self.pos[:, 1] + self.radius))

    def rest(self, duration: float, record_every: int = 60) -> None:
        """Pause after deposition: let the packing settle for ``duration`` s.

        No load is applied during this interval; the simulation simply advances
        in time under gravity so the bed can relax before the platen is engaged.
        """
        steps = int(round(duration / self.dt))
        self.compute_forces()
        for s in range(steps):
            self.step()
            if s % record_every == 0:
                self._record_frame("settling")

    def compress(self, steps: int, record_every: int = 40) -> None:
        """Apply the time-dependent load through the platen."""
        self.platen.place(self.bed_top())
        self.compute_forces()
        for s in range(steps):
            self.step()
            if s % record_every == 0:
                self._record_frame("loading")
                self._record_load()

    def rain_until_full(self, batch_size: int = 15, steps_between: int = 450,
                        patience: int = 4, settle_steps: int = 4000,
                        settle_min: int = 500, max_particles: int = 2000,
                        max_steps: int = 300000, record_every: int = 60,
                        seed: int = 0) -> int:
        """Rain particles in waves until the box is full, then let it settle.

        Each wave drops up to ``batch_size`` particles in a band near the top
        of the box.  Once the growing packing reaches that band no more
        particles can be inserted; after ``patience`` consecutive empty waves
        the box is considered full and a final settling stage brings it to
        rest.  Returns the final number of particles.
        """
        rng = np.random.default_rng(seed)
        self.compute_forces()
        total = 0
        empty_waves = 0
        while empty_waves < patience and self.n < max_particles and total < max_steps:
            n_new = self._append_particles(self._insert_rain_batch(batch_size, rng))
            empty_waves = 0 if n_new else empty_waves + 1
            self.compute_forces()
            for _ in range(steps_between):
                self.step()
                total += 1
                if total % record_every == 0:
                    self._record_frame("deposition")

        # final settling until the packing is quiet
        for s in range(settle_steps):
            self.step()
            total += 1
            if total % record_every == 0:
                self._record_frame("deposition")
            if s >= settle_min and self.mean_kinetic_energy() < self.quiet_tol:
                break
        return self.n

