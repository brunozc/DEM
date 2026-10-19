"""Pore water: a water table that exerts buoyancy and drag on the particles, optionally with an excess pore pressure.

Two levels of coupling between the water and the particles are available.

* Drained (default): below a horizontal free surface the pore pressure is hydrostatic, and any flow through the
  packing is prescribed. Each submerged particle feels buoyancy and the force of the water flowing past it.
* Coupled: in addition, an excess pore pressure is solved on a coarse grid from a Biot storage equation with Darcy
  flow. It builds up when the skeleton changes volume faster than the water can flow in or out, and dissipates by
  flow, so undrained loading and consolidation can be modelled. In a dense packing the skeleton carries the water in
  its pores along, and the gradient of the excess pressure pushes on the particles.

The drag on a particle in a packing follows the correlation of Di Felice (1994). In a packing the flow that drags
the particles also needs a pressure gradient, which pushes on their volume; for a uniform flow the two together are
the drag divided by the porosity (Zhou et al., 2010), which is the force applied here. The local porosity is
averaged on the coarse grid from the disc areas of the particles and converted to the porosity of an equivalent
packing of spheres, which is what the drag correlation expects.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .boundary import Open
from .geometry import Geometry
from .globals import (ACTIVE_NODE_FRACTION, CARRIED_SOLID_FRACTION_DENSE, CARRIED_SOLID_FRACTION_DILUTE,
                      COUPLING_TIME_STEP_FRACTION, DENSE_POROSITY, MAX_POROSITY, MIN_POROSITY, PERMEABILITY_CONSTANT,
                      POROSITY_SAMPLES_PER_CELL, PRESSURE_SOLVER_ITERATIONS, PRESSURE_SOLVER_TOLERANCE)
from .segments import project_onto_segments

TimeFunction = Union[float, Callable[[float], float]]


def evaluate_at(value: TimeFunction, time: float) -> float:
    """Evaluate a quantity that is either constant or a function of time.

    Args:
        value: constant, or function of the simulation time.
        time: simulation time [s].

    Returns:
        The value at that time.
    """
    return float(value(time)) if callable(value) else float(value)


def _check_time_function(name: str, value: TimeFunction) -> None:
    """Check that a quantity is a finite number or a callable.

    Args:
        name: name of the quantity, for the error message.
        value: quantity to check.

    Raises:
        ValueError: if the quantity is neither callable nor a finite number.
    """
    if callable(value):
        return
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Water {name} must be a finite number or a function of time, got {value!r}.")


@dataclass(frozen=True, eq=False)
class Water:
    """Pore water below a horizontal free surface.

    Attributes:
        level: height of the free surface [m]; a constant or a function of the simulation time [s].
        density: density of the water [kg/m^3].
        viscosity: dynamic viscosity of the water [Pa s].
        seepage_velocity: superficial (Darcy) velocity of a vertical flow imposed through the packing, positive
            upwards [m/s]; a constant or a function of the simulation time [s]. The flow is uniform, as in a
            permeameter whose floor and top let the water through.
        cell_size: side of the grid cells over which the porosity, and the pore pressure, are averaged [m]; None uses
            two largest particle diameters.
        coupled: whether an excess pore pressure is solved, so that the water resists fast changes of volume of the
            packing; otherwise the water is drained.
        bulk_modulus: bulk modulus of the pore water, used when coupled [Pa]. Like the Young's modulus of the
            particles it is scaled down to keep the time step affordable; it should stay large compared with the
            bulk modulus of the packing, so that undrained loading is carried by the water.

    To make DEM-size grains drain like a finer soil, increase the viscosity: the permeability enters only as k / mu,
    so a viscosity N times that of water behaves like a soil N times less permeable, for drag, settling and pore
    pressure alike.
    """

    level: TimeFunction
    density: float = 1000.0
    viscosity: float = 1.0e-3
    seepage_velocity: TimeFunction = 0.0
    cell_size: Optional[float] = None
    coupled: bool = False
    bulk_modulus: float = 1.0e7

    def __post_init__(self) -> None:
        """Validate the water.

        Raises:
            ValueError: if the level is neither callable nor a finite number.
            ValueError: if the seepage velocity is neither callable nor a finite number.
            ValueError: if the density is not positive.
            ValueError: if the viscosity is not positive.
            ValueError: if the cell size is not positive.
            ValueError: if the bulk modulus is not positive.
        """
        _check_time_function("level", self.level)
        _check_time_function("seepage_velocity", self.seepage_velocity)
        if self.density <= 0.0:
            raise ValueError(f"Water density must be positive, got {self.density}.")
        if self.viscosity <= 0.0:
            raise ValueError(f"Water viscosity must be positive, got {self.viscosity}.")
        if self.cell_size is not None and self.cell_size <= 0.0:
            raise ValueError(f"Water cell_size must be positive, got {self.cell_size}.")
        if self.bulk_modulus <= 0.0:
            raise ValueError(f"Water bulk_modulus must be positive, got {self.bulk_modulus}.")


def submerged_fraction(heights: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                       level: float) -> npt.NDArray[np.float64]:
    """Fraction of the volume of each sphere that lies below the free surface.

    Args:
        heights: heights of the sphere centres [m].
        radii: sphere radii [m].
        level: height of the free surface [m].

    Returns:
        The submerged volume fractions, between 0 and 1.
    """
    # depth of water above the lowest point of the sphere, i.e. the height of the submerged cap
    cap = np.clip(level - (heights - radii), 0.0, 2.0 * radii)
    return np.asarray(cap**2 * (3.0 * radii - cap) / (4.0 * radii**3))


def porosity_3d(porosity_2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert the porosity of a planar layer of discs into that of an equivalent packing of spheres.

    The conversion maps the hexagonal close packing of discs onto the face-centred cubic packing of spheres and an
    empty layer onto pure water.

    Args:
        porosity_2d: area porosity of the disc layer.

    Returns:
        The volume porosity of the equivalent sphere packing.
    """
    solid = np.clip(1.0 - porosity_2d, 0.0, 1.0)
    return np.asarray(1.0 - 2.0 / math.sqrt(math.sqrt(3.0) * math.pi) * solid**1.5)


def interaction_coefficient(speed: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                            porosity: npt.NDArray[np.float64], water: Water) -> npt.NDArray[np.float64]:
    """Force per unit relative velocity of the water flowing past particles in a packing, apart from buoyancy.

    The drag follows the correlation of Di Felice (1994), ``F_d = 1/2 C_D rho_w pi R^2 eps^2 |u - v| (u - v)
    eps^(-chi)``, with the single-sphere drag coefficient ``C_D = (0.63 + 4.8 / sqrt(Re))^2``, the particle
    Reynolds number ``Re = rho_w 2R eps |u - v| / mu`` and the crowding exponent
    ``chi = 3.7 - 0.65 exp(-(1.5 - log10 Re)^2 / 2)``. The flow that drags the particles also needs a pressure
    gradient, which pushes on their volume; for a uniform flow through a packing the two together amount to the
    drag divided by the porosity (Zhou et al., 2010).

    Args:
        speed: relative speed |u - v| of the water past each particle [m/s].
        radii: particle radii [m].
        porosity: local volume porosity around each particle.
        water: properties of the water.

    Returns:
        The force per unit relative velocity, finite also at zero speed (the Stokes limit) [N s/m].
    """
    diameter = 2.0 * radii
    rho, mu = water.density, water.viscosity
    # C_D |u - v| written so that it stays finite as the relative speed vanishes
    cd_speed = (0.63 * np.sqrt(speed) + 4.8 * np.sqrt(mu / (rho * diameter * porosity)))**2
    reynolds = np.maximum(rho * diameter * porosity * speed / mu, 1.0e-12)
    chi = 3.7 - 0.65 * np.exp(-0.5 * (1.5 - np.log10(reynolds))**2)
    return np.asarray(0.5 * rho * math.pi * radii**2 * porosity**(1.0 - chi) * cd_speed)


def drag_force(relative_velocity: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
               porosity: npt.NDArray[np.float64], water: Water) -> npt.NDArray[np.float64]:
    """Drag of the water on fully submerged spheres, from the correlation of Di Felice (1994).

    Args:
        relative_velocity: velocity of the water relative to each particle, u - v, shape (n, 2) [m/s].
        radii: particle radii [m].
        porosity: local volume porosity around each particle.
        water: properties of the water.

    Returns:
        The drag force on each particle, shape (n, 2) [N].
    """
    return np.asarray(porosity[:, None] * interaction_force(relative_velocity, radii, porosity, water))


def interaction_force(relative_velocity: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                      porosity: npt.NDArray[np.float64], water: Water) -> npt.NDArray[np.float64]:
    """Force of the water flowing past particles in a packing, apart from buoyancy: the drag over the porosity.

    Args:
        relative_velocity: velocity of the water relative to each particle, u - v, shape (n, 2) [m/s].
        radii: particle radii [m].
        porosity: local volume porosity around each particle.
        water: properties of the water.

    Returns:
        The force on each particle, shape (n, 2) [N].
    """
    speed = np.hypot(relative_velocity[:, 0], relative_velocity[:, 1])
    return np.asarray(interaction_coefficient(speed, radii, porosity, water)[:, None] * relative_velocity)


def permeability(radii: npt.NDArray[np.float64], porosity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Permeability of a packing, consistent with the drag law at low Reynolds number.

    Darcy's law with ``k = 4 / (3 * 5.76) R^2 eps^4.7 / (1 - eps)`` gives the same pressure drop through a fixed
    packing as ``interaction_force`` in the Stokes limit.

    Args:
        radii: representative (Sauter mean) particle radii [m].
        porosity: volume porosity, at most MAX_POROSITY.

    Returns:
        The permeability [m^2].
    """
    return np.asarray(PERMEABILITY_CONSTANT * radii**2 * porosity**4.7 / (1.0 - porosity))


class PorosityField:
    """Coarse grid on which the porosity, and the pore pressure, are averaged with cloud-in-cell weights.

    Each particle spreads its disc area over the four grid nodes around its centre, with bilinear weights. The area
    of the domain around each node is spread the same way, from a fine sampling of the domain, so that cells cut by
    walls are handled consistently. Values at a particle are interpolated with the same weights.
    """

    def __init__(self, geometry: Geometry, cell_size: float) -> None:
        """Build the grid over the domain.

        Args:
            geometry: domain.
            cell_size: side of the grid cells [m].
        """
        x_min, y_min, x_max, y_max = geometry.bounding_box
        self.cell_size = cell_size
        self.__origin = np.array([x_min - cell_size, y_min - cell_size])
        self.shape = (int(math.ceil((x_max - x_min) / cell_size)) + 3, int(math.ceil((y_max - y_min) / cell_size)) + 3)

        spacing = cell_size / POROSITY_SAMPLES_PER_CELL
        xs = np.arange(x_min + 0.5 * spacing, x_max, spacing)
        ys = np.arange(y_min + 0.5 * spacing, y_max, spacing)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        samples = samples[geometry.contains(samples)]
        nodes, weights = self.weights(samples)
        self.node_area = self.spread(np.full(samples.shape[0], spacing**2), nodes, weights)

    @property
    def node_positions(self) -> npt.NDArray[np.float64]:
        """Positions of the grid nodes, flattened, shape (nodes, 2) [m]."""
        i, j = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        return np.asarray(self.__origin + self.cell_size * np.column_stack((i.ravel(), j.ravel())))

    def weights(self, points: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return the four surrounding nodes of each point and their bilinear weights.

        Args:
            points: points, shape (n, 2) [m].

        Returns:
            The flat node indices, shape (n, 4), and the weights, shape (n, 4).
        """
        nx, ny = self.shape
        scaled = (points - self.__origin) / self.cell_size
        # points beyond the grid, e.g. escaping particles, are attached to its border
        scaled[:, 0] = np.clip(scaled[:, 0], 0.0, nx - 1.000001)
        scaled[:, 1] = np.clip(scaled[:, 1], 0.0, ny - 1.000001)
        corner = np.floor(scaled).astype(np.int64)
        fx, fy = (scaled - corner).T
        i, j = corner.T
        nodes = np.column_stack((i * ny + j, (i + 1) * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1))
        weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
        return nodes, weights

    def spread(self, amounts: npt.NDArray[np.float64], nodes: npt.NDArray[np.int64],
               weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Spread an amount carried by each point over its four surrounding nodes.

        Args:
            amounts: amount carried by each point, shape (n,) or (n, d).
            nodes: node indices of the points, shape (n, 4).
            weights: bilinear weights of the points, shape (n, 4).

        Returns:
            The total amount at each node, shape (nodes,) or (nodes, d).
        """
        size = self.shape[0] * self.shape[1]
        if amounts.ndim == 1:
            return np.asarray(np.bincount(nodes.ravel(), (weights * amounts[:, None]).ravel(), size), dtype=np.float64)
        return np.column_stack([self.spread(amounts[:, k], nodes, weights) for k in range(amounts.shape[1])])

    def gather(self, node_values: npt.NDArray[np.float64], node_weights: npt.NDArray[np.float64],
               nodes: npt.NDArray[np.int64], weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Interpolate node values to points, each node counting in proportion to a weight of its own.

        Args:
            node_values: value at each node, shape (nodes,) or (nodes, d).
            node_weights: weight of each node, e.g. its domain area; nodes with zero weight are ignored.
            nodes: node indices of the points, shape (n, 4).
            weights: bilinear weights of the points, shape (n, 4).

        Returns:
            The interpolated values, zero where all surrounding nodes have zero weight.
        """
        combined = weights * node_weights[nodes]
        total = combined.sum(axis=1)
        safe = np.where(total > 0.0, total, 1.0)
        if node_values.ndim == 1:
            return np.asarray(np.where(total > 0.0, np.sum(combined * node_values[nodes], axis=1) / safe, 0.0))
        result = np.einsum("nk,nkd->nd", combined, node_values[nodes]) / safe[:, None]
        return np.asarray(np.where(total[:, None] > 0.0, result, 0.0))

    def porosity_2d(self, positions: npt.NDArray[np.float64],
                    radii: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Area porosity of the disc layer around each particle.

        Args:
            positions: particle centres, shape (n, 2) [m].
            radii: particle radii [m].

        Returns:
            The local area porosity at each particle.
        """
        nodes, weights = self.weights(positions)
        solid = self.spread(math.pi * radii**2, nodes, weights)
        # interpolating solid and domain area separately keeps nodes with little domain area from dominating
        local_solid = np.sum(weights * solid[nodes], axis=1)
        local_area = np.sum(weights * self.node_area[nodes], axis=1)
        safe = local_area > 0.0
        porosity = np.ones(radii.shape)
        porosity[safe] = 1.0 - local_solid[safe] / local_area[safe]
        return np.asarray(np.clip(porosity, 0.0, 1.0))


class PorePressure:
    """Excess pore pressure on the nodes of the porosity grid.

    Each node is a control volume, the part of its cell inside the domain, and the pressure follows the Biot
    storage equation ``(eps / K_w) dp/dt = -div(c) + div((k / mu) grad p)``: the water stored by compressing it
    balances the water that the skeleton carries in or out, c, and the Darcy flow driven by the excess pressure.
    Walls are impermeable; nodes above the free surface or next to an open edge are drained (p = 0). The diffusion
    is integrated with backward Euler, so it is stable whatever the permeability.

    Attributes:
        pressure: excess pore pressure at each node, zero outside the domain and at drained nodes [Pa].
        iterations: number of conjugate gradient iterations of the last step.
    """

    def __init__(self, field: PorosityField, geometry: Geometry) -> None:
        """Set up the control volumes, the faces between them and the nodes next to open edges.

        Args:
            field: grid on which the pressure lives.
            geometry: domain.
        """
        self.field = field
        h = field.cell_size
        nx, ny = field.shape
        area = field.node_area
        self.__active = area > ACTIVE_NODE_FRACTION * h**2
        index = np.arange(nx * ny).reshape(nx, ny)
        firsts, seconds, axes = [], [], []
        for axis, (first, second) in enumerate(((index[:-1, :], index[1:, :]), (index[:, :-1], index[:, 1:]))):
            first, second = first.ravel(), second.ravel()
            both = self.__active[first] & self.__active[second]
            firsts.append(first[both])
            seconds.append(second[both])
            axes.append(np.full(int(both.sum()), axis))
        self.__first = np.concatenate(firsts)
        self.__second = np.concatenate(seconds)
        self.__axis = np.concatenate(axes)
        # a face between two cells cut by a wall is only partly open to flow
        self.__open = np.minimum(np.minimum(area[self.__first], area[self.__second]) / h**2, 1.0)

        open_edges = [k for k, condition in enumerate(geometry.boundary_conditions) if isinstance(condition, Open)]
        self.__open_edge_nodes = np.zeros(nx * ny, dtype=bool)
        if open_edges:
            _, offset = project_onto_segments(field.node_positions, geometry.edge_starts[open_edges],
                                              geometry.edge_ends[open_edges])
            distance = np.hypot(offset[..., 0], offset[..., 1]).min(axis=1)
            self.__open_edge_nodes = self.__active & (distance <= h)

        self.pressure: npt.NDArray[np.float64] = np.zeros(nx * ny)
        self.iterations = 0

    def advance(self, dt: float, porosity: npt.NDArray[np.float64], mobility: npt.NDArray[np.float64],
                carried: npt.NDArray[np.float64], level: float, bulk_modulus: float) -> None:
        """Advance the pressure by one backward-Euler step.

        The symmetric positive definite system ``(S / dt + L) p = S / dt p_old - D`` is solved by conjugate gradients
        with a diagonal preconditioner, started from the previous pressure. The operator is applied face by face, so
        the solver needs no matrix and no linear algebra library.

        Args:
            dt: time step [s].
            porosity: volume porosity at each node.
            mobility: permeability over viscosity at each node [m^2/(Pa s)].
            carried: flux of solid and water that the skeleton carries at each node, shape (nodes, 2) [m/s].
            level: height of the free surface [m].
            bulk_modulus: bulk modulus of the water [Pa].

        Raises:
            RuntimeError: if the conjugate gradient solver does not converge.
        """
        size = self.pressure.size
        h = self.field.cell_size
        a, b = self.__first, self.__second
        free = self.__active & ~self.__open_edge_nodes & (self.field.node_positions[:, 1] < level)
        storage = np.where(free, self.field.node_area * porosity / (bulk_modulus * dt), 0.0)
        # harmonic mean, as for two resistances in series
        conductance = 2.0 * mobility[a] * mobility[b] / (mobility[a] + mobility[b]) * self.__open
        # a drained neighbour adds to the diagonal only, since its pressure is zero
        diagonal = storage + np.bincount(a, conductance, size) + np.bincount(b, conductance, size)
        linked = free[a] & free[b]
        link_a, link_b, link_g = a[linked], b[linked], conductance[linked]

        def apply(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            y = diagonal * x - np.bincount(link_a, link_g * x[link_b], size) - np.bincount(
                link_b, link_g * x[link_a], size)
            return np.where(free, y, 0.0)

        # net outflow of the carried flux from each control volume
        face_flux = 0.5 * (carried[a, self.__axis] + carried[b, self.__axis]) * self.__open * h
        outflow = np.bincount(a, face_flux, size) - np.bincount(b, face_flux, size)
        rhs = np.where(free, storage * self.pressure - outflow, 0.0)

        x = np.where(free, self.pressure, 0.0)
        inverse_diagonal = np.where(free & (diagonal > 0.0), 1.0 / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
        residual = rhs - apply(x)
        preconditioned = inverse_diagonal * residual
        direction = preconditioned.copy()
        product = float(residual @ preconditioned)
        target = PRESSURE_SOLVER_TOLERANCE * float(np.sqrt(rhs @ rhs) + np.sqrt(np.sum((storage * x)**2)))
        budget = PRESSURE_SOLVER_ITERATIONS * max(int(free.sum()), 1)
        self.iterations = 0
        while float(np.sqrt(residual @ residual)) > target:
            if self.iterations >= budget:
                raise RuntimeError(f"The pore pressure solver did not converge in {budget} iterations.")
            image = apply(direction)
            step = product / float(direction @ image)
            x += step * direction
            residual -= step * image
            preconditioned = inverse_diagonal * residual
            previous, product = product, float(residual @ preconditioned)
            direction = preconditioned + (product / previous) * direction
            self.iterations += 1
        self.pressure = x

    def gradient(self) -> npt.NDArray[np.float64]:
        """Pressure gradient at each node, averaged over its faces that are open to flow, shape (nodes, 2) [Pa/m]."""
        size = self.pressure.size
        face_gradient = (self.pressure[self.__second] - self.pressure[self.__first]) / self.field.cell_size
        gradient = np.zeros((size, 2))
        for axis in (0, 1):
            along = self.__axis == axis
            first, second = self.__first[along], self.__second[along]
            total = np.bincount(first, face_gradient[along], size) + np.bincount(second, face_gradient[along], size)
            count = np.bincount(first, minlength=size) + np.bincount(second, minlength=size)
            gradient[:, axis] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        return gradient


@dataclass(frozen=True, eq=False)
class FluidForces:
    """Forces of the water on the particles at one evaluation.

    Attributes:
        force: buoyancy and pore pressure force on each particle, shape (n, 2) [N].
        drag_coefficient: drag per unit relative velocity of each particle, so that the drag is
            ``drag_coefficient * (water_velocity - v)`` [N s/m].
        water_velocity: velocity of the water around each particle, shape (n, 2) [m/s].
        porosity: local volume porosity at each particle.
    """

    force: npt.NDArray[np.float64]
    drag_coefficient: npt.NDArray[np.float64]
    water_velocity: npt.NDArray[np.float64]
    porosity: npt.NDArray[np.float64]


class PoreWater:
    """Runtime evaluation of the forces of the water on the particles."""

    def __init__(self, water: Water, geometry: Geometry, max_radius: float) -> None:
        """Set up the porosity grid and, when coupled, the pore pressure.

        Args:
            water: properties of the water.
            geometry: domain.
            max_radius: radius of the largest particle [m].
        """
        self.water = water
        self.max_radius = max_radius
        self.field = PorosityField(geometry, water.cell_size or 4.0 * max_radius)
        self.pore_pressure = PorePressure(self.field, geometry) if water.coupled else None

    def forces(self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64],
               radii: npt.NDArray[np.float64], time: float, gravity: float, dt: float) -> FluidForces:
        """Buoyancy, pore pressure force and drag on every particle.

        Args:
            positions: particle centres, shape (n, 2) [m].
            velocities: particle velocities, shape (n, 2) [m/s].
            radii: particle radii [m].
            time: simulation time [s].
            gravity: magnitude of the gravitational acceleration [m/s^2].
            dt: time step over which the pore pressure is advanced; zero re-evaluates the forces of the current
                pressure [s].

        Returns:
            The forces, with the drag as a coefficient and a water velocity so that it can be integrated
            implicitly.
        """
        water = self.water
        field = self.field
        nodes, weights = field.weights(positions)
        solid_area = field.spread(math.pi * radii**2, nodes, weights)
        local_solid = np.sum(weights * solid_area[nodes], axis=1)
        local_area = np.sum(weights * field.node_area[nodes], axis=1)
        porosity_2d = np.where(local_area > 0.0, 1.0 - local_solid / np.where(local_area > 0.0, local_area, 1.0), 1.0)
        porosity = np.maximum(porosity_3d(np.clip(porosity_2d, 0.0, 1.0)), MIN_POROSITY)

        level = evaluate_at(water.level, time)
        fraction = submerged_fraction(positions[:, 1], radii, level)
        volume = 4.0 / 3.0 * math.pi * radii**3
        force = np.zeros_like(positions)
        force[:, 1] = water.density * gravity * volume * fraction

        # velocity of the water around each particle, apart from the flow driven by the excess pressure
        water_velocity = np.zeros_like(positions)
        water_velocity[:, 1] = evaluate_at(water.seepage_velocity, time) / porosity
        if self.pore_pressure is not None:
            carried_velocity, gradient = self.__couple(velocities, radii, volume, nodes, weights, solid_area, level, dt)
            water_velocity += carried_velocity
            # the excess pressure gradient acts on the volume of mixture that each particle represents
            mixture = volume / (1.0 - np.minimum(porosity, MAX_POROSITY))
            force -= (fraction * mixture)[:, None] * gradient

        relative = water_velocity - velocities
        coefficient = interaction_coefficient(np.hypot(relative[:, 0], relative[:, 1]), radii, porosity, water)
        # a particle cut by the free surface is dragged in proportion to its submerged volume
        return FluidForces(force, fraction * coefficient, water_velocity, porosity)

    def excess_pressure(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Excess pore pressure at given points, zero for drained water.

        Args:
            points: points, shape (n, 2) [m].

        Returns:
            The excess pore pressure at each point [Pa].
        """
        if self.pore_pressure is None:
            return np.zeros(points.shape[0])
        nodes, weights = self.field.weights(points)
        return self.field.gather(self.pore_pressure.pressure, self.field.node_area, nodes, weights)

    def time_step_limit(self, radius: float, density: float) -> float:
        """Largest time step that keeps the explicit pore pressure coupling stable.

        Two time scales of the coupling must be resolved: the time an undrained pressure wave, of speed
        sqrt(K_w / (eps (1 - eps) rho_s)), needs to cross a grid cell, and the time the drained pressure needs to stop
        a collective motion of the skeleton, (1 - eps) rho_s k / mu, for the smallest particles in a dense packing.
        The drag is integrated implicitly and sets no limit, and neither does drained water.

        Args:
            radius: radius of the smallest particle [m].
            density: particle density [kg/m^3].

        Returns:
            The largest time step [s], infinite for drained water.
        """
        if not self.water.coupled:
            return math.inf
        wave_speed = math.sqrt(self.water.bulk_modulus / (MIN_POROSITY * (1.0 - MIN_POROSITY) * density))
        wave = COUPLING_TIME_STEP_FRACTION * self.field.cell_size / wave_speed
        dense = np.array([DENSE_POROSITY])
        relaxation = (1.0 - DENSE_POROSITY) * density * float(permeability(np.array([radius]),
                                                                           dense)[0]) / (self.water.viscosity)
        return min(wave, relaxation)

    def __couple(self, velocities: npt.NDArray[np.float64], radii: npt.NDArray[np.float64],
                 volume: npt.NDArray[np.float64], nodes: npt.NDArray[np.int64], weights: npt.NDArray[np.float64],
                 solid_area: npt.NDArray[np.float64], level: float,
                 dt: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Advance the excess pore pressure and return what it does at the particles.

        Args:
            velocities: particle velocities, shape (n, 2) [m/s].
            radii: particle radii [m].
            volume: particle volumes [m^3].
            nodes: grid nodes around each particle, shape (n, 4).
            weights: bilinear weights of those nodes, shape (n, 4).
            solid_area: disc area spread on each node [m^2].
            level: height of the free surface [m].
            dt: time step over which the pressure is advanced; zero leaves it unchanged [s].

        Returns:
            The velocity of the water carried along by the skeleton at each particle [m/s], and the excess pressure
            gradient at each particle [Pa/m].

        Raises:
            RuntimeError: if the water is not coupled.
        """
        pore_pressure = self.pore_pressure
        if pore_pressure is None:
            raise RuntimeError("The pore pressure is only solved for coupled water.")
        water = self.water
        field = self.field
        area = field.node_area
        node_porosity_2d = np.where(area > 0.0, 1.0 - solid_area / np.where(area > 0.0, area, 1.0), 1.0)
        node_porosity = np.clip(porosity_3d(np.clip(node_porosity_2d, 0.0, 1.0)), MIN_POROSITY, MAX_POROSITY)
        solid_fraction = 1.0 - node_porosity

        # mean velocity of the skeleton at each node, weighted by solid volume
        node_volume = field.spread(volume, nodes, weights)
        momentum = field.spread(volume[:, None] * velocities, nodes, weights)
        occupied = node_volume > 0.0
        skeleton_velocity = np.zeros_like(momentum)
        skeleton_velocity[occupied] = momentum[occupied] / node_volume[occupied, None]
        # share of the pore water that moves with the skeleton: all of it in a dense packing, none in a suspension
        carried_share = np.clip((solid_fraction - CARRIED_SOLID_FRACTION_DILUTE) /
                                (CARRIED_SOLID_FRACTION_DENSE - CARRIED_SOLID_FRACTION_DILUTE), 0.0, 1.0)

        if dt > 0.0:
            # the Sauter mean radius sets the permeability; empty nodes get that of the largest particles
            cubes = field.spread(radii**3, nodes, weights)
            squares = field.spread(radii**2, nodes, weights)
            sauter = np.full(area.shape, self.max_radius)
            sauter[occupied] = cubes[occupied] / squares[occupied]
            mobility = np.asarray(permeability(sauter, node_porosity) / water.viscosity, dtype=np.float64)
            carried = (solid_fraction + node_porosity * carried_share)[:, None] * skeleton_velocity
            pore_pressure.advance(dt, node_porosity, mobility, carried, level, water.bulk_modulus)

        share = field.gather(carried_share, area, nodes, weights)
        velocity = field.gather(skeleton_velocity, node_volume, nodes, weights)
        gradient = field.gather(pore_pressure.gradient(), area, nodes, weights)
        return share[:, None] * velocity, gradient
