"""Consolidation of a submerged granular column under a sudden load, compared with Terzaghi's theory.

A column is rained into water in a box with an impermeable floor and smooth side walls. The water is then coupled to
the particles and made a thousand times more viscous than water, so that these 8 to 12 mm grains drain like a soil a
thousand times less permeable. A load is applied suddenly through a platen on top, which lets the water through like
a porous stone. At first the pore water carries the load; it then drains through the top of the column, the load
passes to the skeleton and the column settles. The excess pore pressure at the base and along the column is compared
with Terzaghi's one-dimensional consolidation theory. Run it with::

    python examples/consolidation.py

and the figures are written to ``output/``.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from DEMcookies import (DEMSimulation, Geometry, Heaviside, LoadedPlaten, MaterialProperties, RainSettings, Wall, Water)
from DEMcookies.water import permeability

OUTPUT_DIR = "output"
LOAD = 20.0
LEVEL = 0.38
VISCOSITY = 1.0
DURATION = 3.0
SAMPLE_INTERVAL = 0.02
PROFILE_TIMES = (0.15, 0.3, 0.6, 1.2, 2.4)
# the platen sets the water ringing for a moment after the load is applied; the fits start once it has calmed down
FIT_START = 0.15

DRAINED = Water(level=LEVEL)
COUPLED = Water(level=LEVEL, coupled=True, viscosity=VISCOSITY)
RAIN = RainSettings(batch_size=20, steps_between=300, insertion_band=(0.2, 0.21), record_every=100, seed=5)


@dataclass
class Consolidation:
    """Response of the column to the load.

    Attributes:
        height: height of the column when the load was applied, its drainage length [m].
        time: time since the load was applied [s].
        settlement: settlement of the platen [m].
        base_pressure: excess pore pressure at the base of the column [Pa].
        floor_share: increase of the force on the floor, as a fraction of the load.
        profiles: excess pore pressure along the column at a few times, as heights [m] and pressures [Pa].
    """

    height: float
    time: List[float] = field(default_factory=list)
    settlement: List[float] = field(default_factory=list)
    base_pressure: List[float] = field(default_factory=list)
    floor_share: List[float] = field(default_factory=list)
    profiles: Dict[float, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = field(default_factory=dict)


def build_simulation() -> DEMSimulation:
    """Define the box with a loaded platen on top, filled with drained water for the deposition.

    Returns:
        The simulation, with an empty box.
    """
    geometry = Geometry.rectangle(width=0.2,
                                  height=0.4,
                                  bottom=Wall(),
                                  right=Wall(friction=0.0),
                                  top=LoadedPlaten(load=Heaviside(LOAD, rise_time=0.01), mass=0.5),
                                  left=Wall(friction=0.0))
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5)
    return DEMSimulation(geometry, material, radius_range=(0.004, 0.006), water=DRAINED)


def terzaghi(depth: npt.NDArray[np.float64], height: float, time_factor: float) -> npt.NDArray[np.float64]:
    """Excess pore pressure of Terzaghi's theory, as a fraction of its initial value.

    Args:
        depth: depth below the drained top of the layer [m].
        height: drainage length, here the height of the column [m].
        time_factor: time factor T = c_v t / H^2.

    Returns:
        The pressure ratio at each depth.
    """
    total = np.zeros_like(depth)
    for m in range(100):
        wave = (2 * m + 1) * math.pi / 2.0
        total += 2.0 / wave * np.sin(wave * depth / height) * math.exp(-wave**2 * time_factor)
    return total


def average_consolidation(time_factor: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Average degree of consolidation of Terzaghi's theory, the fraction of the final settlement reached.

    Args:
        time_factor: time factors T = c_v t / H^2.

    Returns:
        The degree of consolidation at each time factor.
    """
    factors = np.asarray(time_factor, dtype=np.float64)
    total = np.zeros_like(factors)
    for m in range(100):
        wave = (2 * m + 1) * math.pi / 2.0
        total += 2.0 / wave**2 * np.exp(-wave**2 * factors)
    return np.asarray(1.0 - total)


def fit_base_pressure(result: Consolidation) -> Tuple[float, float]:
    """Fit Terzaghi's base pressure to the model, for the initial pressure and the coefficient of consolidation.

    For a given coefficient the best initial pressure follows from linear least squares, so only the coefficient is
    searched for, by golden section on its logarithm.

    Args:
        result: response of the column.

    Returns:
        The coefficient of consolidation [m^2/s] and the initial excess pressure [Pa].
    """
    time = np.array(result.time)
    use = time >= FIT_START
    pressure = np.array(result.base_pressure)[use]
    height = result.height

    def shape(log_coefficient: float) -> npt.NDArray[np.float64]:
        factors = 10.0**log_coefficient * time[use] / height**2
        return np.array([terzaghi(np.array([height]), height, factor)[0] for factor in factors])

    def misfit(log_coefficient: float) -> float:
        model = shape(log_coefficient)
        scale = float(model @ pressure / (model @ model))
        return float(np.sum((scale * model - pressure)**2))

    low, high = -6.0, 2.0
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):
        left, right = high - golden * (high - low), low + golden * (high - low)
        low, high = (low, right) if misfit(left) < misfit(right) else (left, high)
    best = 0.5 * (low + high)
    model = shape(best)
    return float(10.0**best), float(model @ pressure / (model @ model))


def fit_settlement(result: Consolidation, coefficient: float) -> Tuple[float, float]:
    """Fit the settlement with Terzaghi's average degree of consolidation, s = s_0 + s_c U(T).

    Args:
        result: response of the column.
        coefficient: coefficient of consolidation [m^2/s].

    Returns:
        The immediate settlement s_0 and the consolidation settlement s_c [m].
    """
    time = np.array(result.time)
    use = time >= FIT_START
    degree = average_consolidation(coefficient * time[use] / result.height**2)
    settlement = np.array(result.settlement)[use]
    # straight-line least squares of the settlement against the degree of consolidation
    consolidation = float(
        np.sum((degree - degree.mean()) * (settlement - settlement.mean())) / np.sum((degree - degree.mean())**2))
    return float(settlement.mean() - consolidation * degree.mean()), consolidation


def run_loading(sim: DEMSimulation) -> Consolidation:
    """Load the settled column and record its consolidation.

    Args:
        sim: simulation with a settled packing in coupled water.

    Returns:
        The response of the column.
    """
    floor_before = -sim.edge_forces[0, 1]
    sim.compress(0.0)
    result = Consolidation(height=float(sim.platens[0].starts[0, 1]))
    centre = 0.5 * (sim.geometry.bounding_box[0] + sim.geometry.bounding_box[2])
    pending = list(PROFILE_TIMES)
    while sim.load_time < DURATION:
        sim.compress(SAMPLE_INTERVAL, record_every=100)
        result.time.append(sim.load_time)
        result.settlement.append(sim.platens[0].displacement)
        result.base_pressure.append(float(sim.excess_pore_pressure(np.array([[centre, 0.0]]))[0]))
        result.floor_share.append((-sim.edge_forces[0, 1] - floor_before) / LOAD)
        if pending and sim.load_time >= pending[0] - 1e-9:
            heights = np.asarray(np.linspace(0.0, result.height, 41), dtype=np.float64)
            points = np.column_stack((np.full(heights.size, centre), heights))
            result.profiles[pending.pop(0)] = (heights, sim.excess_pore_pressure(points))
    return result


def predicted_consolidation_coefficient(sim: DEMSimulation, result: Consolidation, pressure: float,
                                        consolidation: float) -> Tuple[float, float]:
    """Predict the coefficient of consolidation from the permeability of the model and the stiffness of the column.

    The constrained modulus of the skeleton follows from the consolidation settlement, M = p0 H / s_c.

    Args:
        sim: simulation after loading.
        result: response of the column.
        pressure: initial excess pore pressure p0, i.e. the stress the skeleton takes over [Pa].
        consolidation: consolidation settlement s_c [m].

    Returns:
        The predicted coefficient of consolidation [m^2/s] and the constrained modulus [Pa].
    """
    modulus = pressure * result.height / consolidation
    porosity = float(np.mean(sim.porosity))
    sauter = float(np.sum(sim.radius**3) / np.sum(sim.radius**2))
    mobility = float(permeability(np.array([sauter]), np.array([porosity]))[0]) / VISCOSITY
    # the compressibility of the water adds to that of the skeleton
    return mobility / (1.0 / modulus + porosity / COUPLED.bulk_modulus), modulus


def plot_consolidation(result: Consolidation, coefficient: float, pressure: float, settlement: Tuple[float, float],
                       path: str) -> None:
    """Plot the base pressure, the floor share, the settlement and the pressure profiles against Terzaghi's theory.

    Args:
        result: response of the column.
        coefficient: coefficient of consolidation of Terzaghi's curves [m^2/s].
        pressure: initial excess pore pressure of Terzaghi's curves [Pa].
        settlement: immediate and consolidation settlement of Terzaghi's settlement curve [m].
        path: output image file.
    """
    time = np.array(result.time)
    height = result.height
    curve_time = np.linspace(0.0, time[-1], 200)
    factors = coefficient * curve_time / height**2
    base = [terzaghi(np.array([height]), height, factor)[0] for factor in factors]

    fig, (ax_time, ax_settlement, ax_profile) = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax_time.plot(time, np.array(result.base_pressure) / pressure, "o", markersize=3, label="base pressure, model")
    ax_time.plot(curve_time, base, "-", label="base pressure, Terzaghi")
    ax_time.plot(time, result.floor_share, "s", markersize=3, label="load on the floor, model")
    ax_time.plot(curve_time, 1.0 - np.array(base), "--", label="load on the floor, Terzaghi")
    ax_time.set_xlabel("time since loading [s]")
    ax_time.set_ylabel("fraction of the load")
    ax_time.legend()
    ax_time.grid(alpha=0.3)

    immediate, consolidation = settlement
    ax_settlement.plot(time, np.array(result.settlement) * 1e3, "o", markersize=3, label="model")
    ax_settlement.plot(curve_time, (immediate + consolidation * average_consolidation(factors)) * 1e3,
                       "-",
                       label="Terzaghi")
    ax_settlement.set_xlabel("time since loading [s]")
    ax_settlement.set_ylabel("settlement [mm]")
    ax_settlement.legend()
    ax_settlement.grid(alpha=0.3)

    depth = np.linspace(0.0, height, 100)
    for count, (moment, (heights, values)) in enumerate(sorted(result.profiles.items())):
        colour = f"C{count}"
        ax_profile.plot(values / pressure, heights / height, "o", markersize=3, color=colour)
        theory = terzaghi(height - depth, height, coefficient * moment / height**2)
        ax_profile.plot(theory, depth / height, "-", color=colour, label=f"t = {moment:.2f} s")
    ax_profile.set_xlabel("excess pore pressure / initial")
    ax_profile.set_ylabel("height / column height")
    ax_profile.legend()
    ax_profile.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Fill the column, load it with coupled viscous water and compare with Terzaghi's theory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_simulation()
    sim.rain_until_full(RAIN)
    sim.rest(0.2, record_every=100)
    sim.set_water(COUPLED)
    sim.rest(0.05, record_every=100)
    mixture = float(np.sum(4.0 / 3.0 * math.pi * sim.radius**3 / (1.0 - sim.porosity)))
    print(f"{sim.n} particles, column height = {sim.bed_top():.4f} m, time step = {sim.dt:.3e} s")

    result = run_loading(sim)
    coefficient, pressure = fit_base_pressure(result)
    immediate, consolidation = fit_settlement(result, coefficient)
    predicted, modulus = predicted_consolidation_coefficient(sim, result, pressure, consolidation)
    print(f"initial excess pressure: fitted = {pressure:.0f} Pa, load over mixture area = "
          f"{LOAD * result.height / mixture:.0f} Pa")
    print(f"settlement: immediate = {immediate * 1e3:.2f} mm, consolidation = {consolidation * 1e3:.2f} mm; "
          f"constrained modulus = {modulus:.3g} Pa")
    print(f"coefficient of consolidation: fitted = {coefficient:.4f} m^2/s, predicted = {predicted:.4f} m^2/s")

    plot_consolidation(result, coefficient, pressure, (immediate, consolidation),
                       os.path.join(OUTPUT_DIR, "consolidation.png"))
    print(f"Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
