"""Upward seepage through a submerged granular column: from the quick condition to fluidization.

A column of glass-like spheres is rained into a water-filled permeameter, a box whose side walls are smooth so that
wall friction does not carry part of the bed. An upward flow is then imposed and ramped from zero to twice the
minimum fluidization velocity, at which the force of the water on the fixed bed equals its buoyant weight. As the
flow grows, the force the bed exerts on the floor, its effective weight, falls to zero; beyond that point the bed
lifts, expands and fluidizes. Run it with::

    python examples/upward_seepage.py

and the animation and the fluidization curve are written to ``output/``.
"""

import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from DEMcookies import DEMSimulation, Geometry, MaterialProperties, Open, RainSettings, TimeHistory, Wall, Water
from DEMcookies.plotting import render_animation
from DEMcookies.water import interaction_force

OUTPUT_DIR = "output"
LEVEL = 0.45
RAMP_DURATION = 4.0
RAMP_TOP = 2.0
SAMPLE_INTERVAL = 0.05

WATER = Water(level=LEVEL)
RAIN = RainSettings(batch_size=20, steps_between=300, insertion_band=(0.24, 0.25), record_every=100, seed=5)


def build_simulation() -> DEMSimulation:
    """Define the water-filled permeameter.

    Returns:
        The simulation, with an empty column.
    """
    geometry = Geometry.rectangle(width=0.2,
                                  height=0.5,
                                  bottom=Wall(),
                                  right=Wall(friction=0.0),
                                  top=Open(),
                                  left=Wall(friction=0.0))
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5)
    return DEMSimulation(geometry, material, radius_range=(0.004, 0.006), water=WATER)


def buoyant_weight(sim: DEMSimulation) -> float:
    """Return the weight of the particles minus the buoyancy of the water on them [N]."""
    volume = 4.0 / 3.0 * math.pi * sim.radius**3
    return float(np.sum((sim.material.density - WATER.density) * volume) * sim.settings.gravity)


def fixed_bed_force(sim: DEMSimulation, superficial_velocity: float) -> float:
    """Return the total upward force of a seepage flow on the particles, if they stayed where they are [N].

    Args:
        sim: simulation with water.
        superficial_velocity: superficial velocity of the upward flow [m/s].

    Returns:
        The total force.
    """
    relative = np.zeros((sim.n, 2))
    relative[:, 1] = superficial_velocity / sim.porosity
    return float(interaction_force(relative, sim.radius, sim.porosity, WATER)[:, 1].sum())


def minimum_fluidization_velocity(sim: DEMSimulation) -> float:
    """Return the superficial velocity at which the force on the fixed bed equals its buoyant weight [m/s]."""
    target = buoyant_weight(sim)
    low, high = 0.0, 10.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if fixed_bed_force(sim, middle) < target else (low, middle)
    return 0.5 * (low + high)


def bed_top(sim: DEMSimulation, count: int = 10) -> float:
    """Return the mean top of the highest particles, a measure of the bed height that ignores single grains [m]."""
    return float(np.sort(sim.pos[:, 1] + sim.radius)[-count:].mean())


def run_ramp(sim: DEMSimulation, critical: float) -> List[Tuple[float, float, float, float]]:
    """Ramp the upward flow to RAMP_TOP times the critical velocity and sample the response.

    Args:
        sim: simulation with a settled, submerged packing.
        critical: minimum fluidization velocity [m/s].

    Returns:
        Rows of superficial velocity [m/s], base force over buoyant weight, bed top [m] and mean porosity.
    """
    start = sim.time
    top = RAMP_TOP * critical
    # the flow is a function of the simulation time, so the ramp starts now
    sim.set_water(Water(level=LEVEL, seepage_velocity=TimeHistory([start, start + RAMP_DURATION], [0.0, top])))
    weight = buoyant_weight(sim)
    rows = []
    while sim.time < start + RAMP_DURATION:
        sim.rest(SAMPLE_INTERVAL, record_every=100, phase="seepage")
        velocity = top * min(1.0, (sim.time - start) / RAMP_DURATION)
        rows.append((velocity, -sim.edge_forces[0, 1] / weight, bed_top(sim), float(np.mean(sim.porosity))))
    return rows


def plot_fluidization(rows: List[Tuple[float, float, float, float]], prediction: List[Tuple[float, float]],
                      critical: float, path: str) -> None:
    """Plot the base force and the bed height against the seepage velocity.

    Args:
        rows: samples returned by ``run_ramp``.
        prediction: fixed-bed prediction of the base force, as (velocity, base force over buoyant weight).
        critical: minimum fluidization velocity [m/s].
        path: output image file.
    """
    velocity, base, top, _ = np.array(rows).T
    fig, (ax_force, ax_height) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax_force.plot(velocity / critical, base, "o", markersize=3, label="simulation")
    predicted = np.array(prediction)
    ax_force.plot(predicted[:, 0] / critical, predicted[:, 1], "-", label="fixed bed, $1 - F_f/W'$")
    ax_force.set_xlabel("seepage velocity $U/U_{mf}$")
    ax_force.set_ylabel("base force / buoyant weight")
    ax_force.legend()
    ax_force.grid(alpha=0.3)
    ax_height.plot(velocity / critical, top * 1e3, "o", markersize=3)
    ax_height.set_xlabel("seepage velocity $U/U_{mf}$")
    ax_height.set_ylabel("bed top [mm]")
    ax_height.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Fill the column under water, ramp the upward flow and write the results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_simulation()
    n = sim.rain_until_full(RAIN)
    sim.rest(0.2, record_every=100)
    critical = minimum_fluidization_velocity(sim)
    print(f"{n} particles, bed top = {bed_top(sim):.4f} m, mean porosity = {np.mean(sim.porosity):.3f}")
    print(f"base force = {-sim.edge_forces[0, 1]:.3f} N, buoyant weight = {buoyant_weight(sim):.3f} N")
    print(f"minimum fluidization velocity of the fixed bed = {critical:.4f} m/s")

    weight = buoyant_weight(sim)
    prediction = [(u, max(0.0, 1.0 - fixed_bed_force(sim, u) / weight))
                  for u in np.linspace(0.0, RAMP_TOP * critical, 81)]
    rows = run_ramp(sim, critical)
    for velocity, base, top, porosity in rows[9::10]:
        print(f"U = {velocity:.4f} m/s ({velocity / critical:.2f} U_mf): base force / W' = {base:.3f}, "
              f"bed top = {top:.4f} m, mean porosity = {porosity:.3f}")

    render_animation(sim, os.path.join(OUTPUT_DIR, "seepage.gif"))
    plot_fluidization(rows, prediction, critical, os.path.join(OUTPUT_DIR, "seepage.png"))
    print(f"Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
