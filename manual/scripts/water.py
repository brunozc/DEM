"""Pore water results of the manual: terminal settling velocities and fluidization by upward seepage.

Run from the repository root with ``python manual/scripts/water.py``. The figures are written to
``manual/figures``.
"""

import math

import numpy as np

from DEMcookies import DEMSimulation, Geometry, MaterialProperties, Particle, Water
from DEMcookies.plotting import render_frame
from DEMcookies.water import interaction_force

from common import FIGURES, ROOT, load_script

DENSITY = 2500.0
WATER = Water(level=2.0, cell_size=0.4)


def settle(radius: float) -> None:
    """Drop one particle in still water, and compare its final speed with the drag balance.

    Args:
        radius: particle radius [m].
    """
    material = MaterialProperties(density=DENSITY,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5)
    sim = DEMSimulation(Geometry.rectangle(0.4, 2.0), material, particles=[Particle(0.2, 1.9, radius)], water=WATER)
    buoyant = (DENSITY - WATER.density) * 4.0 / 3.0 * math.pi * radius**3 * sim.settings.gravity
    reduced_gravity = sim.settings.gravity * (1.0 - WATER.density / DENSITY)
    stokes = 2.0 * (DENSITY - WATER.density) * sim.settings.gravity * radius**2 / (9.0 * WATER.viscosity)

    def drag(speed: float) -> float:
        relative = np.array([[0.0, speed]])
        return float(interaction_force(relative, np.array([radius]), sim.porosity[:1], WATER)[0, 1])

    sim.compute_forces()
    low, high = 0.0, 10.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if drag(middle) < buoyant else (low, middle)
    expected = 0.5 * (low + high)
    # a few acceleration times, v_t / g', bring the particle to within a fraction of a per cent of v_t
    sim.rest(8.0 * expected / reduced_gravity, record_every=10**9)
    simulated = -sim.vel[0, 1]
    reynolds = WATER.density * 2.0 * radius * simulated / WATER.viscosity
    print(f"{radius * 1e3:6.2f} & {simulated:.4e} & {expected:.4e} & {100 * (simulated - expected) / expected:+.3f} & "
          f"{reynolds:.3g} & {stokes:.4e} \\\\")


def ergun_velocity(sim: DEMSimulation) -> float:
    """Minimum fluidization velocity from Ergun's equation, for the mean porosity and Sauter diameter of the bed.

    Args:
        sim: simulation with a settled, submerged packing.

    Returns:
        The superficial velocity at which Ergun's pressure drop equals the buoyant weight of the bed [m/s].
    """
    porosity = float(np.mean(sim.porosity))
    diameter = 2.0 * float(np.sum(sim.radius**3) / np.sum(sim.radius**2))
    weight = (1.0 - porosity) * (DENSITY - WATER.density) * sim.settings.gravity
    viscous = 150.0 * WATER.viscosity * (1.0 - porosity)**2 / (porosity**3 * diameter**2)
    inertial = 1.75 * WATER.density * (1.0 - porosity) / (porosity**3 * diameter)
    return (-viscous + math.sqrt(viscous**2 + 4.0 * inertial * weight)) / (2.0 * inertial)


def main() -> None:
    """Print the terminal velocity table, then run the seepage example and write its figures."""
    print("radius mm & simulated m/s & drag balance m/s & difference % & Re & Stokes m/s")
    for radius in (0.05e-3, 0.25e-3, 0.5e-3, 1.0e-3, 2.5e-3, 5.0e-3):
        settle(radius)

    example = load_script(ROOT / "examples" / "upward_seepage.py")
    sim = example.build_simulation()
    sim.rain_until_full(example.RAIN)
    sim.rest(0.2, record_every=100)
    weight = example.buoyant_weight(sim)
    base = []
    for _ in range(10):
        sim.rest(0.01, record_every=10**9)
        base.append(-sim.edge_forces[0, 1])
    critical = example.minimum_fluidization_velocity(sim)
    print(f"N = {sim.n}, bed top = {example.bed_top(sim):.4f} m, mean porosity = {np.mean(sim.porosity):.3f}, "
          f"time = {sim.time:.3f} s")
    print(f"buoyant weight = {weight:.3f} N, dry weight = {weight * DENSITY / (DENSITY - 1000.0):.3f} N, "
          f"base force = {np.mean(base):.3f} +- {np.std(base):.3f} N")
    print(f"U_mf = {critical:.4f} m/s, Ergun = {ergun_velocity(sim):.4f} m/s")
    start_frame = len(sim.frames) - 1
    prediction = [(u, max(0.0, 1.0 - example.fixed_bed_force(sim, u) / weight))
                  for u in np.linspace(0.0, example.RAMP_TOP * critical, 81)]
    for u in (0.25, 0.5, 0.75, 1.0):
        ratio = 1.0 - example.fixed_bed_force(sim, u * critical) / weight
        print(f"  fixed bed at {u:.2f} U_mf: base / W' = {ratio:.3f}")
    rows = example.run_ramp(sim, critical)
    for velocity, ratio, top, porosity in rows:
        print(f"  U/U_mf = {velocity / critical:.3f}  base/W' = {ratio:.3f}  top = {top:.4f} m  "
              f"porosity = {porosity:.3f}")
    print(f"removed = {sim.n_removed}")
    example.plot_fluidization(rows, prediction, critical, str(FIGURES / "seepage_fluidization.png"))
    render_frame(sim, str(FIGURES / "seepage_start.png"), start_frame, width=3.5)
    render_frame(sim, str(FIGURES / "seepage_end.png"), -1, width=3.5)


if __name__ == "__main__":
    main()
