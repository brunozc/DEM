"""Run the reference DEMcookies simulation and visualise the result.

The script

1. rains polydisperse circular particles into a 0.2 m x 0.6 m box under gravity until it is full,
2. lets the packing settle,
3. seats the top edge, a loaded platen, on the packing and applies a load ramped to 1000 N in 0.4 s,

and writes an animation of the process and the load-displacement response to the ``output`` folder.

Every input is defined in ``build_simulation``. To model another problem, change the geometry vertices, the boundary
condition of each edge (``Wall``, ``Open`` or ``LoadedPlaten``) and the load history of the platen (``Pulse``,
``Heaviside``, ``Sinusoidal``, ``TimeHistory`` or any function of time). See ``examples/`` for a non-rectangular case.
"""

import os

from DEMcookies import (DEMSimulation, Geometry, GlobalSettings, Heaviside, LoadedPlaten, MaterialProperties,
                        RainSettings, Wall)
from DEMcookies.plotting import plot_load_curves, render_animation

OUTPUT_DIR = "output"


def build_simulation() -> DEMSimulation:
    """Define the geometry, boundary conditions, load, material and numerical settings of the reference case.

    Returns:
        The simulation, with an empty box.
    """
    load = Heaviside(amplitude=1000.0, rise_time=0.4)
    # the top edge closes the box during filling and is then seated on the bed and loaded
    geometry = Geometry.rectangle(width=0.2,
                                  height=0.6,
                                  bottom=Wall(),
                                  right=Wall(),
                                  top=LoadedPlaten(load=load, mass=0.5),
                                  left=Wall())
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5)
    return DEMSimulation(geometry, material, radius_range=(0.004, 0.02), settings=GlobalSettings(gravity=9.81))


def main(pause: float = 0.1, load_duration: float = 1.0) -> None:
    """Run the deposition, settling and loading stages and write the results.

    Args:
        pause: settling time between deposition and loading [s].
        load_duration: duration of the loading stage [s].
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_simulation()
    print(f"dt = {sim.dt:.3e} s")

    print("Raining particles into the box until it is full ...")
    n = sim.rain_until_full(RainSettings(batch_size=15, steps_between=450, record_every=60, seed=7))
    print(f"  box filled with {n} particles; bed top = {sim.bed_top():.3f} m, "
          f"mean KE = {sim.mean_kinetic_energy():.2e} J")

    print(f"Pausing for {pause:.3f} s to let the packing settle ...")
    sim.rest(pause, record_every=60)

    print("Applying the time-dependent load through the platen ...")
    sim.compress(load_duration, record_every=60)
    for platen in sim.platens:
        print(f"  edge {platen.edge}: displacement = {platen.displacement * 1e3:.3f} mm, "
              f"reaction = {platen.reaction:.2f} N")

    render_animation(sim, os.path.join(OUTPUT_DIR, "simulation.gif"))
    plot_load_curves(sim, os.path.join(OUTPUT_DIR, "load_displacement.png"))
    print(f"Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
