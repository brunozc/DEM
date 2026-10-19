"""Strip footing on a granular fill in a trapezoidal container.

The domain is a trapezoid that widens upwards. The bottom and the left side are rough walls, the right side is a
smooth wall, and the top is open except for a 0.1 m strip in the middle: that strip is a loaded platen, i.e. a
rigid footing. The top side is split into three collinear edges to define where the load is applied. The container
is filled to about 0.2 m, so the footing is seated on the fill 5 cm below the rim.

Together with the weight of the footing, about 1 N, the loads stay at about half the bearing capacity. With this
soft, rolling-prone material a slow ramp gives a stiffening response up to about 8 N and failure between 8 and 10 N
of applied load, after which the footing punches into the fill.

The load history is chosen on the command line, e.g.::

    python examples/footing_on_trapezoid.py sinusoidal

and the animation and load curves are written to ``output/``.
"""

import argparse
import math
import os
from typing import Dict, List

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry, Heaviside, LoadedPlaten, MaterialProperties, Open,
                        Pulse, RainSettings, Sinusoidal, TimeHistory, Wall)
from DEMcookies.loads import LoadFunction
from DEMcookies.plotting import plot_load_curves, render_animation

OUTPUT_DIR = "output"

RAIN = RainSettings(batch_size=25, steps_between=400, insertion_band=(0.19, 0.2), record_every=100, seed=3)

LOADS: Dict[str, LoadFunction] = {
    # 4 N for 0.2 s, starting 0.1 s after the footing is seated
    "pulse": Pulse(amplitude=4.0, duration=0.2, start_time=0.1),
    # sudden 4 N step at 0.1 s, held
    "heaviside": Heaviside(amplitude=4.0, start_time=0.1),
    # cyclic load between 0 and 4 N at 4 Hz, starting smoothly from zero
    "sinusoidal": Sinusoidal(amplitude=2.0, frequency=4.0, mean=2.0, phase=-0.5 * math.pi),
    # loading, partial unloading and reloading to a higher level
    "history": TimeHistory(times=[0.0, 0.2, 0.4, 0.6], values=[0.0, 3.0, 1.0, 4.0]),
}


def build_simulation(load: LoadFunction) -> DEMSimulation:
    """Define the trapezoidal domain with a strip footing.

    Args:
        load: load history of the footing [N].

    Returns:
        The simulation, with an empty domain.
    """
    vertices = [
        (0.05, 0.0),  # bottom left
        (0.35, 0.0),  # bottom right
        (0.40, 0.25),  # top right
        (0.25, 0.25),  # right end of the footing
        (0.15, 0.25),  # left end of the footing
        (0.00, 0.25),  # top left
    ]
    boundary_conditions: List[BoundaryCondition] = [
        Wall(),  # bottom
        Wall(friction=0.0),  # smooth right side
        Open(),  # top, right of the footing
        LoadedPlaten(load=load, mass=0.1, self_weight=True),  # footing, which rests on the fill under its weight
        Open(),  # top, left of the footing
        Wall(),  # rough left side
    ]
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5)
    return DEMSimulation(Geometry(vertices, boundary_conditions), material, radius_range=(0.004, 0.008))


def main() -> None:
    """Fill the container, load the footing and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("load", choices=sorted(LOADS), help="load history of the footing")
    parser.add_argument("--duration", type=float, default=1.0, help="duration of the loading stage [s]")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_simulation(LOADS[args.load])
    n = sim.rain_until_full(RAIN)
    print(f"filled with {n} particles; bed top = {sim.bed_top():.3f} m")
    sim.rest(0.1, record_every=100)
    sim.compress(args.duration, record_every=50)
    footing = sim.platens[0]
    print(f"footing displacement = {footing.displacement * 1e3:.2f} mm, reaction = {footing.reaction:.2f} N")

    render_animation(sim, os.path.join(OUTPUT_DIR, f"footing_{args.load}.gif"))
    plot_load_curves(sim, os.path.join(OUTPUT_DIR, f"footing_{args.load}.png"))
    print(f"Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
