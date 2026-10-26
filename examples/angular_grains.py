"""Strip footing on round and on angular grains: what the shape of the grains does to the bearing response.

The same strip footing is pressed into three fills of the same grading in a box: discs that roll freely, discs whose
rolling is resisted (the usual stand-in for the shape of real grains), and convex polygons, irregular hexagons and
pentagons in random orientations whose corners interlock. Each fill is rained into the box, the footing is seated on
it, and the load is ramped up to its peak. The load-settlement curves show how much stiffer and stronger the angular
grains are than free discs, and how far rolling resistance goes towards them.

A polygon of radius R has the area of the disc of radius R, so the three fills have the same grading and the same
mass. Run it with::

    python examples/angular_grains.py --peak 6

and the load-settlement curves and a snapshot of each fill under the peak load are written to ``output/``. The run
takes a few minutes.
"""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from DEMcookies import (ConvexPolygon, DEMSimulation, Geometry, LoadedPlaten, MaterialProperties, Open, RainSettings,
                        TimeHistory, Wall)
from DEMcookies.plotting import render_frame
from DEMcookies.shapes import Shape

OUTPUT_DIR = "output"

# box width and height, and width of the footing in the middle of its top [m]
WIDTH = 0.24
HEIGHT = 0.2
FOOTING = 0.05

RAIN = RainSettings(batch_size=30, steps_between=400, insertion_band=(0.155, 0.18), record_every=200, seed=5)

# irregular grains drawn once, so that every polygon of a fill is one of a few shapes, as with a sieved sand
GRAINS: List[Optional[Shape]] = [
    ConvexPolygon.random(vertices, seed=seed, aspect=1.3) for seed, vertices in enumerate((5, 6, 6, 7))
]

# name, rolling friction coefficient and shapes of each fill
FILLS: List[Tuple[str, float, Optional[List[Optional[Shape]]]]] = [
    ("discs", 0.0, None),
    ("discs with rolling resistance", 0.2, None),
    ("polygons", 0.0, GRAINS),
]


def build_simulation(peak: float, rolling_friction: float,
                     shapes: Optional[Sequence[Optional[Shape]]]) -> DEMSimulation:
    """Define the box with the strip footing, for one fill.

    Args:
        peak: largest load of the footing [N].
        rolling_friction: rolling resistance coefficient of the grains.
        shapes: shapes of the grains, None for discs.

    Returns:
        The simulation, with an empty box.
    """
    left = 0.5 * (WIDTH - FOOTING)
    load = TimeHistory(times=[0.0, 0.8], values=[0.0, peak])
    footing = LoadedPlaten(load=load, mass=0.05, self_weight=True, seat_on_bed=True)
    geometry = Geometry([(0.0, 0.0), (WIDTH, 0.0), (WIDTH, HEIGHT), (left + FOOTING, HEIGHT), (left, HEIGHT),
                         (0.0, HEIGHT)],
                        [Wall(), Wall(), Open(), footing, Open(), Wall()])
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.3,
                                  friction=0.5,
                                  rolling_friction=rolling_friction)
    return DEMSimulation(geometry, material, radius_range=(0.005, 0.008), shapes=shapes)


def main(output_dir: str = OUTPUT_DIR, peak: float = 6.0) -> None:
    """Load the footing on each fill and compare the load-settlement curves.

    Args:
        output_dir: directory of the figures.
        peak: peak load of the footing [N].
    """
    os.makedirs(output_dir, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6.0, 4.0))
    for name, rolling_friction, shapes in FILLS:
        sim = build_simulation(peak, rolling_friction, shapes)
        n = sim.rain_until_full(RAIN)
        sim.rest(0.1, record_every=200)
        sim.compress(0.9, record_every=50)
        curve = sim.load_curves[0].as_array()
        print(f"{name}: {n} grains, settlement under {peak:g} N = {curve[-1, 2] * 1e3:.2f} mm")
        ax.plot(curve[:, 2] * 1e3, curve[:, 1], label=name)
        render_frame(sim, os.path.join(output_dir, f"angular_grains_{name.replace(' ', '_')}.png"))
    ax.set_xlabel("settlement [mm]")
    ax.set_ylabel("applied load [N]")
    ax.legend()
    ax.grid(alpha=0.3)
    figure.savefig(os.path.join(output_dir, "angular_grains_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--peak", type=float, default=6.0, help="peak load of the footing [N]")
    main(peak=parser.parse_args().peak)
