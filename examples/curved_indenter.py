"""Curved rigid indenter pressed into a granular fill in a stepped container.

The container has a stepped floor, whose upper corner is a re-entrant vertex, a vertical left wall and an inclined
right wall with low friction. The load is not applied on a straight line: it acts on a rigid indenter whose lower
surface is a circular arc, approximated by several edges. All of those edges carry the same ``LoadedPlaten``
object, so they form a single rigid body that moves vertically as one.

The container is filled to about 0.23 m, leaving freeboard below the indenter, which is then seated on the fill and
loaded, held and unloaded with a user-defined time history. Run it with::

    python examples/curved_indenter.py --peak 4

and the animation and load curves are written to ``output/``.
"""

import argparse
import math
import os
from typing import List, Tuple

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry, LoadedPlaten, MaterialProperties, Open,
                        RainSettings, TimeHistory, Wall)
from DEMcookies.loads import LoadFunction
from DEMcookies.plotting import plot_load_curves, render_animation

OUTPUT_DIR = "output"

# lower surface of the indenter: arc of a circle that cuts the top of the container
INDENTER_CENTRE = (0.19, 0.34)
INDENTER_RADIUS = 0.07
INDENTER_EDGES = 8
TOP = 0.30

RAIN = RainSettings(batch_size=30, steps_between=400, insertion_band=(0.22, 0.23), record_every=100, seed=11)


def indenter_arc() -> List[Tuple[float, float]]:
    """Return the vertices of the indenter surface, from its right end to its left end at the top of the container.

    Returns:
        The arc vertices (x, y) [m].
    """
    cx, cy = INDENTER_CENTRE
    # the arc starts and ends where the circle crosses the top of the container
    half_angle = math.asin((cy - TOP) / INDENTER_RADIUS)
    right, left = -half_angle, -math.pi + half_angle
    angles = [right + (left - right) * k / INDENTER_EDGES for k in range(INDENTER_EDGES + 1)]
    return [(cx + INDENTER_RADIUS * math.cos(a), cy + INDENTER_RADIUS * math.sin(a)) for a in angles]


def loading_history(peak: float) -> TimeHistory:
    """Ramp the load to its peak in 0.4 s, hold it for 0.3 s and remove it in 0.3 s.

    Args:
        peak: largest load [N].

    Returns:
        The load history.
    """
    return TimeHistory(times=[0.0, 0.4, 0.7, 1.0], values=[0.0, peak, peak, 0.0])


def build_simulation(load: LoadFunction) -> DEMSimulation:
    """Define the stepped container with the curved indenter.

    Args:
        load: load history of the indenter, positive downwards [N].

    Returns:
        The simulation, with an empty container.
    """
    # its weight keeps the indenter on the fill once the load is removed
    indenter = LoadedPlaten(load=load, mass=0.1, self_weight=True, direction=(0.0, -1.0))
    arc = indenter_arc()
    vertices = [
        (0.00, 0.06),  # left end of the upper floor
        (0.12, 0.06),  # re-entrant corner of the step
        (0.12, 0.00),  # foot of the step
        (0.30, 0.00),  # foot of the inclined wall
        (0.38, TOP),  # top of the inclined wall
        *arc,  # indenter surface, right to left
        (0.00, TOP),  # top of the left wall
    ]
    boundary_conditions: List[BoundaryCondition] = [
        Wall(),  # upper floor
        Wall(),  # riser of the step
        Wall(),  # lower floor
        Wall(friction=0.2),  # inclined right wall
        Open(),  # top, right of the indenter
        *[indenter] * INDENTER_EDGES,  # one rigid body
        Open(),  # top, left of the indenter
        Wall(),  # left wall
    ]
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5)
    return DEMSimulation(Geometry(vertices, boundary_conditions), material, radius_range=(0.003, 0.006))


def main() -> None:
    """Fill the container, load the indenter and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--peak", type=float, default=4.0, help="peak load of the indenter [N]")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_simulation(loading_history(args.peak))
    n = sim.rain_until_full(RAIN)
    print(f"filled with {n} particles; bed top = {sim.bed_top():.3f} m")
    sim.rest(0.1, record_every=100)
    sim.compress(1.2, record_every=50)
    indenter = sim.platens[0]
    print(f"indenter displacement = {indenter.displacement * 1e3:.2f} mm, reaction = {indenter.reaction:.2f} N")

    render_animation(sim, os.path.join(OUTPUT_DIR, "indenter.gif"))
    plot_load_curves(sim, os.path.join(OUTPUT_DIR, "indenter.png"))
    print(f"Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
