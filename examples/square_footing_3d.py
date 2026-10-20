"""Square footing on a granular bed in a box: a load that only a three-dimensional model can apply.

The domain is a box whose top is split into five faces: a square in the middle, which is a loaded platen, i.e. a
rigid footing, and four open faces around it. Every face of a ``Geometry3D`` is a planar polygon given by the indices
of its vertices, listed in the same sense of rotation seen from outside. The box is filled to about two thirds of its
height, the footing is seated on the bed and loaded, held and unloaded with a user-defined time history.

The y axis is vertical, as in the 2D examples, and z is the depth. A strip footing in 2D corresponds to a footing
that spans the whole depth of the box; ``Geometry3D.extrude`` builds such a prism from any 2D domain. Run it with::

    python examples/square_footing_3d.py --peak 15

and the animation, a vertical section through the footing and the load curves are written to ``output/``. The run
takes a few minutes.
"""

import argparse
import os
from typing import List, Tuple

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry3D, LoadedPlaten, MaterialProperties, Open,
                        RainSettings, TimeHistory, Wall)
from DEMcookies.loads import LoadFunction
from DEMcookies.plotting import plot_load_curves, render_animation, render_frame

OUTPUT_DIR = "output"

# box width (x), height (y) and depth (z), and side of the footing in the middle of its top [m]
WIDTH = 0.2
HEIGHT = 0.15
DEPTH = 0.2
FOOTING = 0.07

RAIN = RainSettings(batch_size=40, steps_between=250, insertion_band=(0.09, 0.11), record_every=200, seed=4)


def loading_history(peak: float) -> TimeHistory:
    """Ramp the load to its peak in 0.4 s, hold it for 0.2 s and remove it in 0.2 s.

    Args:
        peak: largest load [N].

    Returns:
        The load history.
    """
    return TimeHistory(times=[0.0, 0.4, 0.6, 0.8], values=[0.0, peak, peak, 0.0])


def box_with_square_footing(load: LoadFunction) -> Geometry3D:
    """Define the box, whose top is the footing surrounded by four open faces.

    Args:
        load: load history of the footing, positive downwards [N].

    Returns:
        The domain.
    """
    low = 0.5 * (WIDTH - FOOTING)
    high = low + FOOTING
    floor = [(0.0, 0.0, 0.0), (WIDTH, 0.0, 0.0), (WIDTH, 0.0, DEPTH), (0.0, 0.0, DEPTH)]
    rim = [(x, HEIGHT, z) for x, _, z in floor]
    corners = [(low, HEIGHT, low), (high, HEIGHT, low), (high, HEIGHT, high), (low, HEIGHT, high)]
    # vertices 0-3 are the corners of the floor, 4-7 those of the rim and 8-11 those of the footing
    vertices = floor + rim + corners
    # its weight keeps the footing on the bed once the load is removed
    footing = LoadedPlaten(load=load, mass=0.2, self_weight=True, direction=(0.0, -1.0, 0.0))
    faces_and_conditions: List[Tuple[Tuple[int, ...], BoundaryCondition]] = [
        ((0, 1, 2, 3), Wall()),  # floor
        ((1, 5, 6, 2), Wall(friction=0.0)),  # smooth right side
        ((4, 8, 9, 5), Open()),  # top, behind the footing
        ((5, 9, 10, 6), Open()),  # top, right of the footing
        ((6, 10, 11, 7), Open()),  # top, in front of the footing
        ((7, 11, 8, 4), Open()),  # top, left of the footing
        ((8, 11, 10, 9), footing),  # the footing
        ((3, 7, 4, 0), Wall(friction=0.0)),  # smooth left side
        ((0, 4, 5, 1), Wall(friction=0.0)),  # smooth back
        ((3, 2, 6, 7), Wall(friction=0.0)),  # smooth front
    ]
    faces = [face for face, _ in faces_and_conditions]
    conditions = [condition for _, condition in faces_and_conditions]
    return Geometry3D(vertices, faces, conditions)


def build_simulation(load: LoadFunction) -> DEMSimulation:
    """Define the box with the square footing, filled with a slightly angular material.

    Args:
        load: load history of the footing, positive downwards [N].

    Returns:
        The simulation, with an empty box.
    """
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5,
                                  rolling_friction=0.1)
    return DEMSimulation(box_with_square_footing(load), material, radius_range=(0.008, 0.012))


def main(output_dir: str = OUTPUT_DIR, peak: float = 15.0) -> None:
    """Fill the box, load the footing and write the results.

    Args:
        output_dir: directory of the animation and the figures.
        peak: peak load of the footing [N].
    """
    os.makedirs(output_dir, exist_ok=True)
    sim = build_simulation(loading_history(peak))
    n = sim.rain_until_full(RAIN)
    print(f"filled with {n} particles; bed top = {sim.bed_top():.3f} m")
    sim.rest(0.1, record_every=200)
    sim.compress(1.0, record_every=100)
    footing = sim.platens[0]
    print(f"footing displacement = {footing.displacement * 1e3:.2f} mm, reaction = {footing.reaction:.2f} N")

    render_animation(sim, os.path.join(output_dir, "square_footing_3d.gif"), fps=20)
    render_frame(sim, os.path.join(output_dir, "square_footing_3d_section.png"), section=0.5 * DEPTH)
    plot_load_curves(sim, os.path.join(output_dir, "square_footing_3d_curves.png"))
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--peak", type=float, default=15.0, help="peak load of the footing [N]")
    main(peak=parser.parse_args().peak)
