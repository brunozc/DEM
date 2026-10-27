"""Square footing on angular stones: a granular bed of polyhedra, the shape of crushed rock such as railway ballast.

Every grain is one of a few irregular convex polyhedra, the hulls of random points on an ellipsoid, rained into a box
in random orientations. The top of the box is split into a square footing, a loaded platen, and four open faces around
it, as in ``square_footing_3d.py``. The footing is seated on the bed, loaded, held and unloaded.

The grains need no rolling resistance: their corners and flat faces interlock, and a face resting on a face resists
twisting. A polyhedron of radius R has the volume of the sphere of radius R, so the grading and the mass of the bed
are those of a bed of spheres of the same radii. Run it with::

    python examples/angular_grains_3d.py --peak 15 --save-every-n-steps 500

and the animation, a vertical section through the footing and the load curves are written to ``output/``, and the
ParaView files, which show the faces of the stones, to ``output/angular_grains_3d_vtk/``. The run takes about twelve
minutes.
"""

import argparse
import os
from typing import List, Optional, Tuple

from DEMcookies import (BoundaryCondition, ConvexPolyhedron, DEMSimulation, Geometry3D, LoadedPlaten,
                        MaterialProperties, Open, RainSettings, TimeHistory, Wall)
from DEMcookies.loads import LoadFunction
from DEMcookies.plotting import plot_load_curves, render_animation, render_frame
from DEMcookies.vtk_export import VTKWriter

OUTPUT_DIR = "output"

# box width (x), height (y) and depth (z), and side of the footing in the middle of its top [m]
WIDTH = 0.12
HEIGHT = 0.12
DEPTH = 0.12
FOOTING = 0.05

RAIN = RainSettings(batch_size=30, steps_between=300, insertion_band=(0.075, 0.095), record_every=200, seed=4)

# stones with ten to fourteen corners, some elongated and some flaky
STONES = [
    ConvexPolyhedron.random(12, seed=1),
    ConvexPolyhedron.random(10, seed=2, aspect=(1.4, 1.0, 0.8)),
    ConvexPolyhedron.random(14, seed=3, aspect=(1.2, 0.7, 1.0)),
    ConvexPolyhedron.random(11, seed=4, aspect=(1.0, 1.0, 0.75)),
]


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
    footing = LoadedPlaten(load=load, mass=0.1, self_weight=True, direction=(0.0, -1.0, 0.0))
    faces_and_conditions: List[Tuple[Tuple[int, ...], BoundaryCondition]] = [
        ((0, 1, 2, 3), Wall()),  # floor
        ((1, 5, 6, 2), Wall()),  # right side
        ((4, 8, 9, 5), Open()),  # top, behind the footing
        ((5, 9, 10, 6), Open()),  # top, right of the footing
        ((6, 10, 11, 7), Open()),  # top, in front of the footing
        ((7, 11, 8, 4), Open()),  # top, left of the footing
        ((8, 11, 10, 9), footing),  # the footing
        ((3, 7, 4, 0), Wall()),  # left side
        ((0, 4, 5, 1), Wall()),  # back
        ((3, 2, 6, 7), Wall()),  # front
    ]
    faces = [face for face, _ in faces_and_conditions]
    conditions = [condition for _, condition in faces_and_conditions]
    return Geometry3D(vertices, faces, conditions)


def build_simulation(load: LoadFunction) -> DEMSimulation:
    """Define the box with the square footing, to be filled with stones.

    Args:
        load: load history of the footing, positive downwards [N].

    Returns:
        The simulation, with an empty box.
    """
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.3,
                                  friction=0.6)
    return DEMSimulation(box_with_square_footing(load), material, radius_range=(0.008, 0.011), shapes=STONES)


def main(output_dir: str = OUTPUT_DIR, peak: float = 15.0, save_every_n_steps: Optional[int] = None) -> None:
    """Fill the box, load the footing and write the results.

    Args:
        output_dir: directory of the animation, the figures and the ParaView files.
        peak: peak load of the footing [N].
        save_every_n_steps: number of time steps between two ParaView exports; None writes none.
    """
    os.makedirs(output_dir, exist_ok=True)
    sim = build_simulation(loading_history(peak))
    if save_every_n_steps is not None:
        sim.add_observer(VTKWriter(os.path.join(output_dir, "angular_grains_3d_vtk"), save_every_n_steps))
    n = sim.rain_until_full(RAIN)
    print(f"filled with {n} stones; bed top = {sim.bed_top():.3f} m; time step = {sim.dt:.2e} s")
    sim.rest(0.1, record_every=200)
    sim.compress(1.0, record_every=100)
    footing = sim.platens[0]
    print(f"footing displacement = {footing.displacement * 1e3:.2f} mm, reaction = {footing.reaction:.2f} N")

    render_animation(sim, os.path.join(output_dir, "angular_grains_3d.gif"), fps=20)
    render_frame(sim, os.path.join(output_dir, "angular_grains_3d_section.png"), section=0.5 * DEPTH)
    plot_load_curves(sim, os.path.join(output_dir, "angular_grains_3d_curves.png"))
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--peak", type=float, default=15.0, help="peak load of the footing [N]")
    parser.add_argument("--save-every-n-steps",
                        type=int,
                        default=None,
                        help="number of time steps between two ParaView exports; none by default")
    arguments = parser.parse_args()
    main(peak=arguments.peak, save_every_n_steps=arguments.save_every_n_steps)
