"""Cone penetration into a granular bed: a rigid 60 degree cone pushed into the fill of a box.

The cone is a rigid body of twelve triangular faces that meet at its tip, all carrying the same ``LoadedPlaten``.
Its base is a regular dodecagon in the middle of the top of the box, so the rest of the top is a square with a
dodecagonal hole. A face of a ``Geometry3D`` must be a simple polygon, so that ring is split into two faces, each a
non-convex polygon around one half of the hole, and the back and front faces list the vertex in the middle of their
top edge, where the split meets them. Every face is listed counter-clockwise seen from outside the domain.

The box is filled to about two thirds of its height, and the cone is seated on the fill with its tip, pushed in with
a load ramped to its peak and held. Run it with::

    python examples/cone_penetration_3d.py --peak 20

and the animation, a vertical section through the axis of the cone and the load curves are written to ``output/``.
The run takes a few minutes.
"""

import argparse
import math
import os
from typing import List, Sequence, Tuple

import numpy as np

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry3D, LoadedPlaten, MaterialProperties, Open,
                        RainSettings, TimeHistory, Wall)
from DEMcookies.loads import LoadFunction
from DEMcookies.plotting import plot_load_curves, render_animation, render_frame

OUTPUT_DIR = "output"

# box width (x), height (y) and depth (z) [m]
WIDTH = 0.16
HEIGHT = 0.2
DEPTH = 0.16
# radius of the base of the cone [m], half of its apex angle [rad] and number of its faces, a multiple of four
CONE_RADIUS = 0.03
HALF_ANGLE = math.radians(30.0)
CONE_FACES = 12

RAIN = RainSettings(batch_size=40, steps_between=250, insertion_band=(0.11, 0.13), record_every=250, seed=6)

Point = Tuple[float, float, float]


def outward(points: Sequence[Point], face: Sequence[int], direction: Point) -> Tuple[int, ...]:
    """Order the vertices of a face counter-clockwise as seen from outside the domain.

    Args:
        points: vertices of the domain [m].
        face: indices of the vertices of the face, in order around it in either sense.
        direction: any direction out of the domain through the face.

    Returns:
        The indices, reversed if their right-hand normal points into the domain.
    """
    polygon = np.array([points[k] for k in face])
    normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
    return tuple(face) if float(normal @ np.array(direction)) > 0.0 else tuple(reversed(face))


def box_with_cone(load: LoadFunction) -> Geometry3D:
    """Define the box, whose top is the base of the cone and a ring of two faces around it.

    Args:
        load: load history of the cone, positive downwards [N].

    Returns:
        The domain.
    """
    cx, cz = 0.5 * WIDTH, 0.5 * DEPTH
    points: List[Point] = []

    def vertex(x: float, y: float, z: float) -> int:
        points.append((x, y, z))
        return len(points) - 1

    corners = [(0.0, 0.0), (WIDTH, 0.0), (WIDTH, DEPTH), (0.0, DEPTH)]
    floor = [vertex(x, 0.0, z) for x, z in corners]
    rim = [vertex(x, HEIGHT, z) for x, z in corners]
    # the line x = cx splits the top in two, and meets the back and front faces in the middle of their top edges
    back_middle = vertex(cx, HEIGHT, 0.0)
    front_middle = vertex(cx, HEIGHT, DEPTH)
    # the base of the cone runs from its front point (high z) round its right (high x), back and left sides
    angles = [0.5 * math.pi - 2.0 * math.pi * k / CONE_FACES for k in range(CONE_FACES)]
    base = [vertex(cx + CONE_RADIUS * math.cos(a), HEIGHT, cz + CONE_RADIUS * math.sin(a)) for a in angles]
    tip = vertex(cx, HEIGHT - CONE_RADIUS / math.tan(HALF_ANGLE), cz)
    half = CONE_FACES // 2

    # its weight keeps the cone on the fill once the load is removed
    cone = LoadedPlaten(load=load, mass=0.3, self_weight=True, direction=(0.0, -1.0, 0.0))
    faces_and_conditions: List[Tuple[Tuple[int, ...], BoundaryCondition]] = [
        (outward(points, floor, (0.0, -1.0, 0.0)), Wall()),
        (outward(points, [floor[1], rim[1], rim[2], floor[2]], (1.0, 0.0, 0.0)), Wall(friction=0.0)),
        (outward(points, [floor[0], floor[3], rim[3], rim[0]], (-1.0, 0.0, 0.0)), Wall(friction=0.0)),
        (outward(points, [floor[0], rim[0], back_middle, rim[1], floor[1]], (0.0, 0.0, -1.0)), Wall(friction=0.0)),
        (outward(points, [floor[3], floor[2], rim[2], front_middle, rim[3]], (0.0, 0.0, 1.0)), Wall(friction=0.0)),
        # the two halves of the top around the cone, each a non-convex polygon
        (outward(points, [back_middle, rim[1], rim[2], front_middle, *base[:half + 1]], (0.0, 1.0, 0.0)), Open()),
        (outward(points, [front_middle, rim[3], rim[0], back_middle, *base[half:], base[0]], (0.0, 1.0, 0.0)), Open()),
    ]
    for k in range(CONE_FACES):
        first, second = base[k], base[(k + 1) % CONE_FACES]
        middle = 0.5 * (np.array(points[first]) + np.array(points[second]))
        # out of the domain is into the cone, towards its axis and upwards
        into_cone = (cx - float(middle[0]), 1.0, cz - float(middle[2]))
        faces_and_conditions.append((outward(points, [first, second, tip], into_cone), cone))
    faces = [face for face, _ in faces_and_conditions]
    conditions = [condition for _, condition in faces_and_conditions]
    return Geometry3D(points, faces, conditions)


def build_simulation(load: LoadFunction) -> DEMSimulation:
    """Define the box with the cone, filled with a slightly angular material.

    Args:
        load: load history of the cone, positive downwards [N].

    Returns:
        The simulation, with an empty box.
    """
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5,
                                  rolling_friction=0.1)
    return DEMSimulation(box_with_cone(load), material, radius_range=(0.007, 0.01))


def main(output_dir: str = OUTPUT_DIR, peak: float = 20.0) -> None:
    """Fill the box, push the cone in and write the results.

    Args:
        output_dir: directory of the animation and the figures.
        peak: peak load of the cone [N].
    """
    os.makedirs(output_dir, exist_ok=True)
    # the load is ramped up in 0.8 s and then held
    sim = build_simulation(TimeHistory(times=[0.0, 0.8], values=[0.0, peak]))
    n = sim.rain_until_full(RAIN)
    print(f"filled with {n} particles; bed top = {sim.bed_top():.3f} m")
    sim.rest(0.05, record_every=250)
    sim.compress(1.0, record_every=100)
    cone = sim.platens[0]
    print(f"cone penetration = {cone.displacement * 1e3:.1f} mm, reaction = {cone.reaction:.2f} N")

    render_animation(sim, os.path.join(output_dir, "cone_penetration_3d.gif"), fps=20)
    render_frame(sim, os.path.join(output_dir, "cone_penetration_3d_section.png"), section=0.5 * DEPTH)
    plot_load_curves(sim, os.path.join(output_dir, "cone_penetration_3d_curves.png"))
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--peak", type=float, default=20.0, help="peak load of the cone [N]")
    main(peak=parser.parse_args().peak)
