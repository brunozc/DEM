"""Discharge of a bin through a hopper: a trap door opens and the grains drain out at a constant rate.

The domain is a square bin whose lower part narrows into a hopper with a square outlet. The outlet is a trap door,
a loaded platen that holds the grains while the bin is filled. When it is released its load pulls it down, away from
the outlet, much faster than the grains can follow, and they drain out of the bin. Grains that leave the bin fall
out of the domain and are removed once they are clear of it, as through any open face.

A granular bed drains at a nearly constant rate, whatever the height of material above the outlet, unlike a liquid,
whose outflow slows as its level falls. Arches of grains that form and collapse over the outlet carry the weight of
the material above, so the flow depends on the outlet and not on the head. The mass discharged therefore grows
linearly with time, as the plot of the discharge shows. Run it with::

    python examples/hopper_discharge_3d.py

and the animation, a vertical section through the outlet and the discharge curve are written to ``output/``. The
run takes a few minutes.
"""

import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry3D, Heaviside, LoadedPlaten, MaterialProperties, Open,
                        RainSettings, Wall)
from DEMcookies.plotting import render_animation, render_frame

OUTPUT_DIR = "output"

# side of the bin, height of the hopper, height of the bin and side of the outlet [m]
WIDTH = 0.12
HOPPER_HEIGHT = 0.05
HEIGHT = 0.24
OUTLET = 0.07

RAIN = RainSettings(batch_size=40, steps_between=250, insertion_band=(0.19, 0.21), record_every=250, seed=8)
DISCHARGE_TIME = 1.0
SAMPLE_INTERVAL = 0.02

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


def bin_with_hopper() -> Geometry3D:
    """Define the bin, the hopper below it and the trap door that closes the outlet.

    Returns:
        The domain.
    """
    centre = 0.5 * WIDTH
    low, high = centre - 0.5 * OUTLET, centre + 0.5 * OUTLET
    points: List[Point] = []
    outlet = [len(points) + k for k in range(4)]
    points += [(x, 0.0, z) for x, z in ((low, low), (high, low), (high, high), (low, high))]
    shoulder = [len(points) + k for k in range(4)]
    points += [(x, HOPPER_HEIGHT, z) for x, z in ((0.0, 0.0), (WIDTH, 0.0), (WIDTH, WIDTH), (0.0, WIDTH))]
    rim = [len(points) + k for k in range(4)]
    points += [(x, HEIGHT, z) for x, z in ((0.0, 0.0), (WIDTH, 0.0), (WIDTH, WIDTH), (0.0, WIDTH))]

    # the trap door falls away from the outlet as soon as it is released, much faster than the grains
    door = LoadedPlaten(load=Heaviside(amplitude=5.0),
                        mass=0.05,
                        self_weight=True,
                        seat_on_bed=False,
                        direction=(0.0, -1.0, 0.0))
    faces_and_conditions: List[Tuple[Tuple[int, ...], BoundaryCondition]] = [
        (outward(points, outlet, (0.0, -1.0, 0.0)), door),
        (outward(points, rim, (0.0, 1.0, 0.0)), Open()),
    ]
    for k in range(4):
        following = (k + 1) % 4
        # the sloping walls of the hopper and the walls of the bin above them face away from the axis of the bin
        middle = 0.5 * (np.array(points[shoulder[k]]) + np.array(points[shoulder[following]]))
        away = (float(middle[0]) - centre, 0.0, float(middle[2]) - centre)
        hopper = [outlet[k], outlet[following], shoulder[following], shoulder[k]]
        side = [shoulder[k], shoulder[following], rim[following], rim[k]]
        faces_and_conditions.append((outward(points, hopper, away), Wall()))
        faces_and_conditions.append((outward(points, side, away), Wall()))
    faces = [face for face, _ in faces_and_conditions]
    conditions = [condition for _, condition in faces_and_conditions]
    return Geometry3D(points, faces, conditions)


def build_simulation() -> DEMSimulation:
    """Define the bin, filled with a slightly angular material.

    Returns:
        The simulation, with an empty bin.
    """
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=0.4,
                                  friction=0.5,
                                  rolling_friction=0.1)
    return DEMSimulation(bin_with_hopper(), material, radius_range=(0.006, 0.008))


def mass_in_bin(sim: DEMSimulation) -> float:
    """Return the mass of the grains still in the bin, above the outlet [kg]."""
    return float(np.sum(sim.mass[sim.pos[:, 1] > 0.0]))


def discharge(sim: DEMSimulation) -> npt.NDArray[np.float64]:
    """Open the trap door and sample the mass of the grains that have left the bin.

    Args:
        sim: simulation with the bin filled and at rest.

    Returns:
        Rows of time since the door opened [s] and discharged mass [kg].
    """
    initial = mass_in_bin(sim)
    rows = [(0.0, 0.0)]
    # releasing the loaded platens opens the trap door
    for _ in range(round(DISCHARGE_TIME / SAMPLE_INTERVAL)):
        sim.compress(SAMPLE_INTERVAL, record_every=100)
        rows.append((sim.load_time, initial - mass_in_bin(sim)))
    return np.array(rows)


def plot_discharge(rows: npt.NDArray[np.float64], path: str) -> float:
    """Plot the discharged mass against time, with a straight line fitted to its steady part.

    Args:
        rows: samples returned by ``discharge``.
        path: output image file.

    Returns:
        The steady mass flow rate [kg/s].
    """
    time, mass = rows.T
    # the flow is steady once it has started and until the bin runs nearly empty
    steady = (mass > 0.1 * mass[-1]) & (mass < 0.8 * mass[-1])
    rate, offset = np.polyfit(time[steady], mass[steady], 1)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(time, mass * 1e3, "o", markersize=3, label="simulation")
    ax.plot(time[steady], (rate * time[steady] + offset) * 1e3, "-", label=f"steady rate {rate * 1e3:.0f} g/s")
    ax.set_xlabel("time since the door opened [s]")
    ax.set_ylabel("discharged mass [g]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return float(rate)


def main(output_dir: str = OUTPUT_DIR) -> None:
    """Fill the bin, open the trap door and write the results.

    Args:
        output_dir: directory of the animation and the figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sim = build_simulation()
    n = sim.rain_until_full(RAIN)
    print(f"filled with {n} particles, {mass_in_bin(sim) * 1e3:.0f} g; bed top = {sim.bed_top():.3f} m")
    sim.rest(0.05, record_every=250)
    rows = discharge(sim)
    rate = plot_discharge(rows, os.path.join(output_dir, "hopper_discharge_3d_curve.png"))
    print(f"discharged {rows[-1, 1] * 1e3:.0f} g in {rows[-1, 0]:.2f} s; steady rate = {rate * 1e3:.0f} g/s, "
          f"{sim.n_removed} particles removed")

    render_animation(sim, os.path.join(output_dir, "hopper_discharge_3d.gif"), fps=20, every=2)
    render_frame(sim, os.path.join(output_dir, "hopper_discharge_3d_section.png"), index=-25, section=0.5 * WIDTH)
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    main()
