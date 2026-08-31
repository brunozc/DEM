"""Run the DEMcookies 2D discrete element model and visualise the result.

The script

1. rains polydisperse circular particles into the box under gravity until
   it is full,
2. closes the top with a rigid platen and applies a time-dependent load,

and finally writes an animation of the process and the load-displacement
response to the ``output`` folder.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

from DEMcookies import (
    Box,
    DEMSimulation,
    LoadingPlaten,
    MaterialProperties,
    linear_ramp,
)

OUTPUT_DIR = "output"


def build_simulation() -> DEMSimulation:
    box = Box(width=0.2, height=0.6)
    material = MaterialProperties(
        density=2500.0,
        youngs_modulus=5.0e6,
        poisson_ratio=0.25,
        restitution=0.4,
        friction=0.5,
    )
    platen = LoadingPlaten(width=box.width, mass=0.5)
    load_fn = linear_ramp(f_max=1000.0, t_ramp=0.4, hold=True)

    # start with an empty box; particles are rained in until it fills up
    return DEMSimulation(box, material, [], platen, load_fn, gravity=9.81,
                         radius_range=(0.004, 0.02))


def render_animation(sim: DEMSimulation, path: str) -> None:
    frames = sim.frames
    box = sim.box

    # axis and colour limits across all (variable size) frames
    y_max = max(float((pos[:, 1] + rad).max())
                for _, pos, rad, *_ in frames if len(rad))
    y_max = max(y_max, box.height) * 1.02
    speeds = np.concatenate([f[3] for f in frames if len(f[3])])
    v_max = max(0.05, float(np.percentile(speeds, 95)))
    norm = Normalize(0.0, v_max)

    fig, ax = plt.subplots(figsize=(5, 5 * y_max / box.width))
    ax.set_xlim(0.0, box.width)
    ax.set_ylim(0.0, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # fixed walls
    ax.plot([0, 0], [0, y_max], color="k", lw=2)
    ax.plot([box.width, box.width], [0, y_max], color="k", lw=2)
    ax.plot([0, box.width], [0, 0], color="k", lw=2)

    platen_line, = ax.plot([], [], color="firebrick", lw=4)
    title = ax.set_title("")

    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="speed [m/s]", shrink=0.8)

    def update(k: int):
        for collection in list(ax.collections):
            collection.remove()
        t, pos, rad, speed, platen_y, phase = frames[k]
        diameters = 2.0 * rad
        circles = EllipseCollection(
            widths=diameters, heights=diameters, angles=np.zeros(len(rad)),
            units="xy", offsets=pos, offset_transform=ax.transData,
            cmap="viridis", norm=norm, edgecolor="k", linewidth=0.3,
        )
        circles.set_array(speed)
        ax.add_collection(circles)
        if platen_y is not None:
            platen_line.set_data([0, box.width], [platen_y, platen_y])
        else:
            platen_line.set_data([], [])
        title.set_text(f"{phase}   t = {t:.3f} s   N = {len(rad)}")
        return ()

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)
    try:
        anim.save(path, writer=PillowWriter(fps=25))
        print(f"Saved animation to {path}")
    except Exception as exc:  # pragma: no cover - depends on optional Pillow
        print(f"Could not save GIF ({exc}); saving final frame instead.")
        update(len(frames) - 1)
        fig.savefig(path.replace(".gif", "_final.png"), dpi=120)
    plt.close(fig)


def plot_load_curve(sim: DEMSimulation, path: str) -> None:
    if not sim.load_curve:
        return
    data = np.array(sim.load_curve)
    t, applied, disp, reaction = data.T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t, applied, label="applied load F(t)")
    ax1.plot(t, reaction, "--", label="contact reaction")
    ax1.set_xlabel("loading time [s]")
    ax1.set_ylabel("force [N]")
    ax1.set_title("Time-dependent load")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(disp * 1e3, reaction)
    ax2.set_xlabel("platen displacement [mm]")
    ax2.set_ylabel("reaction force [N]")
    ax2.set_title("Load-displacement response")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved load curve to {path}")


def main(pause: float = 0.1) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_simulation()
    print(f"dt = {sim.dt:.2e} s")

    print("Raining particles into the box until it is full ...")
    n = sim.rain_until_full(batch_size=15, steps_between=450, record_every=60, seed=7)
    print(f"  box filled with {n} particles; bed top = {sim.bed_top():.3f} m, "
          f"mean KE = {sim.mean_kinetic_energy():.2e} J")

    print(f"Pausing for {pause:.3f} s to let the packing settle ...")
    sim.rest(pause, record_every=60)

    load_steps = int(1 / sim.dt)
    print("Applying time-dependent load through the platen ...")
    sim.compress(steps=load_steps, record_every=60)
    print(f"  final platen displacement = {sim.platen.displacement * 1e3:.3f} mm, "
          f"reaction = {sim.platen.fy:.2f} N")

    render_animation(sim, os.path.join(OUTPUT_DIR, "simulation.gif"))
    plot_load_curve(sim, os.path.join(OUTPUT_DIR, "load_displacement.png"))


if __name__ == "__main__":
    main()
