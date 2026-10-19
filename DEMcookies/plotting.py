"""Animation of the recorded frames and plots of the platen load curves."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

from .boundary import LoadedPlaten, Open
from .solver import DEMSimulation


def render_animation(sim: DEMSimulation, path: str, fps: int = 25) -> None:
    """Write an animated GIF of the recorded frames, with the particles coloured by speed.

    Args:
        sim: simulation whose frames are drawn.
        path: output file [GIF].
        fps: frames per second of the animation.

    Raises:
        ValueError: if the simulation has no recorded frames.
    """
    frames = sim.frames
    if not frames:
        raise ValueError("The simulation has no recorded frames to animate.")
    geometry = sim.geometry
    x_min, y_min, x_max, y_max = geometry.bounding_box
    margin = 0.02 * max(x_max - x_min, y_max - y_min)
    tops = [float((f.positions[:, 1] + f.radii).max()) for f in frames if f.radii.size]
    y_top = max([y_max] + tops) + margin
    speeds = np.concatenate([f.speeds for f in frames])
    norm = Normalize(0.0, max(0.05, float(np.percentile(speeds, 95))) if speeds.size else 0.05)

    width = x_max - x_min + 2.0 * margin
    height = y_top - y_min + margin
    fig, ax = plt.subplots(figsize=(5.0, 5.0 * height / width))
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_top)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    for start, end, condition in zip(geometry.edge_starts, geometry.edge_ends, geometry.boundary_conditions):
        if isinstance(condition, Open):
            ax.plot(*zip(start, end), color="0.6", lw=1, ls="--")
        elif not isinstance(condition, LoadedPlaten):
            ax.plot(*zip(start, end), color="k", lw=2)
    platen_lines = [ax.plot([], [], color="firebrick", lw=4)[0] for _ in sim.platens]
    title = ax.set_title("")

    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="speed [m/s]", shrink=0.8)
    particles: List[EllipseCollection] = []

    def update(k: int) -> List[Artist]:
        for collection in particles:
            collection.remove()
        particles.clear()
        frame = frames[k]
        diameters = 2.0 * frame.radii
        circles = EllipseCollection(widths=diameters,
                                    heights=diameters,
                                    angles=np.zeros(frame.radii.size),
                                    units="xy",
                                    offsets=frame.positions,
                                    offset_transform=ax.transData,
                                    cmap="viridis",
                                    norm=norm,
                                    edgecolor="k",
                                    linewidth=0.3)
        circles.set_array(frame.speeds)
        ax.add_collection(circles)
        particles.append(circles)
        for line, segment in zip(platen_lines, frame.platens):
            line.set_data(segment[:, 0], segment[:, 1])
        title.set_text(f"{frame.phase}   t = {frame.time:.3f} s   N = {frame.radii.size}")
        return []

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_load_curves(sim: DEMSimulation, path: str) -> None:
    """Plot the applied load and reaction against time and the reaction against displacement of every platen.

    Args:
        sim: simulation whose load curves are plotted.
        path: output image file.

    Raises:
        ValueError: if no load curve has been recorded.
    """
    curves = [curve for curve in sim.load_curves if curve.time]
    if not curves:
        raise ValueError("The simulation has no recorded load curves to plot.")

    fig, axes = plt.subplots(len(curves), 2, figsize=(10.0, 4.0 * len(curves)), squeeze=False)
    for (ax_time, ax_displacement), curve in zip(axes, curves):
        time, applied, displacement, reaction = curve.as_array().T
        ax_time.plot(time, applied, label="applied load F(t)")
        ax_time.plot(time, reaction, "--", label="contact reaction")
        ax_time.set_xlabel("loading time [s]")
        ax_time.set_ylabel("force [N]")
        ax_time.set_title(f"Platen on edge {curve.edge}: load history")
        ax_time.legend()
        ax_time.grid(alpha=0.3)

        ax_displacement.plot(displacement * 1e3, reaction)
        ax_displacement.set_xlabel("platen displacement [mm]")
        ax_displacement.set_ylabel("reaction force [N]")
        ax_displacement.set_title(f"Platen on edge {curve.edge}: load-displacement response")
        ax_displacement.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
