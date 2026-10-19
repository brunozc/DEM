"""Animation and snapshots of the recorded frames, and plots of the load curves of the loaded bodies."""

from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from .boundary import LoadedPlaten, Open
from .solver import DEMSimulation


def _clip_below(vertices: npt.NDArray[np.float64], level: float) -> npt.NDArray[np.float64]:
    """Clip a polygon to the half-plane below a horizontal line.

    Args:
        vertices: polygon vertices, shape (n, 2) [m].
        level: height of the line [m].

    Returns:
        The vertices of the part of the polygon below the line, possibly none.
    """
    clipped = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        start_below = start[1] <= level
        if start_below:
            clipped.append(start)
        if start_below != (end[1] <= level):
            t = (level - start[1]) / (end[1] - start[1])
            clipped.append(start + t * (end - start))
    return np.array(clipped, dtype=np.float64).reshape(-1, 2)


def _current_outline(vertices: npt.NDArray[np.float64], platen_edges: Sequence[Tuple[int, ...]],
                     platen_segments: Sequence[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Return the outline of the domain with every loaded body at its current position.

    Args:
        vertices: vertices of the domain as it was defined, shape (n, 2) [m].
        platen_edges: indices of the edges of every loaded body.
        platen_segments: current start and end points of the edges of every loaded body, shape (k, 2, 2) each [m].

    Returns:
        The vertices, with those of the loaded bodies moved along with them, shape (n, 2) [m].
    """
    outline = np.array(vertices, dtype=np.float64)
    n = outline.shape[0]
    for edges, segments in zip(platen_edges, platen_segments):
        for edge, (start, end) in zip(edges, segments):
            outline[edge] = start
            outline[(edge + 1) % n] = end
    return outline


def _edge_label(edges: Tuple[int, ...]) -> str:
    """Describe the edges of a loaded body for a plot title.

    Args:
        edges: indices of the edges of the body.

    Returns:
        A label such as "edge 3" or "edges 3-8".
    """
    if len(edges) == 1:
        return f"edge {edges[0]}"
    if list(edges) == list(range(edges[0], edges[-1] + 1)):
        return f"edges {edges[0]}-{edges[-1]}"
    return "edges " + ", ".join(str(edge) for edge in edges)


class _FramePainter:
    """Draws the domain once and the particles and loaded bodies of any recorded frame on top of it."""

    def __init__(self, sim: DEMSimulation, width: float = 5.0) -> None:
        """Set up the figure, with axis and colour limits that suit every frame.

        Args:
            sim: simulation whose frames are drawn.
            width: width of the figure [in].

        Raises:
            ValueError: if the simulation has no recorded frames.
        """
        self.frames = sim.frames
        if not self.frames:
            raise ValueError("The simulation has no recorded frames to draw.")
        geometry = sim.geometry
        x_min, y_min, x_max, y_max = geometry.bounding_box
        margin = 0.02 * max(x_max - x_min, y_max - y_min)
        tops = [float((f.positions[:, 1] + f.radii).max()) for f in self.frames if f.radii.size]
        y_top = max([y_max] + tops) + margin
        speeds = np.concatenate([f.speeds for f in self.frames])
        self.norm = Normalize(0.0, max(0.05, float(np.percentile(speeds, 95))) if speeds.size else 0.05)

        figure_width = x_max - x_min + 2.0 * margin
        figure_height = y_top - y_min + margin
        self.figure, self.ax = plt.subplots(figsize=(width, width * figure_height / figure_width))
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_top)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        for start, end, condition in zip(geometry.edge_starts, geometry.edge_ends, geometry.boundary_conditions):
            if isinstance(condition, Open):
                self.ax.plot(*zip(start, end), color="0.6", lw=1, ls="--")
            elif not isinstance(condition, LoadedPlaten):
                self.ax.plot(*zip(start, end), color="k", lw=2)
        self.bodies = [LineCollection([], colors="firebrick", linewidths=4) for _ in sim.platens]
        for body in self.bodies:
            self.ax.add_collection(body)
        self.title = self.ax.set_title("")
        self.vertices = geometry.vertices
        self.platen_edges = [platen.edges for platen in sim.platens]
        self.water = Polygon(np.zeros((0, 2)),
                             closed=True,
                             facecolor="#6baed6",
                             edgecolor="#2171b5",
                             linewidth=1.2,
                             alpha=0.55,
                             zorder=0,
                             visible=False)
        self.ax.add_patch(self.water)

        mappable = ScalarMappable(norm=self.norm, cmap="viridis")
        mappable.set_array([])
        self.figure.colorbar(mappable, ax=self.ax, label="speed [m/s]", shrink=0.8)
        self.particles: List[EllipseCollection] = []

    def draw(self, index: int) -> List[Artist]:
        """Draw one frame.

        Args:
            index: index of the frame in the recorded frames.

        Returns:
            The artists that changed.
        """
        for collection in self.particles:
            collection.remove()
        self.particles.clear()
        frame = self.frames[index]
        diameters = 2.0 * frame.radii
        circles = EllipseCollection(widths=diameters,
                                    heights=diameters,
                                    angles=np.zeros(frame.radii.size),
                                    units="xy",
                                    offsets=frame.positions,
                                    offset_transform=self.ax.transData,
                                    cmap="viridis",
                                    norm=self.norm,
                                    edgecolor="k",
                                    linewidth=0.3)
        circles.set_array(frame.speeds)
        self.ax.add_collection(circles)
        self.particles.append(circles)
        for body, segments in zip(self.bodies, frame.platens):
            body.set_segments(list(segments))
        if frame.water_level is not None:
            # a loaded body that moved into the domain displaces the water, so the outline follows it
            outline = _current_outline(self.vertices, self.platen_edges, frame.platens)
            wet = _clip_below(outline, frame.water_level)
            self.water.set_xy(wet if wet.shape[0] >= 3 else np.zeros((0, 2)))
            self.water.set_visible(wet.shape[0] >= 3)
        self.title.set_text(f"{frame.phase}   t = {frame.time:.3f} s   N = {frame.radii.size}")
        return [circles, self.title, self.water, *self.bodies]


def render_animation(sim: DEMSimulation, path: str, fps: int = 25, every: int = 1) -> None:
    """Write an animated GIF of the recorded frames, with the particles coloured by speed.

    Args:
        sim: simulation whose frames are drawn.
        path: output file [GIF].
        fps: frames per second of the animation.
        every: draw every n-th recorded frame, which keeps the animation of a long run short.

    Raises:
        ValueError: if every is not positive.
        ValueError: if the simulation has no recorded frames.
    """
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}.")
    painter = _FramePainter(sim)
    animation = FuncAnimation(painter.figure,
                              painter.draw,
                              frames=range(0, len(painter.frames), every),
                              interval=1000 // fps,
                              blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(painter.figure)


def render_frame(sim: DEMSimulation, path: str, index: int = -1, width: float = 5.0) -> None:
    """Save a snapshot of one recorded frame, coloured on the same speed scale as the animation.

    Args:
        sim: simulation whose frame is drawn.
        path: output image file.
        index: index of the frame; negative values count from the last frame.
        width: width of the figure [in].

    Raises:
        ValueError: if the simulation has no recorded frames.
    """
    painter = _FramePainter(sim, width)
    painter.draw(index)
    painter.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(painter.figure)


def plot_load_curves(sim: DEMSimulation, path: str) -> None:
    """Plot the applied load and reaction against time and the reaction against displacement of every loaded body.

    For a body with self weight the load plus the weight is drawn as well, since that is what the reaction balances.

    Args:
        sim: simulation whose load curves are plotted.
        path: output image file.

    Raises:
        ValueError: if no load curve has been recorded.
    """
    curves = [curve for curve in sim.load_curves if curve.time]
    if not curves:
        raise ValueError("The simulation has no recorded load curves to plot.")

    bodies = {body.edges: body for body in sim.platens}
    gravity = np.array([0.0, -sim.settings.gravity])
    fig, axes = plt.subplots(len(curves), 2, figsize=(10.0, 4.0 * len(curves)), squeeze=False)
    for (ax_time, ax_displacement), curve in zip(axes, curves):
        time, applied, displacement, reaction = curve.as_array().T
        label = _edge_label(curve.edges)
        body = bodies[curve.edges]
        ax_time.plot(time, applied, label="applied load F(t)")
        if body.spec.self_weight:
            weight = body.spec.mass * float(gravity @ body.direction)
            ax_time.plot(time, applied + weight, ":", color="0.3", label="applied load + weight")
        ax_time.plot(time, reaction, "--", label="contact reaction")
        ax_time.set_xlabel("loading time [s]")
        ax_time.set_ylabel("force [N]")
        ax_time.set_title(f"Body on {label}: load history")
        ax_time.legend()
        ax_time.grid(alpha=0.3)

        ax_displacement.plot(displacement * 1e3, reaction)
        ax_displacement.set_xlabel("displacement [mm]")
        ax_displacement.set_ylabel("reaction force [N]")
        ax_displacement.set_title(f"Body on {label}: load-displacement response")
        ax_displacement.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
