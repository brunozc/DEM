"""Animation and snapshots of the recorded frames, and plots of the load curves of the loaded bodies.

A 2D simulation is drawn in its plane. A 3D simulation is drawn in an oblique view of the whole domain, or as a
vertical section z = const through it, which shows the particles that the plane cuts as circles and looks like a 2D
simulation; a 3D domain extruded from a polygon has that polygon as its section.
"""

import io
import math
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection, LineCollection, PathCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
# mplot3d ships no type information
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]
from PIL import Image

from .boundary import LoadedPlaten, Open
from .geometry import Geometry
from .geometry3d import Geometry3D
from .solver import DEMSimulation, Frame

WATER_FACE = "#6baed6"
WATER_EDGE = "#2171b5"

# viewing angles of the oblique 3D view [deg]: the camera looks down at the front-right of the domain
VIEW_ELEVATION = 20.0
VIEW_AZIMUTH = 30.0


def _clip_below(vertices: npt.NDArray[np.float64], level: float) -> npt.NDArray[np.float64]:
    """Clip a polygon to the half-plane (or half-space, in 3D) below a horizontal line (or plane).

    Args:
        vertices: polygon vertices, shape (n, 2) or (n, 3), with y as the height [m].
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
    return np.array(clipped, dtype=np.float64).reshape(-1, vertices.shape[1])


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


def _facet_label(indices: Tuple[int, ...], name: str) -> str:
    """Describe the facets of a loaded body for a plot title.

    Args:
        indices: indices of the facets of the body.
        name: name of one facet, "edge" or "face".

    Returns:
        A label such as "edge 3" or "faces 3-8".
    """
    if len(indices) == 1:
        return f"{name} {indices[0]}"
    if list(indices) == list(range(indices[0], indices[-1] + 1)):
        return f"{name}s {indices[0]}-{indices[-1]}"
    return f"{name}s " + ", ".join(str(index) for index in indices)


def _speed_norm(frames: Sequence[Frame]) -> Normalize:
    """Colour scale of the particle speeds that suits every frame, up to the 95th percentile of all speeds.

    Args:
        frames: recorded frames.

    Returns:
        The colour normalisation [m/s].
    """
    speeds = np.concatenate([f.speeds for f in frames])
    return Normalize(0.0, max(0.05, float(np.percentile(speeds, 95))) if speeds.size else 0.05)


def _plane_crossings(polygon: npt.NDArray[np.float64], axis: int,
                     value: float) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Segments along which a planar polygon crosses the plane on which one coordinate has a given value.

    A vertex on the plane counts as lying above it, so that every crossing is found once.

    Args:
        polygon: vertices of the polygon in order around it, shape (m, 3); repeated vertices are allowed [m].
        axis: coordinate that is constant on the plane, 0 for x, 1 for y and 2 for z.
        value: value of that coordinate on the plane [m].

    Returns:
        The start and end point of each segment of the polygon that lies in the plane [m].
    """
    heights = polygon[:, axis] - value
    above = heights >= 0.0
    following = np.roll(np.arange(polygon.shape[0]), -1)
    points = []
    for k, m in enumerate(following):
        if above[k] != above[m]:
            t = heights[k] / (heights[k] - heights[m])
            points.append(polygon[k] + t * (polygon[m] - polygon[k]))
    if len(points) < 2:
        return []
    # a non-convex polygon can cross the plane several times, in pairs along the line where the two planes meet
    normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
    line = np.cross(normal, np.eye(3)[axis])
    ordered = sorted(points, key=lambda point: float(point @ line))
    return [(ordered[k], ordered[k + 1]) for k in range(0, len(ordered) - 1, 2)]


def _chain_loops(segments: Sequence[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
                 tolerance: float) -> List[npt.NDArray[np.float64]]:
    """Join segments that share end points into closed loops.

    Args:
        segments: start and end point of each segment, in any order and orientation [m].
        tolerance: distance below which two end points are the same point [m].

    Returns:
        The vertices of every closed loop; chains that do not close are dropped.
    """
    remaining = list(segments)
    loops = []
    while remaining:
        start, end = remaining.pop()
        loop = [start, end]
        while True:
            if len(loop) > 2 and float(np.linalg.norm(loop[-1] - loop[0])) <= tolerance:
                loops.append(np.array(loop[:-1]))
                break
            for k, (a, b) in enumerate(remaining):
                if float(np.linalg.norm(a - loop[-1])) <= tolerance:
                    loop.append(b)
                    break
                if float(np.linalg.norm(b - loop[-1])) <= tolerance:
                    loop.append(a)
                    break
            else:
                break
            remaining.pop(k)
    return loops


def _current_polyhedron(geometry: Geometry3D, bodies: Sequence[Tuple[int, ...]],
                        platens: Sequence[npt.NDArray[np.float64]]) -> List[npt.NDArray[np.float64]]:
    """Return the faces of a polyhedral domain with every loaded body at its current position.

    The vertices of a body move with it, and so do the vertices that the faces next to it share with it, as the
    outline of a polygon follows its loaded edges.

    Args:
        geometry: polyhedral domain.
        bodies: indices of the faces of every loaded body.
        platens: current vertices of the faces of every loaded body, shape (k, m, 3) each [m].

    Returns:
        The vertices of each face [m].
    """
    vertices = geometry.vertices
    rest = geometry.vertices
    for faces, current in zip(bodies, platens):
        shift = current[0, 0] - rest[geometry.faces[faces[0]][0]]
        for face in faces:
            vertices[list(geometry.faces[face])] = rest[list(geometry.faces[face])] + shift
    return [vertices[list(face)] for face in geometry.faces]


class _FramePainter:
    """Draws the domain once and the particles and loaded bodies of any recorded frame on top of it."""

    def __init__(self, sim: DEMSimulation, width: float = 5.0) -> None:
        """Set up the figure, with axis and colour limits that suit every frame.

        Args:
            sim: simulation whose frames are drawn.
            width: width of the figure [in].

        Raises:
            ValueError: if the simulation has no recorded frames.
            ValueError: if the simulation is not two-dimensional.
        """
        self.frames = sim.frames
        if not self.frames:
            raise ValueError("The simulation has no recorded frames to draw.")
        geometry = sim.geometry
        if not isinstance(geometry, Geometry):
            raise ValueError("A plane view can only be drawn of a two-dimensional simulation.")
        x_min, y_min, x_max, y_max = geometry.bounding_box
        margin = 0.02 * max(x_max - x_min, y_max - y_min)
        tops = [float((f.positions[:, 1] + f.radii).max()) for f in self.frames if f.radii.size]
        y_top = max([y_max] + tops) + margin
        self.norm = _speed_norm(self.frames)

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


class _SectionPainter:
    """Draws a vertical section z = const through a 3D simulation, with the particles that the plane cuts."""

    def __init__(self, sim: DEMSimulation, section: float, width: float = 5.0) -> None:
        """Set up the figure, with axis and colour limits that suit every frame.

        Args:
            sim: 3D simulation whose frames are drawn.
            section: z coordinate of the section plane [m].
            width: width of the figure [in].

        Raises:
            ValueError: if the simulation has no recorded frames.
            ValueError: if the simulation is not three-dimensional.
            ValueError: if the section plane does not cut the domain.
        """
        self.frames = sim.frames
        if not self.frames:
            raise ValueError("The simulation has no recorded frames to draw.")
        geometry = sim.geometry
        if not isinstance(geometry, Geometry3D):
            raise ValueError("A section can only be drawn of a three-dimensional simulation.")
        low, high = geometry.bounds
        if not low[2] < section < high[2]:
            raise ValueError(f"The section z = {section} does not cut the domain, which spans z = {low[2]:.4g} to "
                             f"{high[2]:.4g}.")
        self.geometry = geometry
        self.section = section
        self.tolerance = 1.0e-9 * float(np.max(high - low))
        x_min, y_min, x_max, y_max = float(low[0]), float(low[1]), float(high[0]), float(high[1])
        margin = 0.02 * max(x_max - x_min, y_max - y_min)
        tops = [float((f.positions[:, 1] + f.radii).max()) for f in self.frames if f.radii.size]
        y_top = max([y_max] + tops) + margin
        self.norm = _speed_norm(self.frames)

        figure_width = x_max - x_min + 2.0 * margin
        figure_height = y_top - y_min + margin
        self.figure, self.ax = plt.subplots(figsize=(width, width * figure_height / figure_width))
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_top)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        vertices = geometry.vertices
        for face, condition in zip(geometry.faces, geometry.boundary_conditions):
            for start, end in _plane_crossings(vertices[list(face)], 2, section):
                if isinstance(condition, Open):
                    self.ax.plot(*zip(start[:2], end[:2]), color="0.6", lw=1, ls="--")
                elif not isinstance(condition, LoadedPlaten):
                    self.ax.plot(*zip(start[:2], end[:2]), color="k", lw=2)
        self.bodies = [LineCollection([], colors="firebrick", linewidths=4) for _ in sim.platens]
        for body in self.bodies:
            self.ax.add_collection(body)
        self.platen_faces = [platen.edges for platen in sim.platens]
        self.title = self.ax.set_title("")
        # the section of a non-convex domain can fall apart into several pieces, each with its own water
        self.water: List[Polygon] = []

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
        # a sphere at a distance dz from the plane cuts it in a circle of radius sqrt(R^2 - dz^2)
        depth = frame.positions[:, 2] - self.section
        cut = np.abs(depth) < frame.radii
        diameters = 2.0 * np.sqrt(frame.radii[cut]**2 - depth[cut]**2)
        circles = EllipseCollection(widths=diameters,
                                    heights=diameters,
                                    angles=np.zeros(diameters.size),
                                    units="xy",
                                    offsets=frame.positions[cut, :2],
                                    offset_transform=self.ax.transData,
                                    cmap="viridis",
                                    norm=self.norm,
                                    edgecolor="k",
                                    linewidth=0.3)
        circles.set_array(frame.speeds[cut])
        self.ax.add_collection(circles)
        self.particles.append(circles)
        for body, current in zip(self.bodies, frame.platens):
            body.set_segments([(start[:2], end[:2]) for face in current
                               for start, end in _plane_crossings(face, 2, self.section)])
        for patch in self.water:
            patch.remove()
        self.water.clear()
        if frame.water_level is not None:
            # a loaded body that moved into the domain displaces the water, so the section follows it
            polygons = _current_polyhedron(self.geometry, self.platen_faces, frame.platens)
            crossings = [crossing for face in polygons for crossing in _plane_crossings(face, 2, self.section)]
            for loop in _chain_loops(crossings, self.tolerance):
                wet = _clip_below(loop[:, :2], frame.water_level)
                if wet.shape[0] >= 3:
                    patch = Polygon(wet,
                                    closed=True,
                                    facecolor=WATER_FACE,
                                    edgecolor=WATER_EDGE,
                                    linewidth=1.2,
                                    alpha=0.55,
                                    zorder=0)
                    self.ax.add_patch(patch)
                    self.water.append(patch)
        self.title.set_text(f"{frame.phase}   t = {frame.time:.3f} s   N = {frame.radii.size}   "
                            f"z = {self.section:.3g} m")
        return [circles, self.title, *self.water, *self.bodies]


class _FramePainter3D:
    """Draws an oblique view of a 3D simulation: the domain once, and the particles and loaded bodies of any frame."""

    def __init__(self, sim: DEMSimulation, width: float = 5.0) -> None:
        """Set up the figure, with axis and colour limits that suit every frame.

        Args:
            sim: 3D simulation whose frames are drawn.
            width: width of the figure [in].

        Raises:
            ValueError: if the simulation has no recorded frames.
            ValueError: if the simulation is not three-dimensional.
        """
        self.frames = sim.frames
        if not self.frames:
            raise ValueError("The simulation has no recorded frames to draw.")
        geometry = sim.geometry
        if not isinstance(geometry, Geometry3D):
            raise ValueError("An oblique view can only be drawn of a three-dimensional simulation.")
        self.geometry = geometry
        low, high = geometry.bounds
        tops = [float((f.positions[:, 1] + f.radii).max()) for f in self.frames if f.radii.size]
        high[1] = max([float(high[1])] + tops)
        self.tolerance = 1.0e-9 * float(np.max(high - low))
        self.norm = _speed_norm(self.frames)

        self.figure = plt.figure(figsize=(width, width))
        # the artists are drawn in a fixed order, so that the loaded bodies are never hidden behind the particles
        self.ax: Axes3D = self.figure.add_subplot(projection="3d", computed_zorder=False)
        self.ax.set_proj_type("ortho")
        # seen from the front (high z), the right (high x) and above, with y up
        self.ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH, vertical_axis="y")
        self.ax.set_xlim(low[0], high[0])
        self.ax.set_ylim(low[1], high[1])
        self.ax.set_zlim(low[2], high[2])
        # equal scales along the three axes, so that spheres look round
        self.ax.set_box_aspect(tuple(float(extent) for extent in high - low), zoom=0.9)
        # a short axis gets fewer ticks, so that its labels do not run into each other
        largest = float(np.max(high - low))
        for k, (axis, name) in enumerate(((self.ax.xaxis, "x"), (self.ax.yaxis, "y"), (self.ax.zaxis, "z"))):
            axis.set_major_locator(MaxNLocator(max(2, round(4.0 * float(high[k] - low[k]) / largest))))
            axis.set_label_text(f"{name} [m]")

        vertices = geometry.vertices
        walls = [
            vertices[list(face)] for face, condition in zip(geometry.faces, geometry.boundary_conditions)
            if not isinstance(condition, (Open, LoadedPlaten))
        ]
        openings = [
            vertices[list(face)] for face, condition in zip(geometry.faces, geometry.boundary_conditions)
            if isinstance(condition, Open)
        ]
        # the axis limits suit every frame, so the collections must not rescale them
        self.ax.add_collection3d(Poly3DCollection(walls,
                                                  facecolors="0.85",
                                                  edgecolors="k",
                                                  linewidths=0.6,
                                                  alpha=0.08,
                                                  zorder=1),
                                 autolim=False)
        outlines = [np.vstack((face, face[:1])) for face in openings]
        self.ax.add_collection3d(Line3DCollection(outlines, colors="0.5", linewidths=0.8, linestyles="--", zorder=2),
                                 autolim=False)
        self.water = Poly3DCollection([], facecolors=WATER_FACE, edgecolors="none", alpha=0.18, zorder=3)
        self.surface = Poly3DCollection([],
                                        facecolors=WATER_FACE,
                                        edgecolors=WATER_EDGE,
                                        linewidths=1.0,
                                        alpha=0.3,
                                        zorder=3)
        self.ax.add_collection3d(self.water, autolim=False)
        self.ax.add_collection3d(self.surface, autolim=False)
        self.bodies = [
            Poly3DCollection([], facecolors="firebrick", edgecolors="darkred", linewidths=1.0, alpha=0.7, zorder=5)
            for _ in sim.platens
        ]
        for body in self.bodies:
            self.ax.add_collection3d(body, autolim=False)
        self.platen_faces = [platen.edges for platen in sim.platens]
        self.title = self.ax.set_title("")

        mappable = ScalarMappable(norm=self.norm, cmap="viridis")
        mappable.set_array([])
        self.figure.colorbar(mappable, ax=self.ax, label="speed [m/s]", shrink=0.55, pad=0.08)
        # the layout is final once drawn, which fixes the scale of the projection
        self.figure.canvas.draw()
        self.points_per_metre = self.__points_per_metre()
        self.particles: List[PathCollection] = []

    def __display(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Position of points on the figure, through the current projection of the axes.

        Args:
            points: points, shape (n, 3) [m].

        Returns:
            Their display coordinates, shape (n, 2) [pixels].
        """
        homogeneous = np.column_stack((points, np.ones(points.shape[0]))) @ self.ax.get_proj().T
        return np.asarray(self.ax.transData.transform(homogeneous[:, :2] / homogeneous[:, 3:4]))

    def __points_per_metre(self) -> float:
        """Length on the figure of one metre seen at right angles to the line of sight [points].

        The orthographic projection with equal axis scales maps every direction at right angles to the line of
        sight with the same scale s, so the squared lengths of the images of the three unit vectors add up to 2 s^2.

        Returns:
            The scale [points/m].
        """
        pixels = self.__display(np.vstack((np.zeros(3), np.eye(3))))
        squared = float(np.sum((pixels[1:] - pixels[0])**2))
        return float(np.sqrt(0.5 * squared)) * 72.0 / self.figure.dpi

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
        x, y, z = frame.positions.T
        spheres = self.ax.scatter(x,
                                  y,
                                  z,
                                  s=(2.0 * frame.radii * self.points_per_metre)**2,
                                  c=frame.speeds,
                                  cmap="viridis",
                                  norm=self.norm,
                                  edgecolors="k",
                                  linewidths=0.2,
                                  depthshade=True,
                                  zorder=4)
        self.particles.append(spheres)
        for body, current in zip(self.bodies, frame.platens):
            body.set_verts(list(current))
        wetted: List[npt.NDArray[np.float64]] = []
        surface: List[npt.NDArray[np.float64]] = []
        if frame.water_level is not None:
            # a loaded body that moved into the domain displaces the water, so the wetted faces follow it
            polygons = _current_polyhedron(self.geometry, self.platen_faces, frame.platens)
            wetted = [wet for wet in (_clip_below(face, frame.water_level) for face in polygons) if wet.shape[0] >= 3]
            crossings = [crossing for face in polygons for crossing in _plane_crossings(face, 1, frame.water_level)]
            surface = _chain_loops(crossings, self.tolerance)
        self.water.set_verts(wetted)
        self.surface.set_verts(surface)
        self.title.set_text(f"{frame.phase}   t = {frame.time:.3f} s   N = {frame.radii.size}")
        return [spheres, self.title, self.water, self.surface, *self.bodies]


Painter = Union[_FramePainter, _SectionPainter, _FramePainter3D]


def _save_cropped_gif(painter: _FramePainter3D, indices: Sequence[int], path: str, fps: int) -> None:
    """Write frames as an animated GIF, each cropped to the part of the figure that the first one draws on.

    A 3D view is drawn in a square of the figure whatever the shape of the domain, so a slender or a flat domain
    leaves blank bands around it, which an animation cannot trim frame by frame as a snapshot can.

    Args:
        painter: painter of the frames.
        indices: indices of the frames to draw.
        path: output file [GIF].
        fps: frames per second of the animation.
    """
    figure = painter.figure
    dpi = figure.dpi
    width, height = (int(size * dpi) for size in figure.get_size_inches())
    images: List[Image.Image] = []
    box: Optional[Tuple[int, int, int, int]] = None
    for index in indices:
        painter.draw(index)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="rgba", dpi=dpi)
        if box is None:
            tight = figure.get_tightbbox().padded(0.05)
            # the box in pixels, from the top left corner of the image
            box = (max(0, int(tight.x0 * dpi)), max(0, int(height - tight.y1 * dpi)),
                   min(width, math.ceil(tight.x1 * dpi)), min(height, math.ceil(height - tight.y0 * dpi)))
        image = Image.frombuffer("RGBA", (width, height), buffer.getvalue(), "raw", "RGBA", 0, 1)
        images.append(image.crop(box))
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def _painter(sim: DEMSimulation, width: float, section: Optional[float]) -> Painter:
    """Choose how to draw the frames of a simulation.

    Args:
        sim: simulation whose frames are drawn.
        width: width of the figure [in].
        section: z coordinate of a vertical section through a 3D simulation [m]; None draws a 3D simulation in an
            oblique view.

    Returns:
        A painter for the frames: in the plane of a 2D simulation, or of a section or an oblique view of a 3D one.

    Raises:
        ValueError: if a section is asked for a 2D simulation.
        ValueError: if the simulation has no recorded frames.
        ValueError: if the section plane does not cut the domain.
    """
    if isinstance(sim.geometry, Geometry):
        if section is not None:
            raise ValueError("A section can only be drawn of a three-dimensional simulation.")
        return _FramePainter(sim, width)
    if section is not None:
        return _SectionPainter(sim, section, width)
    return _FramePainter3D(sim, width)


def render_animation(sim: DEMSimulation,
                     path: str,
                     fps: int = 25,
                     every: int = 1,
                     section: Optional[float] = None) -> None:
    """Write an animated GIF of the recorded frames, with the particles coloured by speed.

    Args:
        sim: simulation whose frames are drawn.
        path: output file [GIF].
        fps: frames per second of the animation.
        every: draw every n-th recorded frame, which keeps the animation of a long run short.
        section: z coordinate of a vertical section through a 3D simulation [m]; None draws a 3D simulation in an
            oblique view. A 2D simulation is always drawn in its plane.

    Raises:
        ValueError: if every is not positive.
        ValueError: if a section is asked for a 2D simulation.
        ValueError: if the simulation has no recorded frames.
        ValueError: if the section plane does not cut the domain.
    """
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}.")
    painter = _painter(sim, 5.0, section)
    indices = range(0, len(painter.frames), every)
    if isinstance(painter, _FramePainter3D):
        _save_cropped_gif(painter, indices, path, fps)
    else:
        animation = FuncAnimation(painter.figure, painter.draw, frames=indices, interval=1000 // fps, blit=False)
        animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(painter.figure)


def render_frame(sim: DEMSimulation,
                 path: str,
                 index: int = -1,
                 width: float = 5.0,
                 section: Optional[float] = None) -> None:
    """Save a snapshot of one recorded frame, coloured on the same speed scale as the animation.

    Args:
        sim: simulation whose frame is drawn.
        path: output image file.
        index: index of the frame; negative values count from the last frame.
        width: width of the figure [in].
        section: z coordinate of a vertical section through a 3D simulation [m]; None draws a 3D simulation in an
            oblique view. A 2D simulation is always drawn in its plane.

    Raises:
        ValueError: if a section is asked for a 2D simulation.
        ValueError: if the simulation has no recorded frames.
        ValueError: if the section plane does not cut the domain.
    """
    painter = _painter(sim, width, section)
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
    gravity = np.zeros(sim.dimension)
    gravity[1] = -sim.settings.gravity
    fig, axes = plt.subplots(len(curves), 2, figsize=(10.0, 4.0 * len(curves)), squeeze=False)
    for (ax_time, ax_displacement), curve in zip(axes, curves):
        time, applied, displacement, reaction = curve.as_array().T
        label = _facet_label(curve.edges, sim.geometry.facet_name)
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
