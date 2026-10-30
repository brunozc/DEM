"""Export of the state of a simulation to VTK files, to be viewed in ParaView.

A ``VTKWriter`` writes the state of a simulation as five time series of VTK PolyData files (``.vtp``), each with a
ParaView data collection (``.pvd``) that lists its files with their simulation time:

* ``particles``: the centre of every particle, as a point, with its radius, velocity, angular velocity, coordination
  number and stress, whether it is a sphere, and, when there is water, its porosity, the force of the water on it and,
  for coupled water, the excess pore pressure. ParaView draws the points of the spheres as spheres of their true size
  with its Point Gaussian representation, scaled by the radius.
* ``shapes``: the outline of every polygonal particle (2D), or the faces of every polyhedral particle (3D), as
  polygons in space, with the index, velocity, angular velocity, coordination number and mean stress of the particle;
  empty when every particle is a sphere.
* ``contacts``: a line for every contact, between the centres of two particles or from the centre of a particle to
  where it touches a facet, with its normal force, tangential force and overlap. Drawn as tubes that grow thicker
  with the normal force, the strongest contacts are the force chains that carry the load through the packing.
* ``boundary``: the facets of the domain, edges in 2D and faces in 3D, at their current position, so that loaded
  bodies move, with their boundary condition and the force and pressure of the particles on them.
* ``water``: the free surface of the water across the domain, empty while the particles are dry.

The writer also puts a ParaView script, ``paraview_setup.py``, next to the collections. Run in ParaView, it opens the
five series and shows them in two views whose cameras move together: the particles coloured by their speed, and the
force chains with the facets coloured by their pressure.

A 2D simulation is written in the plane z = 0, with its rotations about the z axis. The stress of a particle is its
mean stress, from the forces of its contacts (the Love-Weber formula), positive in compression as in soil mechanics.
The files are in the XML format of VTK, with the arrays as raw binary data appended to them, and NumPy writes them.
"""

import os
import posixpath
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .boundary import PLATEN_CONDITIONS, Open
from .geometry3d import Geometry3D
from .polytopes import frame_polygons
from .solver import ContactList, DEMSimulation
from .vectors import norm, scatter_add
from .water import evaluate_at

# values of a data array: real numbers, or integers such as counts and indices
DataArray = Union[npt.NDArray[np.float64], npt.NDArray[np.int64]]

# the series written at every export, each to a subfolder and a collection of its own name
SERIES = ("particles", "contacts", "boundary", "water", "shapes")

# name of the ParaView script written next to the collections
PARAVIEW_SCRIPT = "paraview_setup.py"

# code of each boundary condition in the condition array of the boundary
OPEN_CODE = 0
WALL_CODE = 1
PLATEN_CODE = 2

_PARAVIEW_SCRIPT_TEXT = """\
# Show a DEMcookies export in ParaView. DEMcookies.vtk_export.VTKWriter writes this script next to the files it opens.
#
# Run it in the ParaView Python shell (View > Python Shell > Run Script), or start ParaView with
#
#     paraview --script=paraview_setup.py
#
# Two views side by side, whose cameras move together, show the particles, spheres of their true size and polygons
# or polyhedra, coloured by their speed, and the force chains, the contacts drawn as tubes that grow thicker with
# their normal force, with the facets coloured by the pressure of the particles on them. Open facets are not drawn. The
# views open at the end of the run, and every colour map spans the values of the time shown.

import os

from paraview.simple import (AddCameraLink, AssignViewToLayout, CellDatatoPointData, ColorBy, CreateView,
                             GetActiveViewOrCreate, GetAnimationScene, GetColorTransferFunction, GetLayout, PVDReader,
                             Render, Show, Threshold, Tube)

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# direction from the domain to the camera of a 3D view: from the front right and above, as in DEMcookies.plotting
VIEW_DIRECTION = (0.47, 0.34, 0.81)
# radius of the tube of the weakest and of the strongest contact, as a fraction of the largest particle radius
TUBE_RADIUS = 0.05
TUBE_RADIUS_FACTOR = 10.0
WATER_COLOUR = [0.42, 0.68, 0.84]
WALL_COLOUR = [0.6, 0.6, 0.6]
BODY_COLOUR = [0.7, 0.13, 0.13]
# codes of the condition array of the boundary: 0 is an open facet
WALL, BODY = 1, 2


def open_series(name):
    return PVDReader(registrationName=name, FileName=os.path.join(DIRECTORY, name + ".pvd"))


def facets(name, code):
    return Threshold(registrationName=name, Input=boundary, Scalars=["CELLS", "condition"], LowerThreshold=code,
                     UpperThreshold=code)


def paint(display, colour, opacity):
    display.ColorArrayName = ["POINTS", ""]
    display.DiffuseColor = colour
    display.AmbientColor = colour
    display.Opacity = opacity
    display.LineWidth = 3.0


def colour_by(display, view, array, component=None):
    ColorBy(display, (array[0], array[1]) if component is None else (array[0], array[1], component))
    # the range follows the time shown, so the deposition does not set it for the loading
    GetColorTransferFunction(array[1]).AutomaticRescaleRangeMode = "Clamp and update every timestep"
    display.RescaleTransferFunctionToDataRange(False, True)
    display.SetScalarBarVisibility(view, True)


particles = open_series("particles")
contacts = open_series("contacts")
boundary = open_series("boundary")
water = open_series("water")
shapes = open_series("shapes")
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.GoToLast()
walls = facets("walls", WALL)
bodies = facets("loaded bodies", BODY)
solid = Threshold(registrationName="solid facets", Input=boundary, Scalars=["CELLS", "condition"],
                  LowerThreshold=WALL, UpperThreshold=BODY)
chains = Tube(registrationName="force chains",
              Input=CellDatatoPointData(registrationName="contact forces", Input=contacts))
particles.UpdatePipeline(scene.AnimationTime)
chains.Scalars = ["POINTS", "normal_force"]
chains.VaryRadius = "By Scalar"
chains.Radius = TUBE_RADIUS * max(particles.PointData["radius"].GetRange()[1], 0.0)
chains.RadiusFactor = TUBE_RADIUS_FACTOR
boundary.UpdatePipeline(scene.AnimationTime)
x_min, x_max, y_min, y_max, z_min, z_max = boundary.GetDataInformation().GetBounds()
size = max(x_max - x_min, y_max - y_min, z_max - z_min)
flat = z_max - z_min <= 1.0e-9 * size

left = GetActiveViewOrCreate("RenderView")
layout = GetLayout(left)
layout.SplitHorizontal(0, 0.5)
right = CreateView("RenderView")
AssignViewToLayout(view=right, layout=layout, hint=2)

# the points of polygons and polyhedra are left out, since their shapes are drawn instead
round_ones = Threshold(registrationName="spheres", Input=particles, Scalars=["POINTS", "sphere"], LowerThreshold=1,
                       UpperThreshold=1)
spheres = Show(round_ones, left)
spheres.SetRepresentationType("Point Gaussian")
spheres.ShaderPreset = "Sphere"
spheres.ScaleByArray = 1
spheres.UseScaleFunction = 0
spheres.SetScaleArray = ["POINTS", "radius"]
spheres.GaussianRadius = 1.0
colour_by(spheres, left, ("POINTS", "velocity"), "Magnitude")
colour_by(Show(shapes, left), left, ("CELLS", "velocity"), "Magnitude")
colour_by(Show(chains, right), right, ("POINTS", "normal_force"))
# facets are see-through in 3D, so that they do not hide the particles
paint(Show(walls, left), WALL_COLOUR, 1.0 if flat else 0.25)
paint(Show(bodies, left), BODY_COLOUR, 1.0 if flat else 0.7)
# one display for all the solid facets, so that the colour map spans the pressure on all of them
pressure = Show(solid, right)
pressure.LineWidth = 3.0
pressure.Opacity = 1.0 if flat else 0.35
colour_by(pressure, right, ("CELLS", "pressure"))
for view in (left, right):
    paint(Show(water, view), WATER_COLOUR, 1.0 if flat else 0.35)

centre = [0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)]
direction = (0.0, 0.0, 1.0) if flat else VIEW_DIRECTION
for view in (left, right):
    # the first render frames all the data, which particles that left the domain would spread out
    Render(view)
    view.CameraFocalPoint = centre
    view.CameraPosition = [c + 3.0 * size * u for c, u in zip(centre, direction)]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    if flat:
        view.InteractionMode = "2D"
    view.ResetCamera(x_min, x_max, y_min, y_max, z_min, z_max)
AddCameraLink(left, right, "DEMcookies cameras")
Render(left)
Render(right)
"""


@dataclass(frozen=True, eq=False)
class _PolyData:
    """Points, cells of one kind and data arrays of a VTK PolyData data set.

    Attributes:
        points: coordinates of the points, shape (p, 3) [m].
        cells: kind of the cells: "Verts", "Lines" or "Polys".
        connectivity: points of every cell, one cell after the other.
        offsets: end of every cell in the connectivity.
        point_data: arrays of values at the points, by name.
        cell_data: arrays of values on the cells, by name.
    """

    points: npt.NDArray[np.float64]
    cells: str
    connectivity: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    point_data: Dict[str, DataArray]
    cell_data: Dict[str, DataArray]


class VTKWriter:
    """Writes the state of a simulation to VTK files, as time series that ParaView opens.

    Attach a writer to a simulation with ``sim.add_observer(writer)`` to export its state every ``save_every_n_steps``
    time steps, or call ``writer.write(sim)`` to export its current state, e.g. of a simulation loaded from a
    checkpoint. Every export adds a file to each series and rewrites the collections, so the series can be opened
    while the simulation runs, and stay readable if it stops. A writer pickled with the simulation it observes, e.g.
    in a checkpoint, carries on with its series when the simulation is resumed; a new writer starts the series afresh,
    overwriting the files of an earlier one in the same folder.

    Attributes:
        directory: folder of the collections, of a subfolder per series and of the ParaView script.
        save_every_n_steps: number of time steps between two exports of an observing writer, or None for a writer
            that only exports when ``write`` is called.
        times: simulation time of every export [s].
    """

    def __init__(self, directory: str, save_every_n_steps: Optional[int] = None) -> None:
        """Set the writer up; the folders and the ParaView script are written with the first export.

        Args:
            directory: folder to write to; it is created if needed.
            save_every_n_steps: number of time steps between two exports when the writer observes a simulation; the
                writer exports after the steps whose number, counted from the start of the simulation, is a multiple
                of it. None makes a writer that only exports when ``write`` is called.

        Raises:
            ValueError: if save_every_n_steps is not positive.
        """
        if save_every_n_steps is not None and save_every_n_steps < 1:
            raise ValueError(f"save_every_n_steps must be at least 1, got {save_every_n_steps}.")
        self.directory = directory
        self.save_every_n_steps = save_every_n_steps
        self.times: List[float] = []

    def __call__(self, sim: DEMSimulation) -> None:
        """Export the state of a simulation after a time step, if the number of steps it has taken is due.

        Args:
            sim: the simulation, which has just taken a step.

        Raises:
            ValueError: if the writer has no interval between exports.
        """
        if self.save_every_n_steps is None:
            raise ValueError("A VTKWriter needs save_every_n_steps to observe a simulation.")
        if sim.steps_taken % self.save_every_n_steps == 0:
            self.write(sim)

    def write(self, sim: DEMSimulation) -> None:
        """Export the current state of a simulation.

        Args:
            sim: the simulation.
        """
        # the folders are made at every export, so that a resumed run makes them again if they were removed
        for name in SERIES:
            os.makedirs(os.path.join(self.directory, name), exist_ok=True)
        script = os.path.join(self.directory, PARAVIEW_SCRIPT)
        if not self.times or not os.path.exists(script):
            with open(script, "w", encoding="utf-8") as stream:
                stream.write(_PARAVIEW_SCRIPT_TEXT)
        pairs, walls = sim.contacts()
        index = len(self.times)
        coordination, stress = _particle_stress(sim, pairs, walls)
        data = {
            "particles": _particle_data(sim, coordination, stress),
            "contacts": _contact_data(sim, pairs, walls),
            "boundary": _boundary_data(sim, walls),
            "water": _water_data(sim),
            "shapes": _shape_data(sim, coordination, stress),
        }
        for name in SERIES:
            _write_polydata(os.path.join(self.directory, _series_file(name, index)), data[name])
        self.times.append(float(sim.time))
        for name in SERIES:
            entries = [(time, _series_file(name, k)) for k, time in enumerate(self.times)]
            _write_collection(os.path.join(self.directory, name + ".pvd"), entries)


def _series_file(name: str, index: int) -> str:
    """Path of a file of a series, relative to the folder of the writer, with forward slashes as a collection needs.

    Args:
        name: name of the series.
        index: number of the export.

    Returns:
        The path.
    """
    return posixpath.join(name, f"{name}_{index:06d}.vtp")


def _in_space(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Give vectors three components, the third zero for vectors in the plane z = 0 of a 2D simulation.

    Args:
        vectors: vectors, shape (k, 2) or (k, 3).

    Returns:
        The vectors, shape (k, 3).
    """
    if vectors.shape[1] == 3:
        return vectors
    return np.column_stack((vectors, np.zeros(vectors.shape[0])))


def _rotations_in_space(rotations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Turn rotations, angular velocities or torques into vectors, about the z axis for those of a 2D simulation.

    Args:
        rotations: rotations, shape (k,) in 2D or (k, 3) in 3D.

    Returns:
        The rotation vectors, shape (k, 3).
    """
    if rotations.ndim == 2:
        return rotations
    return np.column_stack((np.zeros((rotations.size, 2)), rotations))


def _particle_stress(sim: DEMSimulation, pairs: ContactList,
                     walls: ContactList) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Count the contacts of every particle and average the stress in it.

    The mean stress in a particle in equilibrium is the sum over its contacts of the branch vector, from its centre to
    the contact point, times the contact force on it, divided by its volume (the Love-Weber formula).

    Args:
        sim: the simulation.
        pairs: contacts between particles.
        walls: contacts between particles and facets.

    Returns:
        The number of contacts of every particle, and its stress, positive in compression, shape (n, d, d) [Pa].
    """
    n, d = sim.n, sim.dimension
    count = np.zeros(n, dtype=np.int64)
    moment = np.zeros((n, d * d))
    if n == 0:
        return count, moment.reshape(0, d, d)
    # particle j of a pair receives the opposite of the force on particle i, at the same point
    for particle, point, force in ((pairs.first, pairs.point, pairs.force), (pairs.second, pairs.point, -pairs.force),
                                   (walls.first, walls.point, walls.force)):
        branch = point - sim.pos[particle]
        moment += scatter_add(particle, (branch[:, :, None] * force[:, None, :]).reshape(-1, d * d), n)
        count += np.bincount(particle, minlength=n)
    volume = 4.0 / 3.0 * np.pi * sim.radius**3
    # contact forces push towards the centre of a particle in compression, against its branch vectors
    return count, -moment.reshape(n, d, d) / volume[:, None, None]


def _particle_data(sim: DEMSimulation, coordination: npt.NDArray[np.int64],
                   stress: npt.NDArray[np.float64]) -> _PolyData:
    """Collect the particles of the current state: their centres, and their data.

    Args:
        sim: the simulation.
        coordination: number of contacts of every particle.
        stress: mean stress tensor in every particle, positive in compression, shape (n, d, d) [Pa].

    Returns:
        The particles, as vertices.
    """
    n, d = sim.n, sim.dimension
    tensor = np.zeros((n, 3, 3))
    tensor[:, :d, :d] = stress
    data: Dict[str, DataArray] = {
        "radius": sim.radius,
        "velocity": _in_space(sim.vel),
        "angular_velocity": _rotations_in_space(sim.omega),
        "coordination": coordination,
        "stress": tensor.reshape(n, 9),
        "mean_stress": np.trace(stress, axis1=1, axis2=2) / d,
        "sphere": (~sim.polytope).astype(np.int64),
    }
    if sim.water is not None:
        data["porosity"] = sim.porosity
        data["fluid_force"] = _in_space(sim.fluid_force)
        if sim.water.coupled:
            data["excess_pore_pressure"] = sim.excess_pore_pressure(sim.pos)
    index = np.arange(n, dtype=np.int64)
    return _PolyData(_in_space(sim.pos), "Verts", index, index + 1, data, {})


def _shape_data(sim: DEMSimulation, coordination: npt.NDArray[np.int64], stress: npt.NDArray[np.float64]) -> _PolyData:
    """Collect the shapes of the polygonal and polyhedral particles in space.

    Args:
        sim: the simulation.
        coordination: number of contacts of every particle.
        stress: mean stress tensor in every particle, positive in compression, shape (n, d, d) [Pa].

    Returns:
        The outline of every polygon, or every face of every polyhedron, as a polygon.
    """
    table = sim.shapes
    rows, vertices = frame_polygons(table, sim.pos, sim.orientation)
    if rows.size == 0:
        none = np.zeros(0, dtype=np.int64)
        return _PolyData(np.zeros((0, 3)), "Polys", none, none.copy(), {}, {})
    if sim.dimension == 2:
        owner = rows
        counts = table.vertex_count[rows]
        corners = vertices
    else:
        faces = table.faces[rows]
        # padding faces have a zero normal, and every face is padded with copies of its last vertex
        real = np.any(table.normals[rows] != 0.0, axis=2)
        particle, face = np.nonzero(real)
        owner = rows[particle]
        loops = faces[particle, face]
        counts = np.argmax(loops == loops[:, -1:], axis=1) + 1
        corners = vertices[particle[:, None], loops]
    used = np.arange(corners.shape[1])[None, :] < counts[:, None]
    mean_stress = np.trace(stress, axis1=1, axis2=2) / sim.dimension
    data: Dict[str, DataArray] = {
        "particle": owner.astype(np.int64),
        "velocity": _in_space(sim.vel[owner].reshape(-1, sim.dimension)),
        "angular_velocity": _rotations_in_space(sim.omega[owner]),
        "coordination": coordination[owner],
        "mean_stress": mean_stress[owner],
    }
    points = corners[used].reshape(-1, sim.dimension)
    return _PolyData(_in_space(points), "Polys", np.arange(points.shape[0], dtype=np.int64),
                     np.cumsum(counts).astype(np.int64), {}, data)


def _contact_data(sim: DEMSimulation, pairs: ContactList, walls: ContactList) -> _PolyData:
    """Collect the contacts of the current state, as lines from the centre of particle i.

    Every line has two points of its own, so that ParaView can turn its data into point data exactly, e.g. to vary
    the radius of a tube along it.

    Args:
        sim: the simulation.
        pairs: contacts between particles, drawn to the centre of particle j.
        walls: contacts between particles and facets, drawn to the contact point.

    Returns:
        The contacts, as lines.
    """
    starts = np.vstack((sim.pos[pairs.first], sim.pos[walls.first]))
    ends = np.vstack((sim.pos[pairs.second], walls.point))
    count = starts.shape[0]
    points = np.stack((starts, ends), axis=1).reshape(2 * count, sim.dimension)
    data: Dict[str, DataArray] = {
        "normal_force": np.concatenate((pairs.normal_force, walls.normal_force)),
        "tangential_force": np.concatenate((norm(pairs.tangential_force), norm(walls.tangential_force))),
        "overlap": np.concatenate((pairs.overlap, walls.overlap)),
        # a particle pair touches no facet
        "facet": np.concatenate((np.full(pairs.first.size, -1, dtype=np.int64), walls.second)),
    }
    return _PolyData(_in_space(points), "Lines", np.arange(2 * count, dtype=np.int64),
                     np.arange(2, 2 * count + 1, 2, dtype=np.int64), {}, data)


def _boundary_data(sim: DEMSimulation, walls: ContactList) -> _PolyData:
    """Collect the facets of the domain at their current position, with the force of the particles on them.

    Args:
        sim: the simulation.
        walls: contacts between particles and facets.

    Returns:
        The facets, as lines in 2D and polygons in 3D.
    """
    geometry = sim.geometry
    vertices = geometry.facets.vertex_array().copy()
    for platen in sim.platens:
        vertices[list(platen.edges)] += platen.shift
    if isinstance(geometry, Geometry3D):
        counts = np.array([len(face) for face in geometry.faces], dtype=np.int64)
        cells = "Polys"
    else:
        counts = np.full(geometry.n_facets, 2, dtype=np.int64)
        cells = "Lines"
    # the faces of a polyhedron are padded with copies of their last vertex, which are left out
    used = np.arange(vertices.shape[1])[None, :] < counts[:, None]
    offsets = np.cumsum(counts)
    # the particles push each facet with the opposite of the force it exerts on them
    force = -scatter_add(walls.second, walls.force, geometry.n_facets)
    pressure = -np.einsum("ij,ij->i", force, geometry.inward_normals) / geometry.facets.measures
    condition = [
        OPEN_CODE if isinstance(c, Open) else PLATEN_CODE if isinstance(c, PLATEN_CONDITIONS) else WALL_CODE
        for c in geometry.boundary_conditions
    ]
    data: Dict[str, DataArray] = {
        "facet": np.arange(geometry.n_facets, dtype=np.int64),
        "condition": np.array(condition, dtype=np.int64),
        "force": _in_space(force),
        "pressure": pressure,
    }
    return _PolyData(_in_space(vertices[used]), cells, np.arange(offsets[-1], dtype=np.int64), offsets, {}, data)


def _water_data(sim: DEMSimulation) -> _PolyData:
    """Collect the free surface of the water across the bounding box of the domain.

    Args:
        sim: the simulation.

    Returns:
        The free surface, a line in 2D and a rectangle in 3D, at the top of the domain if the water is above it, and
        empty if the particles are dry or the water is below the domain.
    """
    d = sim.dimension
    cells = "Lines" if d == 2 else "Polys"
    low, high = sim.geometry.bounds
    level = evaluate_at(sim.water.level, sim.time) if sim.water is not None else -np.inf
    if level < low[1]:
        none = np.zeros(0, dtype=np.int64)
        return _PolyData(np.zeros((0, 3)), cells, none, none.copy(), {}, {})
    level = min(level, float(high[1]))
    if d == 2:
        points = np.array([[low[0], level, 0.0], [high[0], level, 0.0]])
    else:
        # counter-clockwise seen from above, so that the normal of the surface points up
        points = np.array([[low[0], level, low[2]], [low[0], level, high[2]], [high[0], level, high[2]],
                           [high[0], level, low[2]]])
    count = points.shape[0]
    return _PolyData(points, cells, np.arange(count, dtype=np.int64), np.array([count], dtype=np.int64), {}, {})


class _AppendedData:
    """Arrays appended in raw binary to a VTK XML file, each preceded by its size in bytes."""

    def __init__(self) -> None:
        """Start with no arrays."""
        self.blocks: List[bytes] = []
        self.size = 0

    def element(self, name: str, values: DataArray) -> str:
        """Append an array, and return the XML element that refers to it.

        Args:
            name: name of the array.
            values: values of the array, shape (k,) or (k, components); integers are written as integers.

        Returns:
            The DataArray element.
        """
        integer = values.dtype.kind in "biu"
        array = np.ascontiguousarray(values, dtype="<i8" if integer else "<f8")
        raw = array.tobytes()
        components = 1 if array.ndim == 1 else array.shape[1]
        element = (f'<DataArray type="{"Int64" if integer else "Float64"}" Name="{name}" '
                   f'NumberOfComponents="{components}" format="appended" offset="{self.size}"/>')
        self.blocks.append(np.array([len(raw)], dtype="<u8").tobytes() + raw)
        self.size += 8 + len(raw)
        return element


def _write_polydata(path: str, data: _PolyData) -> None:
    """Write a VTK PolyData file in the XML format, with its arrays appended in raw binary.

    Args:
        path: path of the file.
        data: the data set.
    """
    appended = _AppendedData()

    def section(tag: str, arrays: Dict[str, DataArray]) -> List[str]:
        return [f"      <{tag}>"] + [f"        {appended.element(k, v)}"
                                     for k, v in arrays.items()] + [f"      </{tag}>"]

    counts = {"Verts": 0, "Lines": 0, "Polys": 0}
    counts[data.cells] = data.offsets.size
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
        "  <PolyData>",
        f'    <Piece NumberOfPoints="{data.points.shape[0]}" NumberOfVerts="{counts["Verts"]}" '
        f'NumberOfLines="{counts["Lines"]}" NumberOfStrips="0" NumberOfPolys="{counts["Polys"]}">',
    ]
    lines += section("PointData", data.point_data)
    lines += section("CellData", data.cell_data)
    lines += section("Points", {"Points": data.points})
    lines += section(data.cells, {"connectivity": data.connectivity, "offsets": data.offsets})
    lines += ["    </Piece>", "  </PolyData>", '  <AppendedData encoding="raw">', "_"]
    with open(path, "wb") as stream:
        stream.write("\n".join(lines).encode("utf-8"))
        stream.write(b"".join(appended.blocks))
        stream.write(b"\n  </AppendedData>\n</VTKFile>\n")


def _write_collection(path: str, entries: Sequence[Tuple[float, str]]) -> None:
    """Write a ParaView data collection that lists files with their time.

    The collection is written to a temporary file that then replaces the old one, so that a run stopped meanwhile
    leaves a readable collection.

    Args:
        path: path of the collection.
        entries: simulation time [s] and path, relative to the collection, of each file.
    """
    lines = [
        '<?xml version="1.0"?>', '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">', "  <Collection>"
    ]
    lines += [f'    <DataSet timestep="{time!r}" part="0" file="{file}"/>' for time, file in entries]
    lines += ["  </Collection>", "</VTKFile>", ""]
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    os.replace(temporary, path)
