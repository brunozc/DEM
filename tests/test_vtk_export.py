"""Tests of the export to VTK files for ParaView."""

import math
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from DEMcookies import (DEMSimulation, Geometry, Geometry3D, GlobalSettings, Heaviside, LoadedPlaten,
                        MaterialProperties, Open, Particle, Particle3D, Wall, Water, generate_lattice_particles)
from DEMcookies.vtk_export import (OPEN_CODE, PARAVIEW_SCRIPT, PLATEN_CODE, SERIES, WALL_CODE, VTKWriter)

MATERIAL = MaterialProperties(density=2500.0, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.1, friction=0.5)

Arrays = Dict[str, npt.NDArray[np.float64]]


def _read_vtp(path: Path) -> Tuple[Dict[str, str], Arrays]:
    """Read a PolyData file of the writer: the attributes of its piece, and its arrays by section and name."""
    content = path.read_bytes()
    head, tail = content.split(b'<AppendedData encoding="raw">')
    data = tail[tail.index(b"_") + 1:]
    piece = ElementTree.fromstring(head + b"</VTKFile>").find("PolyData/Piece")
    assert piece is not None
    arrays: Arrays = {}
    for section in piece:
        for element in section.findall("DataArray"):
            offset = int(element.attrib["offset"])
            size = int(np.frombuffer(data[offset:offset + 8], dtype="<u8")[0])
            dtype = "<f8" if element.attrib["type"] == "Float64" else "<i8"
            values = np.frombuffer(data[offset + 8:offset + 8 + size], dtype=dtype)
            components = int(element.attrib["NumberOfComponents"])
            arrays[f"{section.tag}/{element.attrib['Name']}"] = values.reshape(-1, components) if components > 1 \
                else values
    return dict(piece.attrib), arrays


def _read_collection(path: Path) -> List[Tuple[float, str]]:
    """Read the time and file of every data set of a collection."""
    root = ElementTree.parse(path).getroot()
    return [(float(d.attrib["timestep"]), d.attrib["file"]) for d in root.iter("DataSet")]


def _last(directory: Path, name: str) -> Tuple[Dict[str, str], Arrays]:
    """Read the last file of a series."""
    return _read_vtp(directory / _read_collection(directory / f"{name}.pvd")[-1][1])


def _resting_particle_3d() -> DEMSimulation:
    """Return a particle at rest on the floor of a box."""
    sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), MATERIAL, particles=[Particle3D(0.05, 0.01, 0.04, 0.01)])
    sim.rest(0.5)
    return sim


def _pressed_particle_3d() -> DEMSimulation:
    """Return a particle on the floor of a box whose top, a loaded platen, has come down and presses on it."""
    platen = LoadedPlaten(load=Heaviside(0.5, rise_time=0.01), mass=0.2)
    sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1, top=platen),
                        MATERIAL,
                        particles=[Particle3D(0.05, 0.01, 0.04, 0.01)])
    sim.rest(0.3)
    sim.compress(0.02, record_every=20)
    return sim


class TestVTKWriter:
    """Tests of the series, collections and data written by VTKWriter."""

    def test_observing_writer_exports_every_n_steps(self, tmp_path: Path) -> None:
        """An observing writer must export after every n-th step, whatever the stage, with its time in every series."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        writer = VTKWriter(str(tmp_path), save_every_n_steps=7)
        sim.add_observer(writer)
        # the recording interval of the stages has nothing to do with the exports
        sim.rest(0.01, record_every=40)
        sim.rest(0.01, record_every=3)
        exports = sim.steps_taken // 7
        assert exports > 5
        times = [7 * k * sim.dt for k in range(1, exports + 1)]
        for name in SERIES:
            entries = _read_collection(tmp_path / f"{name}.pvd")
            assert_allclose([time for time, _ in entries], times, rtol=1e-12)
            assert all((tmp_path / file).is_file() for _, file in entries)
        assert_allclose(writer.times, times, rtol=1e-12)

    def test_resumed_writer_carries_on_with_its_series(self, tmp_path: Path) -> None:
        """A writer pickled with its simulation must add to its series when the copy is resumed."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        sim.add_observer(VTKWriter(str(tmp_path), save_every_n_steps=10))
        for _ in range(30):
            sim.step()
        copy = pickle.loads(pickle.dumps(sim))
        for _ in range(20):
            copy.step()
        entries = _read_collection(tmp_path / "particles.pvd")
        assert [file for _, file in entries] == [f"particles/particles_{k:06d}.vtp" for k in range(5)]
        assert_allclose([time for time, _ in entries], [10 * k * sim.dt for k in range(1, 6)], rtol=1e-12)

    def test_invalid_interval_is_rejected(self, tmp_path: Path) -> None:
        """A non-positive export interval must be rejected."""
        with pytest.raises(ValueError, match=re.escape("save_every_n_steps must be at least 1, got 0.")):
            VTKWriter(str(tmp_path), save_every_n_steps=0)

    def test_observing_writer_without_interval_is_rejected(self, tmp_path: Path) -> None:
        """A writer without an interval between exports must refuse to observe a simulation."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        sim.add_observer(VTKWriter(str(tmp_path)))
        with pytest.raises(ValueError,
                           match=re.escape("A VTKWriter needs save_every_n_steps to observe a simulation.")):
            sim.step()

    def test_paraview_script_is_written(self, tmp_path: Path) -> None:
        """The ParaView script must be written next to the collections, and be valid Python."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        VTKWriter(str(tmp_path)).write(sim)
        script = (tmp_path / PARAVIEW_SCRIPT).read_text(encoding="utf-8")
        compile(script, PARAVIEW_SCRIPT, "exec")
        assert all(f'"{name}"' in script for name in SERIES)

    def test_removed_folders_are_made_again(self, tmp_path: Path) -> None:
        """An export must make the folders and the script again if they were removed, e.g. to free disk space."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        writer = VTKWriter(str(tmp_path / "paraview"))
        writer.write(sim)
        shutil.rmtree(tmp_path / "paraview")
        writer.write(sim)
        assert (tmp_path / "paraview" / PARAVIEW_SCRIPT).is_file()
        assert (tmp_path / "paraview" / "particles" / "particles_000001.vtp").is_file()

    def test_writer_does_not_change_the_run(self, tmp_path: Path) -> None:
        """Exporting after every step must leave the motion of the particles exactly as it was."""
        runs = []
        for export in (False, True):
            geometry = Geometry3D.box(0.05, 0.1, 0.05)
            sim = DEMSimulation(geometry,
                                MaterialProperties(2500.0,
                                                   5.0e6,
                                                   0.25,
                                                   restitution=0.5,
                                                   friction=0.5,
                                                   rolling_friction=0.2),
                                particles=generate_lattice_particles(geometry, 20, (0.004, 0.005)))
            if export:
                sim.add_observer(VTKWriter(str(tmp_path), save_every_n_steps=1))
            sim.rest(0.03, record_every=10)
            runs.append(sim)
        plain, exported = runs
        assert_allclose(exported.pos, plain.pos, rtol=0.0, atol=1e-15)

    def test_particles_carry_their_state_in_3d(self, tmp_path: Path) -> None:
        """The particles must carry their centres, radii, velocities, spins, contacts and stress."""
        sim = _resting_particle_3d()
        writer = VTKWriter(str(tmp_path))
        writer.write(sim)
        piece, arrays = _last(tmp_path, "particles")
        assert piece["NumberOfPoints"] == piece["NumberOfVerts"] == "1"
        assert_allclose(arrays["Points/Points"], sim.pos, rtol=1e-15)
        assert_allclose(arrays["PointData/radius"], sim.radius, rtol=1e-15)
        assert_allclose(arrays["PointData/velocity"], sim.vel, rtol=1e-15, atol=1e-300)
        assert_allclose(arrays["PointData/angular_velocity"], sim.omega, rtol=1e-15, atol=1e-300)
        assert arrays["PointData/coordination"].tolist() == [1]
        _, walls = sim.contacts()
        # one contact below the centre, pushed up by the floor: only the vertical normal stress is not zero
        volume = 4.0 / 3.0 * math.pi * 0.01**3
        sigma_yy = (0.01 - 0.5 * walls.overlap[0]) * walls.normal_force[0] / volume
        expected = np.zeros(9)
        expected[4] = sigma_yy
        assert_allclose(arrays["PointData/stress"][0], expected, rtol=1e-9, atol=1e-9 * sigma_yy)
        assert_allclose(arrays["PointData/mean_stress"], [sigma_yy / 3.0], rtol=1e-9)
        assert "PointData/porosity" not in arrays

    def test_planar_vectors_lie_in_the_plane_z_0(self, tmp_path: Path) -> None:
        """A 2D simulation must be written in the plane z = 0, with its spins about the z axis."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            MATERIAL,
                            particles=[Particle(0.05, 0.05, 0.01, vx=0.1, vy=0.2)],
                            settings=GlobalSettings(gravity=0.0))
        sim.omega[:] = 3.0
        VTKWriter(str(tmp_path)).write(sim)
        _, arrays = _last(tmp_path, "particles")
        assert_allclose(arrays["Points/Points"], [[0.05, 0.05, 0.0]], rtol=1e-15)
        assert_allclose(arrays["PointData/velocity"], [[0.1, 0.2, 0.0]], rtol=1e-15)
        assert_allclose(arrays["PointData/angular_velocity"], [[0.0, 0.0, 3.0]], rtol=1e-15)
        assert arrays["PointData/stress"].shape == (1, 9)

    def test_contacts_are_lines_with_their_forces(self, tmp_path: Path) -> None:
        """Every contact must be a line from the centre of particle i, with the forces of the contact law."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1),
                            MATERIAL,
                            particles=[Particle(0.05, 0.0099, 0.01),
                                       Particle(0.05, 0.029, 0.01)])
        VTKWriter(str(tmp_path)).write(sim)
        piece, arrays = _last(tmp_path, "contacts")
        pairs, walls = sim.contacts()
        assert pairs.first.size == 1 and walls.first.size == 1
        assert piece["NumberOfLines"] == "2" and piece["NumberOfPoints"] == "4"
        assert arrays["Lines/offsets"].tolist() == [2, 4]
        points = arrays["Points/Points"][:, :2]
        assert_allclose(points[:2], sim.pos[[pairs.first[0], pairs.second[0]]], rtol=1e-15)
        assert_allclose(points[2:], [sim.pos[walls.first[0]], walls.point[0]], rtol=1e-15)
        assert arrays["CellData/facet"].tolist() == [-1, 0]
        assert_allclose(arrays["CellData/normal_force"],
                        np.concatenate((pairs.normal_force, walls.normal_force)),
                        rtol=1e-15)
        assert_allclose(arrays["CellData/overlap"], np.concatenate((pairs.overlap, walls.overlap)), rtol=1e-15)

    def test_boundary_moves_with_its_platen_and_carries_the_pressure(self, tmp_path: Path) -> None:
        """The boundary must show the platen where it is, the conditions of the faces and the pressure on them."""
        sim = _pressed_particle_3d()
        VTKWriter(str(tmp_path)).write(sim)
        piece, arrays = _last(tmp_path, "boundary")
        assert piece["NumberOfPolys"] == "6"
        assert arrays["Polys/offsets"].tolist() == [4, 8, 12, 16, 20, 24]
        assert arrays["CellData/condition"].tolist() == [
            WALL_CODE, WALL_CODE, PLATEN_CODE, WALL_CODE, WALL_CODE, WALL_CODE
        ]
        top = arrays["Points/Points"][8:12]
        platen = sim.platens[0]
        assert platen.offset > 0.05
        assert_allclose(top, platen.vertices[0], rtol=1e-15)
        _, walls = sim.contacts()
        assert walls.second.tolist() == [0, 2]
        # the floor pushes the particle up and the platen pushes it down, and both feel its push back as a pressure
        assert_allclose(arrays["CellData/force"][[0, 2]], -walls.force, rtol=1e-15)
        expected = np.zeros(6)
        expected[[0, 2]] = np.abs(walls.force[:, 1]) / 0.01
        assert expected[2] > 0.5 / 0.01
        assert_allclose(arrays["CellData/pressure"], expected, rtol=1e-12, atol=1e-12)

    def test_boundary_keeps_the_vertices_of_each_face(self, tmp_path: Path) -> None:
        """Faces with different numbers of vertices must keep theirs, without the padding, and open faces count."""
        triangle = Geometry([(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)], [Wall(), Open(), Wall()])
        geometry = Geometry3D.extrude(triangle, 0.05)
        sim = DEMSimulation(geometry, MATERIAL, radius_range=(0.005, 0.01))
        VTKWriter(str(tmp_path)).write(sim)
        _, arrays = _last(tmp_path, "boundary")
        counts = [len(face) for face in geometry.faces]
        assert sorted(counts) == [3, 3, 4, 4, 4]
        assert arrays["Polys/offsets"].tolist() == np.cumsum(counts).tolist()
        assert_allclose(arrays["Points/Points"],
                        np.vstack([geometry.vertices[list(face)] for face in geometry.faces]),
                        rtol=1e-15)
        assert arrays["CellData/condition"].tolist() == [WALL_CODE, OPEN_CODE, WALL_CODE, WALL_CODE, WALL_CODE]

    def test_planar_boundary_is_made_of_lines(self, tmp_path: Path) -> None:
        """The edges of a 2D domain must be lines in the plane z = 0."""
        sim = DEMSimulation(Geometry.rectangle(0.2, 0.1), MATERIAL, radius_range=(0.005, 0.01))
        VTKWriter(str(tmp_path)).write(sim)
        piece, arrays = _last(tmp_path, "boundary")
        assert piece["NumberOfLines"] == "4" and piece["NumberOfPolys"] == "0"
        assert_allclose(arrays["Points/Points"][:2], [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], atol=1e-15)

    @pytest.mark.parametrize("level, height", [(0.06, 0.06), (0.5, 0.1), (-0.1, None)])
    def test_water_surface_spans_the_domain(self, tmp_path: Path, level: float, height: float) -> None:
        """The free surface must span the domain at the water level, on its top if above and absent if below."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.05), MATERIAL, particles=[Particle3D(0.05, 0.05, 0.02, 0.01)])
        sim.set_water(Water(level=level))
        sim.compute_forces()
        VTKWriter(str(tmp_path)).write(sim)
        piece, arrays = _last(tmp_path, "water")
        if height is None:
            assert piece["NumberOfPoints"] == piece["NumberOfPolys"] == "0"
            return
        assert piece["NumberOfPolys"] == "1"
        assert_allclose(arrays["Points/Points"],
                        [[0.0, height, 0.0], [0.0, height, 0.05], [0.1, height, 0.05], [0.1, height, 0.0]],
                        atol=1e-15)

    def test_dry_simulation_has_no_water(self, tmp_path: Path) -> None:
        """A dry simulation must write an empty free surface."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        VTKWriter(str(tmp_path)).write(sim)
        piece, _ = _last(tmp_path, "water")
        assert piece["NumberOfPoints"] == "0"

    @pytest.mark.parametrize("coupled", [False, True])
    def test_particles_in_water_carry_its_data(self, tmp_path: Path, coupled: bool) -> None:
        """Submerged particles must carry their porosity and the force of the water, and coupled water its pressure."""
        sim = DEMSimulation(Geometry.rectangle(0.1, 0.1), MATERIAL, particles=[Particle(0.05, 0.05, 0.01)])
        sim.set_water(Water(level=0.08, coupled=coupled))
        sim.step()
        VTKWriter(str(tmp_path)).write(sim)
        _, arrays = _last(tmp_path, "particles")
        assert_allclose(arrays["PointData/porosity"], sim.porosity, rtol=1e-15)
        assert_allclose(arrays["PointData/fluid_force"][:, :2], sim.fluid_force, rtol=1e-15)
        assert arrays["PointData/fluid_force"][0, 1] > 0.0
        assert ("PointData/excess_pore_pressure" in arrays) == coupled

    def test_empty_domain_is_written(self, tmp_path: Path) -> None:
        """An empty domain must give empty particle and contact files, and its boundary."""
        sim = DEMSimulation(Geometry3D.box(0.1, 0.1, 0.1), MATERIAL, radius_range=(0.005, 0.01))
        VTKWriter(str(tmp_path)).write(sim)
        for name in ("particles", "contacts"):
            piece, _ = _last(tmp_path, name)
            assert piece["NumberOfPoints"] == "0"
        piece, _ = _last(tmp_path, "boundary")
        assert piece["NumberOfPolys"] == "6"
