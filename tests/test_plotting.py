"""Tests of the animation and load curve plots."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FFMpegWriter
from numpy.testing import assert_allclose
from PIL import GifImagePlugin, Image

from DEMcookies import (DEMSimulation, Geometry, Geometry3D, Heaviside, LoadedPlaten, MaterialProperties, Open,
                        Particle, Particle3D, PrescribedPlaten, Wall, Water)
from DEMcookies.plotting import (_chain_loops, _clip_below, _current_outline, _current_polyhedron, _facet_label,
                                 _plane_crossings, _sections, plot_load_curves, render_animation, render_frame)
from DEMcookies.shapes import ConvexPolygon, ConvexPolyhedron

MATERIAL = MaterialProperties(density=2500.0, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.1, friction=0.5)

NEEDS_FFMPEG = pytest.mark.skipif(not FFMpegWriter.isAvailable() or shutil.which("ffprobe") is None,
                                  reason="ffmpeg and ffprobe are needed to write and read MP4 videos")


def _probe(path: Path) -> Tuple[int, int, int]:
    """Return the width and height [px] and the number of frames of the video in a file, as ffprobe reads them."""
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_read_frames", "-of", "csv=p=0",
        str(path)
    ]
    output = subprocess.run(command, capture_output=True, check=True, text=True).stdout
    width, height, frames = (int(value) for value in output.strip().split(","))
    return width, height, frames


def _loaded_simulation(self_weight: bool = False) -> DEMSimulation:
    """Return a one-particle simulation whose top platen has been loaded briefly."""
    platen = LoadedPlaten(load=Heaviside(1.0, rise_time=0.01), mass=0.2, self_weight=self_weight)
    geometry = Geometry.rectangle(0.05, 0.05, top=platen)
    sim = DEMSimulation(geometry, MATERIAL, particles=[Particle(0.025, 0.01, 0.01)])
    sim.compress(0.02, record_every=20)
    return sim


class TestPlotting:
    """Tests of render_animation and plot_load_curves."""

    def test_outputs_are_written(self, tmp_path: Path) -> None:
        """The animation and the load curve plot must be written to the given files."""
        sim = _loaded_simulation()
        render_animation(sim, str(tmp_path / "run.gif"))
        render_frame(sim, str(tmp_path / "last.png"))
        plot_load_curves(sim, str(tmp_path / "curves.png"))
        for name in ("run.gif", "last.png", "curves.png"):
            assert (tmp_path / name).stat().st_size > 0

    def test_animation_draws_every_nth_frame(self, tmp_path: Path) -> None:
        """An animation with a stride must hold every n-th recorded frame, starting from the first."""
        sim = _loaded_simulation()
        render_animation(sim, str(tmp_path / "run.gif"), every=3)
        with Image.open(tmp_path / "run.gif") as gif:
            assert isinstance(gif, GifImagePlugin.GifImageFile)
            assert gif.n_frames == len(range(0, len(sim.frames), 3))

    def test_animation_with_invalid_stride_is_rejected(self, tmp_path: Path) -> None:
        """A stride below one must be rejected."""
        with pytest.raises(ValueError, match=re.escape("every must be at least 1, got 0.")):
            render_animation(_loaded_simulation(), str(tmp_path / "run.gif"), every=0)

    @NEEDS_FFMPEG
    def test_mp4_animation_draws_every_nth_frame(self, tmp_path: Path) -> None:
        """An MP4 animation with a stride must hold every n-th recorded frame, at the even size that H.264 needs."""
        sim = _loaded_simulation()
        render_animation(sim, str(tmp_path / "run.mp4"), every=3)
        width, height, frames = _probe(tmp_path / "run.mp4")
        assert frames == len(range(0, len(sim.frames), 3))
        assert width % 2 == 0 and height % 2 == 0

    def test_animation_with_unknown_format_is_rejected(self, tmp_path: Path) -> None:
        """An animation can only be written as a GIF or an MP4 video."""
        path = str(tmp_path / "run.avi")
        with pytest.raises(ValueError,
                           match=re.escape(f"An animation is written as a .gif or an .mp4 file, got '{path}'.")):
            render_animation(_loaded_simulation(), path)

    def test_mp4_animation_without_ffmpeg_is_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An MP4 video cannot be encoded without ffmpeg."""
        monkeypatch.setattr(FFMpegWriter, "isAvailable", lambda: False)
        with pytest.raises(RuntimeError,
                           match=re.escape("An .mp4 animation is encoded by ffmpeg, which was not found.")):
            render_animation(_loaded_simulation(), str(tmp_path / "run.mp4"))
        assert not (tmp_path / "run.mp4").exists()

    def test_snapshot_with_water_is_written(self, tmp_path: Path) -> None:
        """A snapshot of a simulation with water must be written, with the water drawn below its level."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05),
                            MATERIAL,
                            particles=[Particle(0.025, 0.01, 0.01)],
                            water=Water(level=0.03))
        sim.rest(0.01, record_every=10)
        render_frame(sim, str(tmp_path / "wet.png"))
        assert (tmp_path / "wet.png").stat().st_size > 0

    def test_water_stops_at_lowered_platen(self) -> None:
        """The water drawn below a level above a lowered platen must end at the platen, not at the original edge."""
        sim = _loaded_simulation()
        platen = sim.platens[0]
        assert platen.offset > 0.0
        recorded = sim.frames[-1].platens
        outline = _current_outline(sim.geometry.vertices, [platen.edges], recorded)
        wet = _clip_below(outline, 0.06)
        assert_allclose(wet[:, 1].max(), recorded[0][0, 0, 1], atol=1e-12)
        assert wet[:, 1].max() < 0.05 - 1.0e-6
        # the vertices of the fixed edges stay where they were defined
        assert_allclose(np.sort(wet[:, 1])[:2], [0.0, 0.0], atol=1e-15)

    def test_load_curves_of_heavy_platen_are_written(self, tmp_path: Path) -> None:
        """The load curves of a platen with self weight, which add the weight line, must be written."""
        plot_load_curves(_loaded_simulation(self_weight=True), str(tmp_path / "curves.png"))
        assert (tmp_path / "curves.png").stat().st_size > 0

    def test_outputs_of_prescribed_platen_are_written(self, tmp_path: Path) -> None:
        """A platen with a prescribed displacement must be drawn, and its displacement history plotted."""
        platen = PrescribedPlaten(displacement=Heaviside(1.0e-4, rise_time=0.01))
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05, top=platen),
                            MATERIAL,
                            particles=[Particle(0.025, 0.01, 0.01)])
        sim.compress(0.02, record_every=20)
        render_frame(sim, str(tmp_path / "last.png"))
        plot_load_curves(sim, str(tmp_path / "curves.png"))
        for name in ("last.png", "curves.png"):
            assert (tmp_path / name).stat().st_size > 0

    def test_animation_without_frames_is_rejected(self, tmp_path: Path) -> None:
        """A simulation that has not run has nothing to animate."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05), MATERIAL, radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The simulation has no recorded frames to draw.")):
            render_animation(sim, str(tmp_path / "run.gif"))

    def test_snapshot_without_frames_is_rejected(self, tmp_path: Path) -> None:
        """A simulation that has not run has no frame to draw."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05), MATERIAL, radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The simulation has no recorded frames to draw.")):
            render_frame(sim, str(tmp_path / "last.png"))

    def test_load_curves_without_loading_are_rejected(self, tmp_path: Path) -> None:
        """A simulation that was never loaded has no load curves to plot."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05), MATERIAL, radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The simulation has no recorded load curves to plot.")):
            plot_load_curves(sim, str(tmp_path / "curves.png"))


def _loaded_simulation_3d(water: Optional[Water] = None) -> DEMSimulation:
    """Return a one-particle 3D simulation whose lid has been loaded briefly."""
    platen = LoadedPlaten(load=Heaviside(1.0, rise_time=0.01), mass=0.2)
    geometry = Geometry3D.box(0.05, 0.05, 0.05, top=platen)
    sim = DEMSimulation(geometry, MATERIAL, particles=[Particle3D(0.025, 0.01, 0.025, 0.01)], water=water)
    sim.compress(0.02, record_every=20)
    return sim


def _footing_simulation_3d(water: Optional[Water] = None) -> DEMSimulation:
    """Return a 3D simulation of a strip footing, between two open faces, loaded briefly on one particle."""
    footing = LoadedPlaten(load=Heaviside(1.0, rise_time=0.01), mass=0.2)
    section = Geometry([(0.0, 0.0), (0.06, 0.0), (0.06, 0.05), (0.04, 0.05), (0.02, 0.05), (0.0, 0.05)],
                       [Wall(), Wall(), Open(), footing, Open(), Wall()])
    sim = DEMSimulation(Geometry3D.extrude(section, 0.04),
                        MATERIAL,
                        particles=[Particle3D(0.03, 0.01, 0.02, 0.01)],
                        water=water)
    sim.compress(0.02, record_every=20)
    return sim


class TestPlotting3D:
    """Tests of the views of 3D simulations."""

    def test_outputs_are_written(self, tmp_path: Path) -> None:
        """The oblique view, the section, the animation and the load curves must be written to the given files."""
        sim = _footing_simulation_3d()
        render_animation(sim, str(tmp_path / "run.gif"), every=2)
        render_animation(sim, str(tmp_path / "section.gif"), section=0.025)
        render_frame(sim, str(tmp_path / "last.png"))
        render_frame(sim, str(tmp_path / "cut.png"), section=0.02)
        plot_load_curves(sim, str(tmp_path / "curves.png"))
        for name in ("run.gif", "section.gif", "last.png", "cut.png", "curves.png"):
            assert (tmp_path / name).stat().st_size > 0
        with Image.open(tmp_path / "run.gif") as gif:
            assert isinstance(gif, GifImagePlugin.GifImageFile)
            assert gif.n_frames == len(range(0, len(sim.frames), 2))
            # the blank bands around the square in which a 3D view is drawn are cropped away
            full = int(5.0 * plt.rcParams["figure.dpi"])
            assert gif.size[0] < full and gif.size[1] < full

    def test_views_with_water_are_written(self, tmp_path: Path) -> None:
        """Snapshots of a 3D simulation with water must be written, with the water drawn below its level."""
        sim = _loaded_simulation_3d(Water(level=0.03))
        render_frame(sim, str(tmp_path / "wet.png"))
        render_frame(sim, str(tmp_path / "wet_cut.png"), section=0.025)
        assert (tmp_path / "wet.png").stat().st_size > 0 and (tmp_path / "wet_cut.png").stat().st_size > 0
        # the water of the section is redrawn on every frame
        render_animation(_footing_simulation_3d(Water(level=0.03)), str(tmp_path / "wet.gif"), section=0.02)
        assert (tmp_path / "wet.gif").stat().st_size > 0

    @NEEDS_FFMPEG
    def test_mp4_outputs_are_written(self, tmp_path: Path) -> None:
        """The oblique view and the section must be written as MP4 videos, the oblique view cropped to an even size."""
        sim = _footing_simulation_3d()
        render_animation(sim, str(tmp_path / "run.mp4"), every=2)
        render_animation(sim, str(tmp_path / "section.mp4"), section=0.025)
        width, height, frames = _probe(tmp_path / "run.mp4")
        assert frames == len(range(0, len(sim.frames), 2))
        full = int(5.0 * plt.rcParams["figure.dpi"])
        assert width < full and height < full
        assert width % 2 == 0 and height % 2 == 0
        assert _probe(tmp_path / "section.mp4")[2] == len(sim.frames)

    @NEEDS_FFMPEG
    def test_mp4_that_ffmpeg_cannot_write_is_rejected(self, tmp_path: Path) -> None:
        """The reason that ffmpeg gives must be passed on when it cannot write the video of an oblique view."""
        path = str(tmp_path / "missing" / "run.mp4")
        with pytest.raises(RuntimeError, match=re.escape(f"ffmpeg could not write {path}: ") + ".*No such file"):
            render_animation(_footing_simulation_3d(), path)

    def test_views_without_frames_are_rejected(self, tmp_path: Path) -> None:
        """A 3D simulation that has not run has nothing to draw, in an oblique view or in a section."""
        sim = DEMSimulation(Geometry3D.box(0.05, 0.05, 0.05), MATERIAL, radius_range=(0.01, 0.01))
        for section in (None, 0.02):
            with pytest.raises(ValueError, match=re.escape("The simulation has no recorded frames to draw.")):
                render_frame(sim, str(tmp_path / "view.png"), section=section)

    def test_section_of_2d_simulation_is_rejected(self, tmp_path: Path) -> None:
        """A 2D simulation has no third dimension to cut."""
        with pytest.raises(ValueError,
                           match=re.escape("A section can only be drawn of a three-dimensional simulation.")):
            render_frame(_loaded_simulation(), str(tmp_path / "cut.png"), section=0.01)

    def test_section_outside_domain_is_rejected(self, tmp_path: Path) -> None:
        """A section plane must cut the domain."""
        message = "The section z = 0.5 does not cut the domain, which spans z = 0 to 0.05."
        with pytest.raises(ValueError, match=re.escape(message)):
            render_frame(_loaded_simulation_3d(), str(tmp_path / "cut.png"), section=0.5)

    def test_water_stops_at_lowered_platen(self) -> None:
        """The water below a level above a lowered lid must end at the lid, in the section and in the oblique view."""
        sim = _loaded_simulation_3d()
        platen = sim.platens[0]
        assert platen.offset > 0.0
        assert isinstance(sim.geometry, Geometry3D)
        faces = _current_polyhedron(sim.geometry, [platen.faces], sim.frames[-1].platens)
        lid = sim.frames[-1].platens[0][0, 0, 1]
        assert_allclose([face[:, 1].max() for face in faces], [0.0, lid, lid, lid, lid, lid], atol=1e-12)
        crossings = [crossing for face in faces for crossing in _plane_crossings(face, 2, 0.025)]
        loops = _chain_loops(crossings, 1.0e-12)
        assert len(loops) == 1 and loops[0].shape == (4, 3)
        wet = _clip_below(loops[0][:, :2], 0.06)
        assert_allclose(wet[:, 1].max(), lid, atol=1e-12)
        assert lid < 0.05 - 1.0e-6

    def test_plane_crossings_of_non_convex_polygon(self) -> None:
        """A U-shaped face must cross a plane through both of its arms, in two separate segments."""
        u_shape = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        segments = sorted(tuple(sorted((float(a[0]), float(b[0])))) for a, b in _plane_crossings(u_shape, 1, 1.5))
        assert_allclose(segments, [(0.0, 1.0), (2.0, 3.0)])
        assert _plane_crossings(u_shape, 1, 2.5) == []

    def test_chain_loops_joins_segments_in_any_order(self) -> None:
        """Segments given in any order and orientation must be joined into one closed loop."""
        square = [(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                  (np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                  (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),
                  (np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))]
        loops = _chain_loops(square, 1.0e-12)
        assert len(loops) == 1
        assert sorted(map(tuple, loops[0].tolist())) == [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                                                         (1.0, 1.0, 0.0)]
        # an open chain is dropped
        assert _chain_loops(square[:3], 1.0e-12) == []

    def test_facet_label(self) -> None:
        """Loaded bodies must be described by the name and the indices of their facets."""
        assert _facet_label((3, ), "face") == "face 3"
        assert _facet_label((3, 4, 5), "face") == "faces 3-5"
        assert _facet_label((1, 4), "edge") == "edges 1, 4"


class TestShapePlotting:
    """Tests of the drawing of polygonal and polyhedral particles."""

    def test_polygons_are_drawn(self, tmp_path: Path) -> None:
        """A snapshot of a 2D simulation with a polygon and a disc must be written."""
        square = ConvexPolygon.rectangle(1.0, 1.0)
        sim = DEMSimulation(
            Geometry.rectangle(0.05, 0.05),
            MATERIAL,
            particles=[Particle(0.015, 0.012, 0.008, shape=square, angle=0.4),
                       Particle(0.035, 0.012, 0.008)])
        sim.rest(0.01, record_every=5)
        render_frame(sim, str(tmp_path / "polygons.png"))
        assert (tmp_path / "polygons.png").stat().st_size > 0

    def test_polyhedra_are_drawn_in_both_views(self, tmp_path: Path) -> None:
        """Snapshots of a 3D simulation with polyhedra must be written in the oblique view and in a section."""
        cube = ConvexPolyhedron.box(1.0, 1.0, 1.0)
        sim = DEMSimulation(
            Geometry3D.box(0.05, 0.05, 0.05),
            MATERIAL,
            particles=[Particle3D(0.015, 0.012, 0.025, 0.008, shape=cube),
                       Particle3D(0.035, 0.012, 0.025, 0.008)])
        sim.rest(0.01, record_every=5)
        render_frame(sim, str(tmp_path / "oblique.png"))
        render_frame(sim, str(tmp_path / "section.png"), section=0.025)
        assert (tmp_path / "oblique.png").stat().st_size > 0 and (tmp_path / "section.png").stat().st_size > 0

    def test_section_of_a_cube_is_a_square(self) -> None:
        """A plane through the middle of an upright cube must cut a square out of it."""
        cube = ConvexPolyhedron.box(1.0, 1.0, 1.0)
        sim = DEMSimulation(Geometry3D.box(0.05, 0.05, 0.05),
                            MATERIAL,
                            particles=[Particle3D(0.025, 0.02, 0.025, cube.equivalent_radius * 0.01, shape=cube)])
        sim.rest(0.001, record_every=1)
        rows, polygons = _sections(sim.frames[0], 0.025)
        assert list(rows) == [0]
        assert_allclose(np.ptp(polygons[0], axis=0), [0.01, 0.01], rtol=1e-9)
        rows, polygons = _sections(sim.frames[0], 0.04)
        assert rows.size == 0
