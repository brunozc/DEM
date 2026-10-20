"""Tests of the animation and load curve plots."""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import GifImagePlugin, Image

from DEMcookies import (DEMSimulation, Geometry, Geometry3D, Heaviside, LoadedPlaten, MaterialProperties, Open,
                        Particle, Particle3D, Wall, Water)
from DEMcookies.plotting import (_chain_loops, _clip_below, _current_outline, _current_polyhedron, _facet_label,
                                 _plane_crossings, plot_load_curves, render_animation, render_frame)

MATERIAL = MaterialProperties(density=2500.0, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.1, friction=0.5)


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
