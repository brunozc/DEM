"""Tests of the animation and load curve plots."""

import re
from pathlib import Path

import pytest

from DEMcookies import DEMSimulation, Geometry, Heaviside, LoadedPlaten, MaterialProperties, Particle, Water
from DEMcookies.plotting import plot_load_curves, render_animation, render_frame

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

    def test_snapshot_with_water_is_written(self, tmp_path: Path) -> None:
        """A snapshot of a simulation with water must be written, with the water drawn below its level."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05),
                            MATERIAL,
                            particles=[Particle(0.025, 0.01, 0.01)],
                            water=Water(level=0.03))
        sim.rest(0.01, record_every=10)
        render_frame(sim, str(tmp_path / "wet.png"))
        assert (tmp_path / "wet.png").stat().st_size > 0

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
