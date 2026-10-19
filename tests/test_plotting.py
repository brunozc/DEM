"""Tests of the animation and load curve plots."""

import re
from pathlib import Path

import pytest

from DEMcookies import DEMSimulation, Geometry, Heaviside, LoadedPlaten, MaterialProperties, Particle
from DEMcookies.plotting import plot_load_curves, render_animation

MATERIAL = MaterialProperties(density=2500.0, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.1, friction=0.5)


def _loaded_simulation() -> DEMSimulation:
    """Return a one-particle simulation whose top platen has been loaded briefly."""
    geometry = Geometry.rectangle(0.05, 0.05, top=LoadedPlaten(load=Heaviside(1.0, rise_time=0.01), mass=0.2))
    sim = DEMSimulation(geometry, MATERIAL, particles=[Particle(0.025, 0.01, 0.01)])
    sim.compress(0.02, record_every=20)
    return sim


class TestPlotting:
    """Tests of render_animation and plot_load_curves."""

    def test_outputs_are_written(self, tmp_path: Path) -> None:
        """The animation and the load curve plot must be written to the given files."""
        sim = _loaded_simulation()
        render_animation(sim, str(tmp_path / "run.gif"))
        plot_load_curves(sim, str(tmp_path / "curves.png"))
        assert (tmp_path / "run.gif").stat().st_size > 0
        assert (tmp_path / "curves.png").stat().st_size > 0

    def test_animation_without_frames_is_rejected(self, tmp_path: Path) -> None:
        """A simulation that has not run has nothing to animate."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05), MATERIAL, radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The simulation has no recorded frames to animate.")):
            render_animation(sim, str(tmp_path / "run.gif"))

    def test_load_curves_without_loading_are_rejected(self, tmp_path: Path) -> None:
        """A simulation that was never loaded has no load curves to plot."""
        sim = DEMSimulation(Geometry.rectangle(0.05, 0.05), MATERIAL, radius_range=(0.01, 0.01))
        with pytest.raises(ValueError, match=re.escape("The simulation has no recorded load curves to plot.")):
            plot_load_curves(sim, str(tmp_path / "curves.png"))
