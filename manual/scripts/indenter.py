"""Curved indenter example: capacity under a slow ramp, and the load-hold-unload cycle of the example.

Run from the repository root with ``python manual/scripts/indenter.py``. The container is filled once; one copy of
the packing is loaded with a slow ramp to 20 N over 2 s, another with the 4 N cycle of the example.
"""

import copy
import dataclasses

import numpy as np

from DEMcookies import Heaviside
from DEMcookies.plotting import plot_load_curves, render_frame

from common import FIGURES, ROOT, load_script

PEAK = 4.0


def main() -> None:
    """Fill the container, load two copies of the packing and print and plot the results."""
    example = load_script(ROOT / "examples" / "curved_indenter.py")
    sim = example.build_simulation(example.loading_history(PEAK))
    sim.rain_until_full(example.RAIN)
    sim.rest(0.1, record_every=100)
    print(f"N = {sim.n}, removed = {sim.n_removed}, bed top = {sim.bed_top():.4f} m")
    print(f"indenter edges = {sim.platens[0].edges}, direction = {sim.platens[0].direction}")

    ramp = copy.deepcopy(sim)
    ramp.platens[0].spec = dataclasses.replace(ramp.platens[0].spec, load=Heaviside(20.0, rise_time=2.0))
    ramp.compress(2.0, record_every=50)
    print(f"seating travel = {ramp.platens[0].seat_offset * 1e3:.2f} mm")
    time, applied, displacement, reaction = ramp.load_curves[0].as_array().T
    for load in (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0):
        k = min(int(np.searchsorted(applied, load)), applied.size - 1)
        print(f"F = {load:4.1f} N  u = {displacement[k] * 1e3:7.2f} mm  reaction = {reaction[k]:6.2f} N")
    print(f"removed after the ramp = {ramp.n_removed}")
    render_frame(ramp, str(FIGURES / "indenter_ramp_final.png"), -1, width=5.0)

    cycle = copy.deepcopy(sim)
    cycle.compress(1.2, record_every=50)
    time, applied, displacement, reaction = cycle.load_curves[0].as_array().T
    for target in (0.2, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2):
        k = int(np.argmin(np.abs(time - target)))
        print(f"  t = {time[k]:.3f} s  F = {applied[k]:6.2f} N  u = {displacement[k] * 1e3:7.2f} mm  "
              f"R = {reaction[k]:6.2f} N")
    hold = (time >= 0.45) & (time <= 0.7)
    print(f"hold: mean reaction = {reaction[hold].mean():.2f} N, std = {reaction[hold].std():.2f} N")
    body = cycle.platens[0]
    print(f"end: velocity = {body.velocity * 1e3:.3f} mm/s, reaction = {body.reaction:.3f} N, "
          f"weight = {body.spec.mass * cycle.settings.gravity:.3f} N")
    filled = max(k for k, frame in enumerate(cycle.frames) if frame.phase == "settling")
    render_frame(cycle, str(FIGURES / "indenter_filled.png"), filled, width=5.0)
    render_frame(cycle, str(FIGURES / "indenter_final.png"), -1, width=5.0)
    plot_load_curves(cycle, str(FIGURES / "indenter_load.png"))


if __name__ == "__main__":
    main()
