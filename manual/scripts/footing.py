"""Bearing test of the strip footing example: a slow ramp to 20 N over 2 s.

Run from the repository root with ``python manual/scripts/footing.py``.
"""

import numpy as np

from DEMcookies import Heaviside
from DEMcookies.plotting import render_frame

from common import FIGURES, ROOT, load_script


def main() -> None:
    """Fill the trapezoid, ramp the footing load and print the settlement at a series of loads."""
    example = load_script(ROOT / "examples" / "footing_on_trapezoid.py")
    sim = example.build_simulation(Heaviside(20.0, rise_time=2.0))
    sim.rain_until_full(example.RAIN)
    print(f"N = {sim.n}, bed top = {sim.bed_top():.4f} m")
    sim.rest(0.1, record_every=100)
    sim.compress(2.0, record_every=50)
    print(f"seating travel = {sim.platens[0].seat_offset * 1e3:.2f} mm")
    time, applied, displacement, reaction = sim.load_curves[0].as_array().T
    for load in (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
        k = int(np.searchsorted(applied, load))
        print(f"F = {load:4.1f} N  u = {displacement[k] * 1e3:7.2f} mm  reaction = {reaction[k]:6.2f} N")
    print(f"footing weight = {sim.platens[0].spec.mass * sim.settings.gravity:.3f} N")
    render_frame(sim, str(FIGURES / "footing_ramp_final.png"), -1, width=5.0)


if __name__ == "__main__":
    main()
