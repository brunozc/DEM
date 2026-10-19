"""Consolidation results of the manual: undrained loading of a submerged column and Terzaghi's theory.

Run from the repository root with ``python manual/scripts/pore_pressure.py``. The figures are written to
``manual/figures``.
"""

import math

import numpy as np

from DEMcookies.plotting import render_frame

from common import FIGURES, ROOT, load_script


def main() -> None:
    """Run the consolidation example and print and plot its comparison with Terzaghi's theory."""
    example = load_script(ROOT / "examples" / "consolidation.py")
    sim = example.build_simulation()
    sim.rain_until_full(example.RAIN)
    sim.rest(0.2, record_every=100)
    floor_drained = -sim.edge_forces[0, 1]
    sim.set_water(example.COUPLED)
    sim.rest(0.05, record_every=100)
    volume = 4.0 / 3.0 * math.pi * sim.radius**3
    mixture = float(np.sum(volume / (1.0 - sim.porosity)))
    buoyant = float(np.sum(volume) * (sim.material.density - example.COUPLED.density) * sim.settings.gravity)
    print(f"N = {sim.n}, column height = {sim.bed_top():.4f} m, mean porosity = {np.mean(sim.porosity):.3f}, "
          f"dt = {sim.dt:.3e} s")
    print(f"floor force before loading = {floor_drained:.3f} N, buoyant weight = {buoyant:.3f} N")

    result = example.run_loading(sim)
    coefficient, pressure = example.fit_base_pressure(result)
    immediate, consolidation = example.fit_settlement(result, coefficient)
    predicted, modulus = example.predicted_consolidation_coefficient(sim, result, pressure, consolidation)
    expected = example.LOAD * result.height / mixture
    print(f"drainage length H = {result.height:.4f} m, peak base pressure = {max(result.base_pressure):.0f} Pa")
    print(f"initial excess pressure: fitted = {pressure:.0f} Pa, load over mixture area = {expected:.0f} Pa "
          f"({100 * (pressure - expected) / expected:+.1f} %)")
    print(f"settlement: immediate = {immediate * 1e3:.2f} mm, consolidation = {consolidation * 1e3:.2f} mm, "
          f"final simulated = {result.settlement[-1] * 1e3:.2f} mm; M = {modulus:.3g} Pa, "
          f"B = {1.0 / (1.0 + np.mean(sim.porosity) * modulus / example.COUPLED.bulk_modulus):.3f}")
    print(f"c_v: fitted = {coefficient:.4f} m^2/s, predicted = {predicted:.4f} m^2/s "
          f"({100 * (predicted - coefficient) / coefficient:+.1f} %)")
    time = np.array(result.time)
    for target in (0.05, 0.15, 0.3, 0.6, 1.2, 2.4, 3.0):
        k = int(np.argmin(np.abs(time - target)))
        factor = coefficient * time[k] / result.height**2
        theory = example.terzaghi(np.array([result.height]), result.height, factor)[0]
        print(f"  t = {time[k]:.2f} s  T = {factor:.3f}  base p/p0 = {result.base_pressure[k] / pressure:.3f} "
              f"(Terzaghi {theory:.3f})  floor share = {result.floor_share[k]:.3f}  "
              f"settlement = {result.settlement[k] * 1e3:.2f} mm")
    example.plot_consolidation(result, coefficient, pressure, (immediate, consolidation),
                               str(FIGURES / "consolidation.png"))
    render_frame(sim, str(FIGURES / "consolidation_column.png"), -1, width=4.0)


if __name__ == "__main__":
    main()
