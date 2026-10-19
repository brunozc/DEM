"""Reference simulation of the manual: deposition statistics, snapshots and the 1000 N and 100 N loadings.

Run from the repository root with ``python manual/scripts/reference.py``. The packing is built once with
``main.build_simulation`` and the settings of ``main.main``, and then loaded twice, with peak loads of 1000 N and
100 N. The figures are written to ``manual/figures``.
"""

import copy
import dataclasses
from typing import Sequence

import numpy as np

from DEMcookies import DEMSimulation, Heaviside, RainSettings
from DEMcookies.plotting import plot_load_curves, render_frame

from common import FIGURES, ROOT, floor_force, load_script, particle_contacts, static_overlap, total_weight

PAUSE = 0.1
LOAD_DURATION = 1.0
RISE_TIME = 0.4


def report_packing(sim: DEMSimulation) -> None:
    """Print the properties of the deposited packing.

    Args:
        sim: simulation after deposition and settling.
    """
    radii = sim.radius * 1e3
    width = sim.geometry.bounding_box[2] - sim.geometry.bounding_box[0]
    bed = sim.bed_top()
    overlap, mean_radius = particle_contacts(sim)
    relative = overlap / mean_radius
    force, floor_contacts = floor_force(sim)
    print(f"N = {sim.n}, removed = {sim.n_removed}")
    print(f"bed height = {bed:.4f} m = {100 * bed / sim.geometry.bounding_box[3]:.1f} % of L")
    print(f"mean R = {radii.mean():.2f} mm, std R = {radii.std():.2f} mm, min {radii.min():.2f}, max {radii.max():.2f}")
    print(f"R < 6 mm: {np.sum(radii < 6.0)}, R > 18 mm: {np.sum(radii > 18.0)}")
    print(f"area fraction = {np.sum(np.pi * sim.radius**2) / (width * bed):.3f}")
    print(f"particle contacts = {overlap.size}, Z = {2.0 * overlap.size / sim.n:.2f}")
    print(f"mean overlap = {100 * relative.mean():.2f} %, max overlap = {100 * relative.max():.2f} %")
    print(f"mean kinetic energy = {sim.mean_kinetic_energy():.2e} J")
    weight = total_weight(sim)
    print(f"weight = {weight:.2f} N, floor force = {force:.2f} N ({100 * force / weight:.1f} %), "
          f"floor contacts = {floor_contacts}, mean floor contact force = {force / floor_contacts:.3f} N")
    mean_r = float(np.mean(sim.radius))
    estimate = static_overlap(sim.material, mean_r, force / floor_contacts)
    print(f"Hertz overlap of a mean floor contact on a mean particle = {estimate * 1e3:.3f} mm "
          f"= {100 * estimate / mean_r:.2f} % of mean R")


def report_loading(sim: DEMSimulation, peak: float, seated: float, sample_times: Sequence[float]) -> None:
    """Print the response of a loaded copy of the packing.

    Args:
        sim: simulation after loading.
        peak: peak load [N].
        seated: height of the platen when it was seated [m].
        sample_times: load times at which the load history is printed [s].
    """
    platen = sim.platens[0]
    curve = sim.load_curves[0].as_array()
    time, applied, displacement, reaction = curve.T
    hold = time >= RISE_TIME + 0.1
    print(f"--- peak load {peak:.0f} N")
    print(f"final reaction = {platen.reaction:.2f} N ({100 * (platen.reaction - peak) / peak:+.2f} %)")
    print(f"hold: mean reaction = {reaction[hold].mean():.2f} N, std = {reaction[hold].std():.2f} N")
    ramp = (time > 0.0) & (time <= RISE_TIME)
    print(f"ramp: largest |reaction - load| = {np.max(np.abs(reaction[ramp] - applied[ramp])):.2f} N")
    print(f"final displacement = {platen.displacement * 1e3:.2f} mm, seated bed height = {seated:.4f} m, "
          f"strain = {100 * platen.displacement / seated:.1f} %")
    print(f"secant stiffness = {peak / (platen.displacement * 1e3):.2f} N/mm")
    quarter = int(np.searchsorted(applied, 0.25 * peak))
    print(f"displacement at a quarter of the peak load = {displacement[quarter] * 1e3:.2f} mm")
    for target in sample_times:
        k = int(np.argmin(np.abs(time - target)))
        print(f"  t = {time[k]:.3f} s  F = {applied[k]:7.2f} N  u = {displacement[k] * 1e3:7.3f} mm  "
              f"R = {reaction[k]:7.2f} N")

    # contacts with the platen, whose underside is horizontal
    under = platen.starts[0, 1]
    overlap = sim.radius - (under - sim.pos[:, 1])
    touching = overlap > 0.0
    relative = overlap[touching] / sim.radius[touching]
    print(f"platen contacts = {touching.sum()}, mean overlap = {100 * relative.mean():.1f} %, "
          f"max overlap = {100 * relative.max():.1f} % of R")
    particle_overlap, mean_radius = particle_contacts(sim)
    print(f"particle contacts: mean overlap = {100 * np.mean(particle_overlap / mean_radius):.1f} %, "
          f"max overlap = {100 * np.max(particle_overlap / mean_radius):.1f} %")
    force, _ = floor_force(sim)
    carried = peak + total_weight(sim)
    print(f"floor force = {force:.1f} N = {100 * force / carried:.1f} % of load plus weight ({carried:.1f} N)")


def main() -> None:
    """Build the packing, load it with two peak loads and write the figures."""
    reference = load_script(ROOT / "main.py")
    sim = reference.build_simulation()
    print(f"dt = {sim.dt:.4e} s")
    sim.rain_until_full(RainSettings(batch_size=15, steps_between=450, record_every=60, seed=7))
    print(f"after raining: N = {sim.n}, mean kinetic energy = {sim.mean_kinetic_energy():.2e} J, t = {sim.time:.3f} s")
    sim.rest(PAUSE, record_every=60)
    report_packing(sim)

    samples = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
    for peak, name in ((1000.0, "load_displacement_1000N.png"), (100.0, "load_displacement_100N.png")):
        loaded = copy.deepcopy(sim)
        platen = loaded.platens[0]
        platen.spec = dataclasses.replace(platen.spec, load=Heaviside(peak, rise_time=RISE_TIME))
        loaded.compress(LOAD_DURATION, record_every=60)
        seated = float(platen.starts[0, 1] - platen.displacement * platen.direction[1])
        report_loading(loaded, peak, seated, samples)
        plot_load_curves(loaded, str(FIGURES / name))
        if peak == 1000.0:
            n_frames = [frame.radii.size for frame in loaded.frames]
            early = int(np.searchsorted(n_frames, 0.3 * sim.n))
            mid = int(np.searchsorted(n_frames, 0.9 * sim.n))
            print(f"snapshots: early t = {loaded.frames[early].time:.3f} s, mid t = {loaded.frames[mid].time:.3f} s")
            render_frame(loaded, str(FIGURES / "deposition_early.png"), early, width=3.5)
            render_frame(loaded, str(FIGURES / "deposition_mid.png"), mid, width=3.5)
            render_frame(loaded, str(FIGURES / "loading_final.png"), -1, width=3.5)


if __name__ == "__main__":
    main()
