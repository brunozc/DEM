"""Post-processing of the 3D ballast study: settlement of the sleeper, effect of the water and motion of the grains.

Reads the ``result.npz`` that ``ballast_cyclic_3d.py`` writes for every packing and case, also of runs that are still
going, and compares the cases over the cycles that every run has completed. Run it from the repository root with::

    python examples/ballast_cyclic_3d_post.py

and it writes to ``output/ballast_3d/post/``:

* ``settlement.png``: the permanent settlement after every cycle, the final settlement of every packing, the
  resilient displacement and the load-displacement loop of the last cycle;
* ``differences.png``: the settlement of each wet case less that of the dry case, packing by packing, which removes
  the scatter between packings;
* ``displacement_fields.png``: the vertical displacement of the grains over the loading, in a longitudinal section
  averaged along the sleeper and in a cross-section averaged across it;
* ``displacement_profiles.png``: the vertical displacement of the grains under the sleeper against their height;
* ``pore_pressure.png``: the largest excess pore pressure under the sleeper, for runs with ``--coupled`` only;
* ``summary.txt`` and ``summary.json``: the statistics of every case and of the paired differences, with a paired
  t-test of each wet case against the dry one;
* ``per_cycle.csv``: the permanent, peak and resilient settlement of every run after every cycle.

Standard deviations are those of a sample, over the packings.
"""

import argparse
import csv
import dataclasses
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from ballast_cyclic_3d import CASES, COLOURS, LABELS, LINE_STYLES, OUTPUT_DIR, per_cycle

# side of the square bins of the displacement fields, about a grain diameter, and height of the bins of the profiles
FIELD_BIN = 0.04
PROFILE_BIN = 0.025
MARKERS = {"dry": "o", "half": "s", "saturated": "^"}
# displacements below this are taken as none at all, e.g. no scatter, which leaves the t statistic undefined [m]
DISPLACEMENT_TOLERANCE = 1.0e-12


@dataclass(frozen=True, eq=False)
class RunResult:
    """Results of one packing loaded in one case, as saved by ``ballast_cyclic_3d.py``.

    Attributes:
        case: name of the case, one of CASES.
        packing: index of the packing.
        amplitude: amplitude of the rail seat load, which cycles between zero and twice it [N].
        frequency: frequency of the load [Hz].
        coupled: whether the excess pore pressure was solved.
        cycles: number of completed load cycles.
        dt: time step [s].
        wall_time: wall-clock time spent loading [s].
        n_grains: number of grains.
        water_level: height of the water table, None when dry [m].
        sleeper_base: height of the sleeper base when loading started [m].
        crib_top: top of the highest grain when loading started [m].
        buoyancy: upward force of the water on the grains at rest, before loading [N].
        uplift: upward force of the water on the sleeper base [N].
        footprint: x coordinates of the sides of the sleeper and its length from the centreline [m].
        box: width, height and length of the box [m].
        time: time since loading started at every sample [s].
        rail_seat_load: rail seat load at every sample [N].
        displacement: settlement of the sleeper at every sample [m].
        excess_pressure: largest excess pore pressure under the sleeper at every sample [Pa].
        positions: positions of the grains after the cycles at which they were kept, 0 being the start of loading
            [m].
        residual: permanent settlement after every cycle, when the load is back to zero [m].
        peak: largest settlement during every cycle [m].
    """

    case: str
    packing: int
    amplitude: float
    frequency: float
    coupled: bool
    cycles: int
    dt: float
    wall_time: float
    n_grains: int
    water_level: Optional[float]
    sleeper_base: float
    crib_top: float
    buoyancy: float
    uplift: float
    footprint: Tuple[float, float, float]
    box: Tuple[float, float, float]
    time: npt.NDArray[np.float64]
    rail_seat_load: npt.NDArray[np.float64]
    displacement: npt.NDArray[np.float64]
    excess_pressure: npt.NDArray[np.float64]
    positions: Dict[int, npt.NDArray[np.float64]]
    residual: npt.NDArray[np.float64]
    peak: npt.NDArray[np.float64]

    @property
    def resilient(self) -> npt.NDArray[np.float64]:
        """Elastic rebound of the sleeper during every cycle [m]."""
        return self.peak - self.residual

    @property
    def base_pressure(self) -> float:
        """Peak pressure of the rail seat load on the sleeper base [Pa]."""
        left, right, length = self.footprint
        return 2.0 * self.amplitude / ((right - left) * length)

    def truncated(self, cycles: int) -> "RunResult":
        """Return the results of the first cycles only.

        Args:
            cycles: number of cycles to keep, at most the number completed.

        Returns:
            The results, with the records after those cycles left out.
        """
        kept = self.time <= cycles / self.frequency + 0.5 * self.dt
        return dataclasses.replace(self,
                                   cycles=cycles,
                                   time=self.time[kept],
                                   rail_seat_load=self.rail_seat_load[kept],
                                   displacement=self.displacement[kept],
                                   excess_pressure=self.excess_pressure[kept],
                                   positions={
                                       k: v
                                       for k, v in self.positions.items() if k <= cycles
                                   },
                                   residual=self.residual[:cycles],
                                   peak=self.peak[:cycles])


def load_result(path: str) -> RunResult:
    """Read the results of one run.

    Args:
        path: file written by ``ballast_cyclic_3d.py`` [npz].

    Returns:
        The results.
    """
    with np.load(path) as data:
        cycles = int(data["cycles"])
        frequency = float(data["frequency"])
        time = data["time"]
        displacement = data["displacement"]
        residual, peak = per_cycle(time, displacement, frequency, cycles)
        level = float(data["water_level"])
        left, right, length = (float(value) for value in data["sleeper_footprint"])
        width, height, depth = (float(value) for value in data["box"])
        return RunResult(case=str(data["case"]),
                         packing=int(data["packing"]),
                         amplitude=float(data["amplitude"]),
                         frequency=frequency,
                         coupled=bool(data["coupled"]),
                         cycles=cycles,
                         dt=float(data["dt"]),
                         wall_time=float(data["wall_time"]),
                         n_grains=int(data["n_grains"]),
                         water_level=None if math.isnan(level) else level,
                         sleeper_base=float(data["sleeper_base"]),
                         crib_top=float(data["crib_top"]),
                         buoyancy=float(data["buoyancy"]),
                         uplift=float(data["uplift"]),
                         footprint=(left, right, length),
                         box=(width, height, depth),
                         time=time,
                         rail_seat_load=data["applied"] + float(data["uplift"]),
                         displacement=displacement,
                         excess_pressure=data["excess_pressure"],
                         positions={int(k): data[f"positions_{int(k)}"]
                                    for k in data["snapshot_cycles"]},
                         residual=residual,
                         peak=peak)


def collect_results(output_dir: str) -> Tuple[Dict[str, List[RunResult]], List[str]]:
    """Read every run and cut them all to the cycles that every run has completed.

    Args:
        output_dir: directory of the results of ``ballast_cyclic_3d.py``.

    Returns:
        The results of every case, sorted by packing, and notes on runs that are missing or incomplete.

    Raises:
        FileNotFoundError: if there are no results.
        ValueError: if the runs were made with different loads or water models.
    """
    paths = sorted(glob.glob(os.path.join(output_dir, "packing_*", "*", "result.npz")))
    if not paths:
        raise FileNotFoundError(f"No results in {output_dir}; run examples/ballast_cyclic_3d.py first.")
    runs = [load_result(path) for path in paths]
    loads = {(run.amplitude, run.frequency, run.coupled) for run in runs}
    if len(loads) > 1:
        raise ValueError(f"The runs in {output_dir} differ in their load or water model (amplitude [N], frequency "
                         f"[Hz], coupled): {sorted(loads)}; keep the runs of one study in one directory.")

    notes = []
    cycles = min(run.cycles for run in runs)
    longest = max(run.cycles for run in runs)
    if cycles < longest:
        unfinished = ", ".join(f"packing {run.packing} {run.case} ({run.cycles})" for run in runs
                               if run.cycles < longest)
        notes.append(f"compared over the first {cycles} cycles; runs short of {longest} cycles: {unfinished}")
    packings = sorted({run.packing for run in runs})
    found = {(run.packing, run.case) for run in runs}
    missing = [f"packing {packing} {case}" for packing in packings for case in CASES if (packing, case) not in found]
    if missing:
        notes.append(f"missing runs: {', '.join(missing)}")

    results: Dict[str, List[RunResult]] = {case: [] for case in CASES}
    for run in sorted(runs, key=lambda run: run.packing):
        results[run.case].append(run.truncated(cycles))
    return {case: case_results for case, case_results in results.items() if case_results}, notes


def regularized_beta(a: float, b: float, x: float) -> float:
    """Return the regularized incomplete beta function I_x(a, b).

    It is evaluated from its continued fraction with the modified Lentz method (Press et al., Numerical Recipes,
    section 6.4), on the side of the symmetry I_x(a, b) = 1 - I_(1-x)(b, a) where the fraction converges quickly.

    Args:
        a: first shape parameter, positive.
        b: second shape parameter, positive.
        x: upper limit of the integral, in [0, 1].

    Returns:
        The value of the function, in [0, 1].
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Return the continued fraction of the incomplete beta function, evaluated with the modified Lentz method.

    Args:
        a: first shape parameter.
        b: second shape parameter.
        x: upper limit of the integral, below (a + 1) / (a + b + 2).

    Returns:
        The value of the continued fraction.
    """
    tiny = 1.0e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    fraction = d
    for m in range(1, 300):
        # the even and the odd term of the fraction
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            fraction *= c * d
        if abs(c * d - 1.0) < 1.0e-14:
            break
    return fraction


def student_t_p_value(t: float, dof: int) -> float:
    """Return the two-sided p-value of a t statistic.

    Args:
        t: t statistic.
        dof: degrees of freedom.

    Returns:
        The probability that Student's t distribution lies at least as far from zero as t.
    """
    return regularized_beta(0.5 * dof, 0.5, dof / (dof + t * t))


def sample_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Return the mean and the sample standard deviation of some values, and the values.

    Args:
        values: one value per packing.

    Returns:
        The count, mean, standard deviation (NaN for a single value) and values.
    """
    array = np.asarray(values, dtype=np.float64)
    spread = float(array.std(ddof=1)) if array.size > 1 else math.nan
    return {"count": int(array.size), "mean": float(array.mean()), "std": spread, "values": array.tolist()}


def late_rate(result: RunResult) -> float:
    """Return the mean settlement per cycle over the second half of the cycles [m]."""
    late = result.cycles // 2
    if late < 1:
        return math.nan
    return float((result.residual[-1] - result.residual[late - 1]) / (result.cycles - late))


def case_statistics(case_results: Sequence[RunResult]) -> Dict[str, Any]:
    """Return the statistics of one case over its packings.

    Args:
        case_results: results of one case, one per packing.

    Returns:
        The statistics, in mm, um/cycle, N, Pa and h as their keys say.
    """
    late = case_results[0].cycles // 2
    levels = [result.water_level for result in case_results if result.water_level is not None]
    statistics: Dict[str, Any] = {
        "packings": [result.packing for result in case_results],
        "water_level_m": [min(levels), max(levels)] if levels else None,
        "grains": [result.n_grains for result in case_results],
        "sleeper_base_m": [result.sleeper_base for result in case_results],
        "crib_top_m": [result.crib_top for result in case_results],
        "buoyancy_N": sample_statistics([result.buoyancy for result in case_results]),
        "uplift_N": sample_statistics([result.uplift for result in case_results]),
        "settlement_cycle_1_mm": sample_statistics([result.residual[0] * 1e3 for result in case_results]),
        "settlement_final_mm": sample_statistics([result.residual[-1] * 1e3 for result in case_results]),
        "late_rate_um_per_cycle": sample_statistics([late_rate(result) * 1e6 for result in case_results]),
        "resilient_mm": sample_statistics([result.resilient[late:].mean() * 1e3 for result in case_results]),
        "wall_time_h": sample_statistics([result.wall_time / 3600.0 for result in case_results]),
    }
    if case_results[0].coupled and levels:
        statistics["excess_pressure_max_Pa"] = max(float(result.excess_pressure.max()) for result in case_results)
    return statistics


def paired_differences(results: Dict[str, List[RunResult]], case: str) -> Optional[Dict[str, Any]]:
    """Compare the final settlement of a wet case with that of the dry case, packing by packing.

    Args:
        results: results of every case.
        case: wet case to compare.

    Returns:
        The differences, case less dry, their statistics and a paired t-test of a zero mean difference; None if no
        packing was run in both cases.
    """
    dry = {result.packing: result for result in results.get("dry", [])}
    pairs = [(result, dry[result.packing]) for result in results.get(case, []) if result.packing in dry]
    if not pairs:
        return None
    differences = np.array([(wet.residual[-1] - reference.residual[-1]) * 1e3 for wet, reference in pairs])
    dry_mean = float(np.mean([reference.residual[-1] * 1e3 for _, reference in pairs]))
    comparison: Dict[str, Any] = {
        "packings": [wet.packing for wet, _ in pairs],
        "difference_mm": sample_statistics(differences.tolist()),
        "relative_to_dry_percent": 100.0 * float(differences.mean()) / dry_mean,
        "larger_than_dry": int(np.sum(differences > 0.0)),
        "t": math.nan,
        "p": math.nan,
    }
    spread = comparison["difference_mm"]["std"]
    if differences.size > 1 and spread > DISPLACEMENT_TOLERANCE * 1e3:
        t = float(differences.mean() / (spread / math.sqrt(differences.size)))
        comparison["t"] = t
        comparison["p"] = student_t_p_value(t, differences.size - 1)
    return comparison


def _style(case: str) -> Dict[str, Any]:
    """Return the colour and line style of a case."""
    return {"color": COLOURS[case], "linestyle": LINE_STYLES[case]}


def plot_settlement(results: Dict[str, List[RunResult]], path: str) -> None:
    """Plot the settlement and the resilient displacement of every case, for every packing and on average.

    Args:
        results: results of every case.
        path: output image file.
    """
    fig, ((ax_settlement, ax_final), (ax_resilient, ax_loop)) = plt.subplots(2, 2, figsize=(12.0, 8.4))
    offsets = np.linspace(-0.2, 0.2, len(results))
    for offset, (case, case_results) in zip(offsets, results.items()):
        cycles = np.arange(1, case_results[0].cycles + 1)
        residual = np.array([result.residual for result in case_results]) * 1e3
        for row in residual:
            ax_settlement.plot(cycles, row, lw=0.6, alpha=0.4, **_style(case))
        ax_settlement.plot(cycles, residual.mean(axis=0), lw=2, label=LABELS[case], **_style(case))

        packings = np.array([result.packing for result in case_results])
        ax_final.plot(packings + offset,
                      residual[:, -1],
                      MARKERS[case],
                      color=COLOURS[case],
                      markersize=8,
                      label=LABELS[case])

        resilient = np.array([result.resilient for result in case_results]) * 1e3
        mean = resilient.mean(axis=0)
        if len(case_results) > 1:
            spread = resilient.std(axis=0, ddof=1)
            ax_resilient.fill_between(cycles, mean - spread, mean + spread, color=COLOURS[case], alpha=0.15, lw=0)
        ax_resilient.plot(cycles, mean, lw=1.5, label=LABELS[case], **_style(case))

        # the load-displacement loop of the last cycle of the first packing
        first = case_results[0]
        last = first.time >= (first.cycles - 1) / first.frequency
        ax_loop.plot(first.displacement[last] * 1e3,
                     first.rail_seat_load[last] / 1e3,
                     label=f"{LABELS[case]}, packing {first.packing}",
                     **_style(case))

    any_result = next(iter(results.values()))[0]
    ax_settlement.set_xlabel("cycle")
    ax_settlement.set_ylabel("permanent settlement [mm]")
    ax_settlement.set_title("Sleeper settlement (thick: mean, thin: single packings)")
    ax_final.set_xlabel("packing")
    ax_final.set_ylabel(f"settlement after {any_result.cycles} cycles [mm]")
    ax_final.set_xticks(sorted({result.packing for case_results in results.values() for result in case_results}))
    ax_final.set_title("Final settlement of every packing")
    ax_resilient.set_xlabel("cycle")
    ax_resilient.set_ylabel("resilient displacement [mm]")
    ax_resilient.set_title("Elastic rebound per cycle (mean and standard deviation)")
    ax_loop.set_xlabel("settlement [mm]")
    ax_loop.set_ylabel("rail seat load [kN]")
    ax_loop.set_title(f"Cycle {any_result.cycles}")
    for ax in (ax_settlement, ax_final, ax_resilient, ax_loop):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_differences(results: Dict[str, List[RunResult]], path: str) -> None:
    """Plot the settlement of each wet case less that of the dry case, packing by packing.

    Args:
        results: results of every case.
        path: output image file.
    """
    fig, (ax_cycles, ax_final) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    dry = {result.packing: result for result in results.get("dry", [])}
    wet_cases = [case for case in CASES if case != "dry" and case in results]
    for offset, case in zip(np.linspace(-0.1, 0.1, len(wet_cases)), wet_cases):
        pairs = [(result, dry[result.packing]) for result in results[case] if result.packing in dry]
        if not pairs:
            continue
        cycles = np.arange(1, pairs[0][0].cycles + 1)
        differences = np.array([wet.residual - reference.residual for wet, reference in pairs]) * 1e3
        for row in differences:
            ax_cycles.plot(cycles, row, lw=0.6, alpha=0.4, **_style(case))
        ax_cycles.plot(cycles, differences.mean(axis=0), lw=2, label=f"{LABELS[case]} - dry", **_style(case))
        packings = np.array([wet.packing for wet, _ in pairs])
        ax_final.plot(packings + offset,
                      differences[:, -1],
                      MARKERS[case],
                      color=COLOURS[case],
                      markersize=8,
                      label=f"{LABELS[case]} - dry")
        if len(pairs) > 1:
            mean, spread = differences[:, -1].mean(), differences[:, -1].std(ddof=1)
            ax_final.errorbar(packings.max() + 0.8 + offset,
                              mean,
                              yerr=spread,
                              fmt=MARKERS[case],
                              color=COLOURS[case],
                              markerfacecolor="white",
                              capsize=4)
    for ax in (ax_cycles, ax_final):
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend()
    ax_cycles.set_xlabel("cycle")
    ax_cycles.set_ylabel("settlement difference to dry [mm]")
    ax_cycles.set_title("Wet less dry (thick: mean, thin: single packings)")
    ax_final.set_xlabel("packing (open markers: mean and standard deviation)")
    ax_final.set_ylabel("final settlement difference to dry [mm]")
    ax_final.set_xticks(sorted({result.packing for case_results in results.values() for result in case_results}))
    ax_final.set_title("Final difference of every packing")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def field_cycle(results: Dict[str, List[RunResult]]) -> int:
    """Return the last cycle after which the positions of the grains were kept in every run."""
    kept = set.intersection(*(set(result.positions) for case_results in results.values() for result in case_results))
    return max(kept)


def displacement_field(case_results: Sequence[RunResult], cycle: int, section: str,
                       edges: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Return the mean vertical displacement of the grains in the bins of a section, over the packings of a case.

    Every grain is binned at its position when loading started. The longitudinal section collects the grains along
    the sleeper, between the centreline and its end, and the cross-section those under it, between its sides.

    Args:
        case_results: results of one case, one per packing.
        cycle: cycle up to which the displacement is taken.
        section: ``"longitudinal"`` (bins in x and y) or ``"cross"`` (bins in z and y).
        edges: edges of the bins along the horizontal axis of the section and along y [m].

    Returns:
        The mean displacement, NaN in empty bins, shape (horizontal bins, vertical bins) [m].

    Raises:
        ValueError: if the section is unknown.
    """
    if section not in ("longitudinal", "cross"):
        raise ValueError(f"Unknown section {section!r}; expected 'longitudinal' or 'cross'.")
    sums = np.zeros((edges[0].size - 1, edges[1].size - 1))
    counts = np.zeros_like(sums)
    for result in case_results:
        start, end = result.positions[0], result.positions[cycle]
        # grains that left the domain would mismatch the two snapshots
        if start.shape != end.shape:
            continue
        left, right, length = result.footprint
        if section == "longitudinal":
            selected, horizontal = start[:, 2] <= length, 0
        else:
            selected, horizontal = (start[:, 0] >= left) & (start[:, 0] <= right), 2
        grains = start[selected]
        settlement = end[selected, 1] - grains[:, 1]
        sums += np.histogram2d(grains[:, horizontal], grains[:, 1], bins=edges, weights=settlement)[0]
        counts += np.histogram2d(grains[:, horizontal], grains[:, 1], bins=edges)[0]
    return np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)


def plot_displacement_fields(results: Dict[str, List[RunResult]], path: str) -> None:
    """Plot the vertical displacement of the grains over the loading in two sections, for every case.

    Args:
        results: results of every case.
        path: output image file.
    """
    cycle = field_cycle(results)
    reference = next(iter(results.values()))[0]
    width, _, depth = reference.box
    top = max(result.crib_top for case_results in results.values() for result in case_results)
    vertical = np.arange(0.0, top + FIELD_BIN, FIELD_BIN)
    sections = {
        "longitudinal": (np.arange(0.0, width + 0.5 * FIELD_BIN, FIELD_BIN), "x, along the track [m]"),
        "cross": (np.arange(0.0, depth + 0.5 * FIELD_BIN, FIELD_BIN), "z, from the track centreline [m]"),
    }
    titles = {
        "longitudinal": "Longitudinal section: grains along the sleeper",
        "cross": "Cross-section: grains under the sleeper",
    }
    fields = {
        (case, section): displacement_field(case_results, cycle, section, (edges, vertical)) * 1e3
        for case, case_results in results.items()
        for section, (edges, _) in sections.items()
    }
    magnitudes = np.concatenate([np.abs(field[np.isfinite(field)]) for field in fields.values()] + [np.zeros(0)])
    # the odd sparse bin would set the scale, so the largest 2 % of the bins saturate instead
    largest = float(np.percentile(magnitudes, 98)) if magnitudes.size else 0.0
    if largest <= DISPLACEMENT_TOLERANCE * 1e3:
        # a colour scale needs a range, even when nothing has moved yet
        largest = 1.0
    norm = Normalize(vmin=-largest, vmax=largest)

    row_height = 0.5 + 11.0 * top / (width + depth)
    fig, axes = plt.subplots(len(results),
                             2,
                             figsize=(13.0, 0.4 + row_height * len(results)),
                             squeeze=False,
                             sharex="col",
                             sharey=True,
                             gridspec_kw={"width_ratios": [width, depth]},
                             layout="constrained")
    mesh = None
    for row, (case, case_results) in zip(axes, results.items()):
        left, right, length = case_results[0].footprint
        base = float(np.mean([result.sleeper_base for result in case_results]))
        for ax, (section, (edges, _)) in zip(row, sections.items()):
            mesh = ax.pcolormesh(edges,
                                 vertical,
                                 np.ma.masked_invalid(fields[(case, section)].T),
                                 cmap="RdBu",
                                 norm=norm)
            low, high = (left, right) if section == "longitudinal" else (0.0, length)
            ax.add_patch(Rectangle((low, base), high - low, top - base, facecolor="0.85", edgecolor="0.3", lw=1.0))
            if case_results[0].water_level is not None:
                ax.axhline(case_results[0].water_level, color=COLOURS[case], lw=1.2, linestyle=LINE_STYLES[case])
            ax.set_aspect("equal")
        row[0].set_ylabel(f"{LABELS[case]}\ny [m]")
    for ax, (section, (_, label)) in zip(axes[0], sections.items()):
        ax.set_title(titles[section], fontsize=10)
    for ax, (_, (_, label)) in zip(axes[-1], sections.items()):
        ax.set_xlabel(label)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, shrink=0.8, extend="both", label=f"vertical displacement after cycle {cycle} [mm]")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def column_statistics(stack: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the mean and the sample standard deviation of every column, leaving out NaNs.

    Args:
        stack: one row per packing.

    Returns:
        The mean, NaN where a column has no value, and the standard deviation, NaN where it has fewer than two.
    """
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    mean = np.where(counts > 0, np.where(finite, stack, 0.0).sum(axis=0) / np.maximum(counts, 1), np.nan)
    squares = np.where(finite, (stack - mean)**2, 0.0).sum(axis=0)
    spread = np.where(counts > 1, np.sqrt(squares / np.maximum(counts - 1, 1)), np.nan)
    return mean, spread


def plot_displacement_profiles(results: Dict[str, List[RunResult]], path: str) -> None:
    """Plot the vertical displacement of the grains under the sleeper against their height, for every case.

    Args:
        results: results of every case.
        path: output image file.
    """
    cycle = field_cycle(results)
    base = max(result.sleeper_base for case_results in results.values() for result in case_results)
    edges = np.arange(0.0, base + PROFILE_BIN, PROFILE_BIN)
    heights = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for case, case_results in results.items():
        profiles = []
        for result in case_results:
            start, end = result.positions[0], result.positions[cycle]
            if start.shape != end.shape:
                continue
            left, right, length = result.footprint
            under = (start[:, 0] >= left) & (start[:, 0] <= right) & (start[:, 2] <= length)
            sums = np.histogram(start[under, 1], bins=edges, weights=end[under, 1] - start[under, 1])[0]
            counts = np.histogram(start[under, 1], bins=edges)[0]
            profiles.append(np.where(counts > 0, sums / np.where(counts > 0, counts, 1), np.nan) * 1e3)
        if not profiles:
            continue
        mean, spread = column_statistics(np.array(profiles))
        ax.fill_betweenx(heights, mean - spread, mean + spread, color=COLOURS[case], alpha=0.15, lw=0)
        ax.plot(mean, heights, lw=1.8, label=LABELS[case], **_style(case))
        level = case_results[0].water_level
        if level is not None and level < base:
            ax.axhline(level, lw=0.8, **_style(case))
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_ylim(0.0, base)
    ax.set_xlabel(f"vertical displacement after cycle {cycle} [mm]")
    ax.set_ylabel("height of the grain when loading started [m]")
    ax.set_title("Grains under the sleeper (mean and standard deviation)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pore_pressure(results: Dict[str, List[RunResult]], path: str) -> None:
    """Plot the largest excess pore pressure under the sleeper in every cycle, relative to the base pressure.

    Args:
        results: results of every case.
        path: output image file.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for case, case_results in results.items():
        if case == "dry":
            continue
        for result in case_results:
            cycle = np.clip((result.time * result.frequency).astype(np.int64), 0, result.cycles - 1)
            largest = np.zeros(result.cycles)
            np.maximum.at(largest, cycle, result.excess_pressure)
            label = LABELS[case] if result is case_results[0] else None
            ax.plot(np.arange(1, result.cycles + 1),
                    100.0 * largest / result.base_pressure,
                    lw=1.0,
                    label=label,
                    **_style(case))
    ax.set_xlabel("cycle")
    ax.set_ylabel("largest excess pore pressure [% of base pressure]")
    ax.set_title("Excess pore pressure under the sleeper")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_per_cycle(results: Dict[str, List[RunResult]], path: str) -> None:
    """Write the permanent, peak and resilient settlement of every run after every cycle.

    Args:
        results: results of every case.
        path: output file [csv], in mm.
    """
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "packing", "cycle", "permanent_mm", "peak_mm", "resilient_mm"])
        for case, case_results in results.items():
            for result in case_results:
                for cycle in range(result.cycles):
                    writer.writerow([
                        case, result.packing, cycle + 1, f"{result.residual[cycle] * 1e3:.4f}",
                        f"{result.peak[cycle] * 1e3:.4f}", f"{result.resilient[cycle] * 1e3:.4f}"
                    ])


def _finite(value: Any) -> Any:
    """Replace the NaNs of a nested structure by None, which JSON can represent."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _mean_and_spread(statistics: Dict[str, Any], digits: int) -> str:
    """Format a mean and a standard deviation."""
    if math.isnan(statistics["std"]):
        return f"{statistics['mean']:.{digits}f}"
    return f"{statistics['mean']:.{digits}f} +- {statistics['std']:.{digits}f}"


def summarise(results: Dict[str, List[RunResult]], notes: Sequence[str]) -> Tuple[Dict[str, Any], str]:
    """Collect the statistics of the study and write them as text.

    Args:
        results: results of every case.
        notes: notes on runs that are missing or incomplete.

    Returns:
        The statistics, and the same as text.
    """
    reference = next(iter(results.values()))[0]
    every = [result for case_results in results.values() for result in case_results]
    water = "coupled water (excess pore pressure solved)" if reference.coupled else "drained water"
    study: Dict[str, Any] = {
        "cycles": reference.cycles,
        "amplitude_kN": reference.amplitude / 1e3,
        "peak_load_kN": 2.0 * reference.amplitude / 1e3,
        "frequency_Hz": reference.frequency,
        "base_pressure_kPa": reference.base_pressure / 1e3,
        "coupled": reference.coupled,
        "time_step_s": reference.dt,
        "grains": [min(r.n_grains for r in every), max(r.n_grains for r in every)],
        "notes": list(notes),
        "cases": {
            case: case_statistics(case_results)
            for case, case_results in results.items()
        },
        "paired": {},
    }
    for case in CASES:
        if case != "dry":
            comparison = paired_differences(results, case)
            if comparison is not None:
                study["paired"][f"{case} - dry"] = comparison

    lines = [
        f"3D ballast study: {reference.cycles} cycles of 0 to {2.0 * reference.amplitude / 1e3:.1f} kN at "
        f"{reference.frequency:g} Hz on half a sleeper ({reference.base_pressure / 1e3:.0f} kPa peak on its base), "
        f"{water}", f"time step {reference.dt:.3e} s; {study['grains'][0]} to {study['grains'][1]} grains per packing"
    ]
    lines += [f"note: {note}" for note in notes]
    lines.append("")
    for case, statistics in study["cases"].items():
        level = statistics["water_level_m"]
        table = "none" if level is None else f"{level[0]:.3f} to {level[1]:.3f} m"
        lines.append(f"{case} ({len(statistics['packings'])} packings): water table {table}, buoyancy on the grains "
                     f"{statistics['buoyancy_N']['mean']:.0f} N, uplift on the sleeper "
                     f"{statistics['uplift_N']['mean']:.0f} N")
        lines.append(f"    settlement after cycle 1: {_mean_and_spread(statistics['settlement_cycle_1_mm'], 2)} mm, "
                     f"after cycle {reference.cycles}: {_mean_and_spread(statistics['settlement_final_mm'], 2)} mm")
        lines.append(f"    late rate: {_mean_and_spread(statistics['late_rate_um_per_cycle'], 1)} um/cycle, "
                     f"resilient: {_mean_and_spread(statistics['resilient_mm'], 3)} mm, loading took "
                     f"{statistics['wall_time_h']['mean']:.2f} h per run")
        if "excess_pressure_max_Pa" in statistics:
            pressure = statistics["excess_pressure_max_Pa"]
            lines.append(f"    largest excess pore pressure under the sleeper: {pressure:.1f} Pa, "
                         f"{100.0 * pressure / reference.base_pressure:.3f} % of the base pressure")
    lines.append("")
    for name, comparison in study["paired"].items():
        difference = comparison["difference_mm"]
        lines.append(f"{name}, final settlement, {difference['count']} packings: "
                     f"{_mean_and_spread(difference, 2)} mm ({comparison['relative_to_dry_percent']:+.1f} % of dry), "
                     f"larger than dry in {comparison['larger_than_dry']} of {difference['count']}; paired t-test "
                     f"t = {comparison['t']:.2f}, p = {comparison['p']:.3f}")
    lines.append("")
    lines.append(f"final settlement after cycle {reference.cycles} [mm]:")
    packings = sorted({result.packing for result in every})
    for packing in packings:
        values = []
        for case, case_results in results.items():
            final = [result.residual[-1] * 1e3 for result in case_results if result.packing == packing]
            values.append(f"{case} {final[0]:6.2f}" if final else f"{case}    --")
        lines.append(f"    packing {packing}: {', '.join(values)}")
    return study, "\n".join(lines) + "\n"


def main(output_dir: str = OUTPUT_DIR) -> None:
    """Read the results of the study and write its figures and statistics.

    Args:
        output_dir: directory of the results of ``ballast_cyclic_3d.py``; the figures go to its ``post`` folder.
    """
    results, notes = collect_results(output_dir)
    post = os.path.join(output_dir, "post")
    os.makedirs(post, exist_ok=True)
    plot_settlement(results, os.path.join(post, "settlement.png"))
    plot_differences(results, os.path.join(post, "differences.png"))
    plot_displacement_fields(results, os.path.join(post, "displacement_fields.png"))
    plot_displacement_profiles(results, os.path.join(post, "displacement_profiles.png"))
    if next(iter(results.values()))[0].coupled:
        plot_pore_pressure(results, os.path.join(post, "pore_pressure.png"))
    write_per_cycle(results, os.path.join(post, "per_cycle.csv"))
    study, text = summarise(results, notes)
    with open(os.path.join(post, "summary.json"), "w") as stream:
        json.dump(_finite(study), stream, indent=2)
    with open(os.path.join(post, "summary.txt"), "w") as stream:
        stream.write(text)
    print(text, end="")
    print(f"Results written to {post}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"directory of the results (default {OUTPUT_DIR})")
    main(parser.parse_args().output_dir)
