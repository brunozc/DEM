"""Effect of the shape of the grains on the 3D ballast study: spheres with rolling resistance against polyhedral stones.

Reads the results of ``ballast_cyclic_3d.py`` for spheres and for polyhedra (``--grains polyhedra``), or, with
``--preloaded``, those of ``ballast_cyclic_3d_preloaded.py``, and compares them over the cycles that every run of both
studies has completed. Run it from the repository root with::

    python examples/ballast_cyclic_3d_shapes.py
    python examples/ballast_cyclic_3d_shapes.py --preloaded

and it writes to ``output/ballast_3d_shapes/`` (``output/ballast_3d_shapes_preloaded/`` with ``--preloaded``):

* ``settlement.png``: the permanent settlement after every cycle and the resilient displacement of every case, mean
  and standard deviation over the packings, spheres against polyhedra;
* ``water_effect.png``: the settlement of each wet case less that of the dry case, packing by packing, which removes
  the scatter between packings, spheres against polyhedra;
* ``summary.txt`` and ``summary.json``: the statistics of every case for both shapes, the effect of the water for each
  shape, with a paired t-test of each wet case against the dry one, and the effect of the shape in every case,
  polyhedra less spheres. The packings of the two shapes are different, so that comparison is unpaired, with Welch's
  t-test.

Standard deviations are those of a sample, over the packings.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ballast_cyclic_3d import CASES, COLOURS, LABELS
from ballast_cyclic_3d import OUTPUT_DIRS as FIRST_DIRS
from ballast_cyclic_3d_post import (DISPLACEMENT_TOLERANCE, RunResult, _finite, case_statistics, collect_results,
                                    paired_differences, regularized_beta)
from ballast_cyclic_3d_preloaded import OUTPUT_DIRS as PRELOADED_DIRS

SHAPES = ("spheres", "polyhedra")
SHAPE_LABELS = {"spheres": "spheres, rolling resistance", "polyhedra": "polyhedra"}
SHAPE_STYLES = {"spheres": "-", "polyhedra": "--"}
OUTPUT_DIR = os.path.join("output", "ballast_3d_shapes")
PRELOADED_OUTPUT_DIR = os.path.join("output", "ballast_3d_shapes_preloaded")

Results = Dict[str, List[RunResult]]


def collect_both(directories: Dict[str, str]) -> Tuple[Dict[str, Results], List[str]]:
    """Read the runs of both shapes and cut them all to the cycles that every run has completed.

    Args:
        directories: directory of the results of each shape.

    Returns:
        The results of every case for each shape, and notes on runs that are missing or incomplete.

    Raises:
        ValueError: if the two studies were made with different loads or water models.
    """
    both: Dict[str, Results] = {}
    notes: List[str] = []
    for shape in SHAPES:
        results, shape_notes = collect_results(directories[shape])
        both[shape] = results
        notes += [f"{shape}: {note}" for note in shape_notes]
    loads = {(run.amplitude, run.frequency, run.coupled)
             for results in both.values()
             for runs in results.values()
             for run in runs}
    if len(loads) > 1:
        raise ValueError(f"The two studies differ in their load or water model (amplitude [N], frequency [Hz], "
                         f"coupled): {sorted(loads)}.")
    cycles = min(run.cycles for results in both.values() for runs in results.values() for run in runs)
    longest = max(run.cycles for results in both.values() for runs in results.values() for run in runs)
    if cycles < longest:
        notes.append(f"both shapes compared over the first {cycles} cycles")
    cut = {
        shape: {
            case: [run.truncated(cycles) for run in runs]
            for case, runs in results.items()
        }
        for shape, results in both.items()
    }
    return cut, notes


def welch(spheres: Sequence[float], polyhedra: Sequence[float]) -> Dict[str, Any]:
    """Compare the values of two independent samples, polyhedra less spheres, with Welch's t-test.

    Args:
        spheres: one value per packing of spheres.
        polyhedra: one value per packing of polyhedra.

    Returns:
        The difference of the means, its ratio to the mean of the spheres, the t statistic, the degrees of freedom of
        Welch and Satterthwaite and the two-sided p-value; t and p are NaN if a sample has fewer than two values or
        no scatter.
    """
    a = np.asarray(spheres, dtype=np.float64)
    b = np.asarray(polyhedra, dtype=np.float64)
    difference = float(b.mean() - a.mean())
    comparison: Dict[str, Any] = {
        "difference": difference,
        "relative_to_spheres_percent": 100.0 * difference / float(a.mean()) if a.mean() != 0.0 else math.nan,
        "t": math.nan,
        "dof": math.nan,
        "p": math.nan,
    }
    if a.size < 2 or b.size < 2:
        return comparison
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    if va + vb <= (DISPLACEMENT_TOLERANCE * 1e3)**2:
        return comparison
    t = difference / math.sqrt(va + vb)
    dof = (va + vb)**2 / (va * va / (a.size - 1) + vb * vb / (b.size - 1))
    comparison.update(t=t, dof=dof, p=regularized_beta(0.5 * dof, 0.5, dof / (dof + t * t)))
    return comparison


def mean_and_spread(stack: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the mean over the rows of a stack and its sample standard deviation, zero for a single row."""
    spread = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])
    return stack.mean(axis=0), spread


def plot_settlement(both: Dict[str, Results], path: str) -> None:
    """Plot the settlement and the resilient displacement of every case, spheres against polyhedra.

    Args:
        both: results of every case for each shape.
        path: output image file.
    """
    fig, (ax_settlement, ax_resilient) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for shape, results in both.items():
        for case, runs in results.items():
            cycles = np.arange(1, runs[0].cycles + 1)
            style = {"color": COLOURS[case], "linestyle": SHAPE_STYLES[shape]}
            label = f"{LABELS[case]}, {SHAPE_LABELS[shape]}"
            for ax, values in ((ax_settlement, [run.residual
                                                for run in runs]), (ax_resilient, [run.resilient for run in runs])):
                mean, spread = mean_and_spread(np.array(values) * 1e3)
                ax.fill_between(cycles, mean - spread, mean + spread, color=COLOURS[case], alpha=0.1, lw=0)
                ax.plot(cycles, mean, lw=1.8, label=label, **style)
    ax_settlement.set_ylabel("permanent settlement [mm]")
    ax_settlement.set_title("Sleeper settlement (mean and standard deviation over the packings)")
    ax_resilient.set_ylabel("resilient displacement [mm]")
    ax_resilient.set_title("Elastic rebound per cycle")
    for ax in (ax_settlement, ax_resilient):
        ax.set_xlabel("cycle")
        ax.grid(alpha=0.3)
    ax_settlement.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_water_effect(both: Dict[str, Results], path: str) -> None:
    """Plot the settlement of each wet case less that of the dry case, packing by packing, for both shapes.

    Args:
        both: results of every case for each shape.
        path: output image file.
    """
    wet_cases = [case for case in CASES if case != "dry"]
    fig, axes = plt.subplots(1, len(wet_cases), figsize=(6.0 * len(wet_cases), 4.6), squeeze=False)
    for ax, case in zip(axes[0], wet_cases):
        for shape, results in both.items():
            dry = {run.packing: run for run in results.get("dry", [])}
            pairs = [(run, dry[run.packing]) for run in results.get(case, []) if run.packing in dry]
            if not pairs:
                continue
            cycles = np.arange(1, pairs[0][0].cycles + 1)
            mean, spread = mean_and_spread(
                np.array([(wet.residual - reference.residual) * 1e3 for wet, reference in pairs]))
            ax.fill_between(cycles, mean - spread, mean + spread, color=COLOURS[case], alpha=0.12, lw=0)
            ax.plot(cycles,
                    mean,
                    color=COLOURS[case],
                    linestyle=SHAPE_STYLES[shape],
                    lw=1.8,
                    label=f"{SHAPE_LABELS[shape]} ({len(pairs)} packings)")
        ax.set_xlabel("cycle")
        ax.set_ylabel("settlement less that of the dry case [mm]")
        ax.set_title(f"{LABELS[case]} less dry")
        ax.grid(alpha=0.3)
        # a case run with neither shape leaves its panel empty
        if ax.lines:
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _line(name: str, statistics: Dict[str, Any], digits: int) -> str:
    """Format the mean and the standard deviation of a statistic, or the mean alone for a single packing."""
    spread = statistics["std"]
    text = f"{statistics['mean']:.{digits}f}"
    if not math.isnan(spread):
        text += f" +- {spread:.{digits}f}"
    return f"    {name}: {text}"


def summarise(both: Dict[str, Results], notes: Sequence[str]) -> Tuple[Dict[str, Any], str]:
    """Collect the statistics of both shapes and the effect of the shape, and write them as text.

    Args:
        both: results of every case for each shape.
        notes: notes on missing or incomplete runs.

    Returns:
        The statistics, and their text.
    """
    any_run = next(iter(both["spheres"].values()))[0]
    lines = [
        f"Shape of the grains in the 3D ballast study: {any_run.cycles} cycles of {2.0 * any_run.amplitude / 1e3:g} kN "
        f"at {any_run.frequency:g} Hz", ""
    ]
    lines += [f"note: {note}" for note in notes]
    study: Dict[str, Any] = {"cycles": any_run.cycles, "notes": list(notes), "shapes": {}, "shape_effect": {}}
    for shape, results in both.items():
        lines.append(f"{SHAPE_LABELS[shape]}:")
        shape_study: Dict[str, Any] = {"cases": {}, "water_effect": {}}
        for case, runs in results.items():
            statistics = case_statistics(runs)
            shape_study["cases"][case] = statistics
            lines.append(f"  {LABELS[case]} ({len(runs)} packings, {statistics['grains'][0]} grains in the first)")
            lines.append(_line("settlement after cycle 1 [mm]", statistics["settlement_cycle_1_mm"], 2))
            lines.append(_line(f"settlement after cycle {runs[0].cycles} [mm]", statistics["settlement_final_mm"], 2))
            lines.append(_line("rate over the second half [um/cycle]", statistics["late_rate_um_per_cycle"], 1))
            lines.append(_line("resilient displacement over the second half [mm]", statistics["resilient_mm"], 3))
            lines.append(_line("wall time of loading [h]", statistics["wall_time_h"], 1))
        for case in CASES[1:]:
            comparison = paired_differences(results, case)
            if comparison is None:
                continue
            shape_study["water_effect"][case] = comparison
            difference = comparison["difference_mm"]
            lines.append(f"  {LABELS[case]} less dry: {difference['mean']:+.2f} +- {difference['std']:.2f} mm "
                         f"({comparison['relative_to_dry_percent']:+.0f} % of dry), larger in "
                         f"{comparison['larger_than_dry']}/{difference['count']}, t = {comparison['t']:.2f}, "
                         f"p = {comparison['p']:.3g}")
        study["shapes"][shape] = shape_study
        lines.append("")

    lines.append("polyhedra less spheres (Welch's t-test):")
    for case in CASES:
        if case not in both["spheres"] or case not in both["polyhedra"]:
            continue
        effect: Dict[str, Any] = {}
        for key, name, scale in (("settlement_final_mm", "final settlement [mm]", 1e3),
                                 ("settlement_cycle_1_mm", "settlement after cycle 1 [mm]", 1e3)):
            spheres = study["shapes"]["spheres"]["cases"][case][key]["values"]
            polyhedra = study["shapes"]["polyhedra"]["cases"][case][key]["values"]
            comparison = welch(spheres, polyhedra)
            effect[key] = comparison
            lines.append(f"  {LABELS[case]}, {name}: {comparison['difference']:+.2f} "
                         f"({comparison['relative_to_spheres_percent']:+.0f} % of spheres), t = {comparison['t']:.2f}, "
                         f"p = {comparison['p']:.3g}")
        study["shape_effect"][case] = effect
    for case in CASES[1:]:
        effects = [study["shapes"][shape]["water_effect"].get(case) for shape in SHAPES]
        if any(effect is None for effect in effects):
            continue
        comparison = welch(effects[0]["difference_mm"]["values"], effects[1]["difference_mm"]["values"])
        study["shape_effect"][f"water_effect_{case}"] = comparison
        lines.append(f"  water effect of {LABELS[case]} [mm]: {comparison['difference']:+.2f}, "
                     f"t = {comparison['t']:.2f}, p = {comparison['p']:.3g}")
    return study, "\n".join(lines) + "\n"


def main(preloaded: bool = False, output_dir: str = "") -> None:
    """Read the results of both shapes and write the comparison.

    Args:
        preloaded: whether to compare the studies of ``ballast_cyclic_3d_preloaded.py`` rather than those of
            ``ballast_cyclic_3d.py``.
        output_dir: directory of the comparison; empty for the default of the study.
    """
    directories = PRELOADED_DIRS if preloaded else FIRST_DIRS
    output_dir = output_dir or (PRELOADED_OUTPUT_DIR if preloaded else OUTPUT_DIR)
    both, notes = collect_both(directories)
    os.makedirs(output_dir, exist_ok=True)
    plot_settlement(both, os.path.join(output_dir, "settlement.png"))
    plot_water_effect(both, os.path.join(output_dir, "water_effect.png"))
    study, text = summarise(both, notes)
    with open(os.path.join(output_dir, "summary.json"), "w") as stream:
        json.dump(_finite(study), stream, indent=2)
    with open(os.path.join(output_dir, "summary.txt"), "w") as stream:
        stream.write(text)
    print(text, end="")
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--preloaded",
                        action="store_true",
                        help="compare the preloaded studies of ballast_cyclic_3d_preloaded.py")
    parser.add_argument("--output-dir", default="", help="directory of the comparison (default by study)")
    arguments = parser.parse_args()
    main(arguments.preloaded, arguments.output_dir)
