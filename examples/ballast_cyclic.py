"""Cyclic loading of railway ballast under a sleeper: dry, half submerged and fully saturated.

The model is a longitudinal section through the track that holds one sleeper bay. The sleeper rests on 0.3 m of ballast
on a rigid formation, and ballast fills the cribs on both sides of it. The side walls are frictionless planes of
symmetry halfway to the next sleepers, 0.6 m apart. Ballast grains are 30 to 50 mm across. They are spheres in a
plane, so rolling resistance stands in for their angular shape; without it the sleeper punches through the bed.

The ballast is deposited dry: first the layer under the sleeper, which is then seated on it, then the cribs. The same
packing is loaded three times, dry, with the water table halfway down the ballast under the sleeper, and fully
saturated with the water at the top of the cribs. Each time the sleeper receives cycles of a 5 Hz load from zero to
2 kN, about 200 kPa on its base for a slice as thick as a grain.

Clean ballast drains almost instantly, so the water is drained: it buoys the grains below the water table and drags
on them when they move. A short coupled run of the saturated case checks that no excess pore pressure builds up.

The response of a bed of some 140 grains depends on the packing, so several packings are deposited and loaded in
parallel; only differences larger than the scatter between packings say something about the water. Run it with::

    python examples/ballast_cyclic.py

and the settlement curves, their data and snapshots of the first packing are written to ``output/``. The five
packings and three cases run as fifteen parallel processes; on sixteen cores that takes about twenty minutes.
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from DEMcookies import (BoundaryCondition, DEMSimulation, Geometry, LoadedPlaten, MaterialProperties, Open,
                        RainSettings, Sinusoidal, Wall, Water)
from DEMcookies.plotting import render_frame

OUTPUT_DIR = "output"

# track section: sleeper spacing, width of the sleeper base, ballast depth under the sleeper and box height [m]
SLEEPER_SPACING = 0.6
SLEEPER_WIDTH = 0.25
BALLAST_DEPTH = 0.3
BOX_HEIGHT = 0.65
# band in which the crib ballast is rained, which sets the height of the cribs [m]
CRIB_BAND = (0.47, 0.50)

# a 300 kg, 2.6 m long concrete sleeper, per slice as thick as a grain [kg]
SLEEPER_MASS = 4.0
PEAK_LOAD = 2000.0
FREQUENCY = 5.0
CYCLES = 50

# granite, with a Young's modulus ten times lower than the real one to keep the time step affordable
MATERIAL = MaterialProperties(density=2650.0,
                              youngs_modulus=5.0e9,
                              poisson_ratio=0.25,
                              restitution=0.5,
                              friction=0.7,
                              rolling_friction=0.3)
RADIUS_RANGE = (0.015, 0.025)

PACKINGS = 5
CASES = ("dry", "half", "saturated")
COLOURS = {"dry": "tab:brown", "half": "tab:cyan", "saturated": "tab:blue"}
LABELS = {"dry": "dry", "half": "water table at half depth", "saturated": "saturated"}

# time between rain waves, longest settling time of each rain pass, rest after flooding and recording interval [s]
WAVE_INTERVAL = 0.022
SETTLE_TIME = 0.3
REST_TIME = 0.3
RECORD_INTERVAL = 0.005
COUPLED_CYCLES = 2


@dataclass(frozen=True, eq=False)
class CaseResult:
    """Response of one packing to the cyclic load.

    Attributes:
        case: name of the case, one of CASES.
        packing: index of the packing.
        water_level: height of the water table, None when dry [m].
        sleeper_base: height of the sleeper base when loading starts [m].
        buoyancy: upward force of the water on the grains at rest, before loading [N].
        time: time since loading started at every record [s].
        displacement: settlement of the sleeper at every record [m].
        residual: settlement at the end of every cycle, when the load is back to zero [m].
        peak: largest settlement during every cycle [m].
    """

    case: str
    packing: int
    water_level: Optional[float]
    sleeper_base: float
    buoyancy: float
    time: npt.NDArray[np.float64]
    displacement: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]
    peak: npt.NDArray[np.float64]

    @property
    def resilient(self) -> npt.NDArray[np.float64]:
        """Elastic rebound of the sleeper during every cycle [m]."""
        return self.peak - self.residual


def build_simulation() -> DEMSimulation:
    """Define the sleeper bay.

    Returns:
        The simulation, with an empty bay.
    """
    left = 0.5 * (SLEEPER_SPACING - SLEEPER_WIDTH)
    right = left + SLEEPER_WIDTH
    load = Sinusoidal(amplitude=0.5 * PEAK_LOAD, frequency=FREQUENCY, mean=0.5 * PEAK_LOAD, phase=-0.5 * math.pi)
    # the sides of the sleeper reach the top of the box, so the crib ballast cannot spill over them onto the sleeper
    sleeper = LoadedPlaten(load=load, mass=SLEEPER_MASS, self_weight=True, direction=(0.0, -1.0))
    vertices = [(0.0, 0.0), (SLEEPER_SPACING, 0.0), (SLEEPER_SPACING, BOX_HEIGHT), (right, BOX_HEIGHT),
                (right, BALLAST_DEPTH), (left, BALLAST_DEPTH), (left, BOX_HEIGHT), (0.0, BOX_HEIGHT)]
    conditions: List[BoundaryCondition] = [
        Wall(), Wall(friction=0.0),
        Open(), sleeper, sleeper, sleeper,
        Open(), Wall(friction=0.0)
    ]
    return DEMSimulation(Geometry(vertices, conditions), MATERIAL, radius_range=RADIUS_RANGE)


def deposit(packing: int) -> DEMSimulation:
    """Deposit the ballast under the sleeper, seat the sleeper on it and fill the cribs.

    Args:
        packing: index of the packing, which seeds the random positions of the grains.

    Returns:
        The simulation, with the sleeper seated and the ballast at rest.
    """
    sim = build_simulation()
    steps_between = round(WAVE_INTERVAL / sim.dt)
    settle_steps = round(SETTLE_TIME / sim.dt)
    # the grains of the first pass must fit under the sleeper, which is seated on them afterwards
    under = (BALLAST_DEPTH - 2.5 * RADIUS_RANGE[1], BALLAST_DEPTH - 1.2 * RADIUS_RANGE[1])
    sim.rain_until_full(
        RainSettings(batch_size=10,
                     steps_between=steps_between,
                     insertion_band=under,
                     settle_steps=settle_steps,
                     record_every=settle_steps,
                     seed=2 * packing))
    sim.seat_platens()
    sim.rain_until_full(
        RainSettings(batch_size=6,
                     steps_between=steps_between,
                     insertion_band=CRIB_BAND,
                     settle_steps=2 * settle_steps,
                     record_every=settle_steps,
                     seed=2 * packing + 1))
    return sim


def water_level(sim: DEMSimulation, case: str) -> Optional[float]:
    """Return the height of the water table of a case, None when dry [m].

    Args:
        sim: simulation with the sleeper seated on the ballast.
        case: name of the case, one of CASES.

    Returns:
        The height of the water table.

    Raises:
        ValueError: if the case is unknown.
    """
    if case == "dry":
        return None
    if case == "half":
        return 0.5 * sleeper_base(sim)
    if case == "saturated":
        return sim.bed_top()
    raise ValueError(f"Unknown case {case!r}; expected one of {CASES}.")


def sleeper_base(sim: DEMSimulation) -> float:
    """Return the height of the sleeper base [m]."""
    return float(sim.platens[0].starts[:, 1].min())


def run_case(task: Tuple[DEMSimulation, str, int, str]) -> CaseResult:
    """Flood a deposited packing, if the case has water, and load it cyclically.

    Args:
        task: deposited simulation, name of the case, index of the packing and directory of the snapshot of the
            first packing. The simulation arrives as a copy in the worker process, so the cases of one packing start
            from the same state.

    Returns:
        The response of the sleeper.
    """
    sim, case, packing, output_dir = task
    level = water_level(sim, case)
    if level is not None:
        sim.set_water(Water(level=level))
    record_every = round(RECORD_INTERVAL / sim.dt)
    # the dry bed rests as long as the flooded ones, so all cases start loading after the same history
    sim.rest(REST_TIME, record_every=record_every)
    base = sleeper_base(sim)
    buoyancy = float(sim.fluid_force[:, 1].sum())
    period = 1.0 / FREQUENCY
    sim.compress(CYCLES * period, record_every=record_every)
    time, _, displacement, _ = sim.load_curves[0].as_array().T

    ends = period * np.arange(1, CYCLES + 1)
    residual = np.interp(ends, time, displacement)
    cycle = np.minimum((time / period).astype(np.int64), CYCLES - 1)
    peak = np.full(CYCLES, -np.inf)
    np.maximum.at(peak, cycle, displacement)
    if packing == 0:
        render_frame(sim, os.path.join(output_dir, f"ballast_{case}.png"), width=5.0)
    return CaseResult(case, packing, level, base, buoyancy, time, displacement, residual, peak)


def coupled_check(sim: DEMSimulation) -> Tuple[float, float]:
    """Load the saturated bed with the excess pore pressure solved and return its largest value under the sleeper.

    Args:
        sim: deposited simulation.

    Returns:
        The largest excess pore pressure under the sleeper and the peak pressure of the sleeper on its base [Pa].
    """
    sim.set_water(Water(level=sim.bed_top(), coupled=True))
    sim.rest(REST_TIME, record_every=round(REST_TIME / sim.dt))
    left = 0.5 * (SLEEPER_SPACING - SLEEPER_WIDTH)
    base = sleeper_base(sim)
    x, y = np.meshgrid(np.linspace(left, left + SLEEPER_WIDTH, 5), np.linspace(0.05, base - 0.03, 4))
    points = np.column_stack((x.ravel(), y.ravel()))

    largest = 0.0
    chunk = RECORD_INTERVAL
    for _ in range(round(COUPLED_CYCLES / FREQUENCY / chunk)):
        sim.compress(chunk, record_every=round(chunk / sim.dt))
        largest = max(largest, float(np.max(np.abs(sim.excess_pore_pressure(points)))))

    # the slice is as thick as the volume of the mixture of grains and pores over the area of the bed
    mixture = float(np.sum(4.0 / 3.0 * math.pi * sim.radius**3 / (1.0 - sim.porosity)))
    area = SLEEPER_SPACING * sim.bed_top() - SLEEPER_WIDTH * (sim.bed_top() - base)
    return largest, PEAK_LOAD / (SLEEPER_WIDTH * mixture / area)


def mean_and_spread(results: Sequence[CaseResult],
                    quantity: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the mean and the standard deviation over the packings of a per-cycle quantity.

    Args:
        results: results of one case, one per packing.
        quantity: name of the per-cycle attribute of CaseResult.

    Returns:
        The mean and the standard deviation of every cycle [m].
    """
    values = np.array([getattr(result, quantity) for result in results])
    return values.mean(axis=0), values.std(axis=0)


def plot_settlement(results: Dict[str, List[CaseResult]], path: str) -> None:
    """Plot the settlement and the resilient displacement of every case, for every packing and on average.

    Args:
        results: results of every case, one per packing.
        path: output image file.
    """
    fig, ((ax_settlement, ax_final), (ax_resilient, ax_loop)) = plt.subplots(2, 2, figsize=(12.0, 8.4))
    cycles = np.arange(1, CYCLES + 1)
    offsets = np.linspace(-0.2, 0.2, len(results))
    for offset, (case, case_results) in zip(offsets, results.items()):
        colour = COLOURS[case]
        for result in case_results:
            ax_settlement.plot(cycles, result.residual * 1e3, color=colour, lw=0.6, alpha=0.4)
        mean, _ = mean_and_spread(case_results, "residual")
        ax_settlement.plot(cycles, mean * 1e3, color=colour, lw=2, label=LABELS[case])

        packings = np.array([result.packing for result in case_results])
        final = np.array([result.residual[-1] for result in case_results])
        ax_final.plot(packings + offset, final * 1e3, "o", color=colour, label=LABELS[case])

        mean, spread = mean_and_spread(case_results, "resilient")
        ax_resilient.fill_between(cycles, (mean - spread) * 1e3, (mean + spread) * 1e3, color=colour, alpha=0.15)
        ax_resilient.plot(cycles, mean * 1e3, color=colour, lw=1.5, label=LABELS[case])

        # the load-displacement loop of the last cycle of the first packing
        first = case_results[0]
        last = first.time >= (CYCLES - 1) / FREQUENCY
        load = 0.5 * PEAK_LOAD * (1.0 - np.cos(2.0 * math.pi * FREQUENCY * first.time[last]))
        ax_loop.plot(first.displacement[last] * 1e3, load / 1e3, color=colour, label=LABELS[case])

    ax_settlement.set_xlabel("cycle")
    ax_settlement.set_ylabel("permanent settlement [mm]")
    ax_settlement.set_title("Sleeper settlement (thick: mean, thin: single packings)")
    ax_final.set_xlabel("packing")
    ax_final.set_ylabel(f"settlement after {CYCLES} cycles [mm]")
    ax_final.set_xticks(range(PACKINGS))
    ax_final.set_title("Final settlement of every packing")
    ax_resilient.set_xlabel("cycle")
    ax_resilient.set_ylabel("resilient displacement [mm]")
    ax_resilient.set_title("Elastic rebound per cycle (mean and standard deviation)")
    ax_loop.set_xlabel("settlement [mm]")
    ax_loop.set_ylabel("applied load [kN]")
    ax_loop.set_title("Last cycle of packing 0")
    for ax in (ax_settlement, ax_final, ax_resilient, ax_loop):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_results(results: Dict[str, List[CaseResult]], path: str) -> None:
    """Save the per-cycle settlement and resilient displacement of every case and packing.

    Args:
        results: results of every case, one per packing.
        path: output file [npz]; arrays are named ``<case>_<quantity>``, with one row per packing [m].
    """
    # typed loosely because the numpy stub of savez mistakes keyword arrays for its allow_pickle flag
    arrays: Dict[str, Any] = {}
    for case, case_results in results.items():
        arrays[f"{case}_residual"] = np.array([result.residual for result in case_results])
        arrays[f"{case}_peak"] = np.array([result.peak for result in case_results])
        arrays[f"{case}_sleeper_base"] = np.array([result.sleeper_base for result in case_results])
        levels = [np.nan if result.water_level is None else result.water_level for result in case_results]
        arrays[f"{case}_water_level"] = np.array(levels)
    np.savez(path, **arrays)


def main(output_dir: str = OUTPUT_DIR) -> None:
    """Deposit the packings, load every case and write the results.

    Args:
        output_dir: directory of the figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, PACKINGS * len(CASES) + 1)) as pool:
        packings = list(pool.map(deposit, range(PACKINGS)))
        for packing, sim in enumerate(packings):
            print(f"packing {packing}: {sim.n} grains, ballast under the sleeper {sleeper_base(sim):.3f} m, "
                  f"crib top {sim.bed_top():.3f} m")
        check = pool.submit(coupled_check, packings[0])
        tasks = [(sim, case, packing, output_dir) for packing, sim in enumerate(packings) for case in CASES]
        results: Dict[str, List[CaseResult]] = {case: [] for case in CASES}
        for result in pool.map(run_case, tasks):
            results[result.case].append(result)
        pressure, base_pressure = check.result()

    print(f"time step = {packings[0].dt:.3e} s, {CYCLES} cycles of 0 to {PEAK_LOAD / 1e3:.1f} kN at "
          f"{FREQUENCY:.0f} Hz")
    late = CYCLES // 2
    for case, case_results in results.items():
        levels = [r.water_level for r in case_results]
        level = "none" if levels[0] is None else ", ".join(f"{h:.3f}" for h in levels if h is not None) + " m"
        residual = np.array([r.residual for r in case_results])
        rate = (residual[:, -1] - residual[:, late - 1]) / (CYCLES - late)
        resilient = np.array([r.resilient[late:].mean() for r in case_results])
        buoyancy = np.mean([r.buoyancy for r in case_results])
        print(f"{case:>9}: water table {level}; buoyancy = {buoyancy:.1f} N "
              f"({100.0 * buoyancy / PEAK_LOAD:.1f} % of the peak load); settlement after cycle 1 = "
              f"{residual[:, 0].mean() * 1e3:.2f} +- {residual[:, 0].std() * 1e3:.2f} mm, after cycle {CYCLES} = "
              f"{residual[:, -1].mean() * 1e3:.2f} +- {residual[:, -1].std() * 1e3:.2f} mm; late rate = "
              f"{rate.mean() * 1e6:.1f} um/cycle; resilient = {resilient.mean() * 1e3:.3f} mm")
    for packing in range(PACKINGS):
        final = ", ".join(f"{case} {results[case][packing].residual[-1] * 1e3:5.2f}" for case in CASES)
        print(f"  packing {packing}: settlement after cycle {CYCLES} [mm]: {final}")
    print(f"coupled check, saturated: largest excess pore pressure under the sleeper = {pressure:.1f} Pa, "
          f"{100.0 * pressure / base_pressure:.3f} % of the {base_pressure / 1e3:.0f} kPa on the sleeper base")

    plot_settlement(results, os.path.join(output_dir, "ballast_settlement.png"))
    save_results(results, os.path.join(output_dir, "ballast_settlement.npz"))
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    main()
