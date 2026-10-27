"""Cyclic loading of a railway sleeper on 3D ballast that is first preloaded dry, then flooded and loaded again.

The follow-up of ``ballast_cyclic_3d.py``. There the sleeper was seated on the ballast and the bed flooded before the
first load, and the water added to the settlement only while the sleeper bedded in, in the first cycle. This study
tests whether the water still matters once the ballast is bedded. Every packing deposited by ``ballast_cyclic_3d.py``
is

* preloaded dry: it is loaded as in the dry case of ``ballast_cyclic_3d.py``, from the same deposit, with the same rest
  before the first load and the same load, by default for 50 cycles, so the preload repeats that run step for step
  when both run on the same device;
* flooded at the end of the last cycle of the preload, when the rail seat is unloaded: the water rises at once to the
  level of the case, and the load stays at its trough for 0.3 s, as long as the rest before the first load of
  ``ballast_cyclic_3d.py``. The sleeper stays free on the ballast meanwhile, under its own weight less the uplift of
  the water;
* loaded again, by default for 50 cycles, in the three cases of that study: dry, with the water table halfway down the
  ballast under the sleeper, and saturated. The water table is set from the preloaded bed: at half the height of the
  sleeper base, which has settled meanwhile, or at the top of the ballast.

All three cases of a packing start from the same preloaded bed, so they can be compared grain by grain, as in
``ballast_cyclic_3d.py``. The dry case rests through the same pause as the others, so all cases share one history.

Like ``ballast_cyclic_3d.py``, the study is split into jobs that a cluster can run in parallel, and that resume where
they stopped if they are killed. It reads the deposits of that study, so deposit the packings first. From the
repository root::

    python examples/ballast_cyclic_3d_preloaded.py preload --packing 0
    python examples/ballast_cyclic_3d_preloaded.py load --packing 0 --case saturated

``preload`` loads one deposited packing dry and stores the preloaded bed; ``load`` floods it in one case and loads it
again, saves a checkpoint after every cycle and continues from it when it is run again, also to add cycles to a
finished run. Both take the load (``--amplitude``, ``--frequency``) and the length of the preload
(``--preload-cycles``), which must be the same in both. ``--task`` works as in ``ballast_cyclic_3d.py``, and ``all``
runs every job on one machine; ``ballast_cyclic_3d_preloaded.sbatch`` submits the jobs to a SLURM cluster.

Every case writes to ``output/ballast_3d_preloaded/packing_<k>/<case>/`` a ``result.npz`` in the format of
``ballast_cyclic_3d.py``, so that::

    python examples/ballast_cyclic_3d_post.py --output-dir output/ballast_3d_preloaded

compares the cases after the flooding. In it the time runs from the end of the pause, when the load cycles again, and
is negative during the pause; the settlement and the positions of the grains count from the flooding, so they include
what the bed settles while it is flooded, which is also stored as ``flood_settlement``. It also stores
``preload_cycles`` and ``preload_settlement``, the settlement of the sleeper over the preload. The preload writes its
own records to ``packing_<k>/preload/history.npz``, out of the way of the post-processing, and the preloaded bed to
``packing_<k>/preloaded.pkl``, which every case of the packing starts from. ParaView files are written as in
``ballast_cyclic_3d.py``, to ``preload/paraview/`` and ``<case>/paraview/``.

A preload of 50 cycles takes about five hours on one core, and a case about five and a half, so the whole study takes
some eleven hours on a cluster that runs the five preloads and then the fifteen cases in parallel.

``--grains polyhedra``, given to every command, runs the study on the polyhedral stones that
``ballast_cyclic_3d.py --grains polyhedra`` deposits: it reads them from ``output/ballast_3d_polyhedra`` and writes to
``output/ballast_3d_polyhedra_preloaded``. The stored packings carry their grains, so nothing else changes. A
polyhedral cycle costs as much as in ``ballast_cyclic_3d.py``, so pass ``--device gpu`` here too.
"""

import argparse
import dataclasses
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ballast_cyclic_3d import (AMPLITUDE, CASES, CYCLES, FREQUENCY, GRAINS, PACKINGS, POLYHEDRA_SAVE_EVERY_N_STEPS,
                               RAIL_SEAT, REST_TIME, SAMPLES_PER_CYCLE, SAVE_EVERY_N_STEPS, SNAPSHOT_CYCLES, LoadRun,
                               LoadSettings, case_dir, load_pickle, packing_dir, paraview_writer, per_cycle,
                               pressure_points, record_sample, resume_paraview, save_pickle, save_results, sleeper_base,
                               sleeper_uplift, start_run, water_level)
from ballast_cyclic_3d import OUTPUT_DIR as DEPOSIT_DIR
from ballast_cyclic_3d import OUTPUT_DIRS as DEPOSIT_DIRS
from DEMcookies import DEMSimulation, Water
from DEMcookies.backend import DEVICES
from DEMcookies.plotting import render_frame
from DEMcookies.vtk_export import VTKWriter

OUTPUT_DIR = os.path.join("output", "ballast_3d_preloaded")
# directory of the results for each shape of the grains
OUTPUT_DIRS = {"spheres": OUTPUT_DIR, "polyhedra": os.path.join("output", "ballast_3d_polyhedra_preloaded")}

# cycles of the dry preload: the first cycle sets most of the bedding-in, and by the 50th the rate of settlement has
# dropped tenfold in ballast_cyclic_3d.py
PRELOAD_CYCLES = 50
# records of the sleeper while the bed is flooded, at the rate of SAMPLES_PER_CYCLE at the default frequency
FLOOD_SAMPLES = 60


@dataclass(frozen=True)
class FloodedLoad:
    """Load on the sleeper from the flooding on: the rail seat load held at its trough, then cycling again.

    ``F = A (1 - cos 2 pi f (t - t_r)) - U`` from the resume time ``t_r`` on, and ``-U`` before it, where ``U`` is the
    uplift of the water on the sleeper base. It is a class rather than a closure so that it can be pickled with the
    checkpoints.

    Attributes:
        amplitude: amplitude of the rail seat load, which cycles between zero and twice it [N].
        frequency: frequency of the load [Hz].
        resume_time: load time at which the rail seat load cycles again [s].
        uplift: upward force of the water on the sleeper base [N].
    """

    amplitude: float
    frequency: float
    resume_time: float
    uplift: float

    def __call__(self, time: float) -> float:
        """Evaluate the load.

        Args:
            time: time since loading started [s].

        Returns:
            The load on the sleeper, positive downwards [N].
        """
        elapsed = max(time - self.resume_time, 0.0)
        return self.amplitude * (1.0 - math.cos(2.0 * math.pi * self.frequency * elapsed)) - self.uplift


@dataclass(eq=False)
class StagedRun:
    """The preload of a packing, or one case after it, checkpointed after every cycle.

    Attributes:
        load: the run, whose cycles, records and snapshots count from the origins below.
        preload_cycles: number of cycles of the dry preload.
        time_origin: load time at which the cycles of the run start: zero for the preload, the end of the pause for
            a case [s].
        displacement_origin: settlement of the sleeper from which the records count: zero for the preload, that at the
            flooding, i.e. the settlement over the preload, for a case [m].
    """

    load: LoadRun
    preload_cycles: int
    time_origin: float = 0.0
    displacement_origin: float = 0.0


def preload_dir(output_dir: str, packing: int) -> str:
    """Return the directory of the preload of a packing."""
    return os.path.join(packing_dir(output_dir, packing), "preload")


def preloaded_path(output_dir: str, packing: int) -> str:
    """Return the file of the preloaded bed of a packing."""
    return os.path.join(packing_dir(output_dir, packing), "preloaded.pkl")


def preload_settings(settings: LoadSettings) -> LoadSettings:
    """Return the settings of the preload of a study, which is always dry."""
    return dataclasses.replace(settings, coupled=False)


def check_preload(run: StagedRun, settings: LoadSettings, cycles: int, path: str) -> None:
    """Check that a stored preload was made with the load and the length of a study.

    Args:
        run: stored preload.
        settings: load of the study.
        cycles: number of cycles of the preload of the study.
        path: file the preload was read from, for the error message.

    Raises:
        ValueError: if the preload was made with another load or another number of cycles.
    """
    expected = preload_settings(settings)
    if run.load.settings != expected or run.preload_cycles != cycles:
        raise ValueError(f"{path} holds a preload of {run.preload_cycles} cycles with {run.load.settings}, not of "
                         f"{cycles} cycles with {expected}; choose another output directory to change the preload.")


def record(run: StagedRun) -> None:
    """Append the current state of the sleeper to a run, counted from the origins of its records."""
    record_sample(run.load)
    run.load.time[-1] -= run.time_origin
    run.load.displacement[-1] -= run.displacement_origin


def advance_to(run: StagedRun, load_time: float) -> None:
    """Advance a run to a load time, to the nearest time step, and record it.

    Args:
        run: run to advance.
        load_time: load time to reach [s].
    """
    sim = run.load.sim
    steps = max(1, round((load_time - sim.load_time) / sim.dt))
    # one frame and one load record per call; only the latest frame is kept
    sim.compress(steps * sim.dt, record_every=steps)
    record(run)
    run.load.last_frame = sim.frames[-1]
    sim.frames.clear()


def load_cycles(run: StagedRun,
                cycles: int,
                label: str,
                checkpoint: str,
                results: str,
                extra: Optional[Dict[str, npt.NDArray[Any]]] = None) -> None:
    """Load a run up to a number of cycles, checkpointing it and saving its results after every cycle.

    Args:
        run: run to load.
        cycles: number of cycles to reach.
        label: name of the run in the progress messages.
        checkpoint: file of the checkpoint [pkl].
        results: file of the results [npz].
        extra: further arrays to store with the results, by name.
    """
    load = run.load
    sim = load.sim
    period = 1.0 / load.settings.frequency
    started = perf_counter()
    first_cycle = load.cycles_done
    while load.cycles_done < cycles:
        cycle_start = perf_counter()
        for sample in range(SAMPLES_PER_CYCLE):
            # every sample is aimed at its own phase of the load, so rounding to whole steps never accumulates
            advance_to(run, run.time_origin + (load.cycles_done + (sample + 1) / SAMPLES_PER_CYCLE) * period)
        load.cycles_done += 1
        load.wall_time += perf_counter() - cycle_start
        if load.cycles_done in SNAPSHOT_CYCLES:
            load.snapshots[load.cycles_done] = sim.pos.copy()
        save_pickle(run, checkpoint)
        save_results(load, results, extra)

        residual, peak = per_cycle(np.array(load.time), np.array(load.displacement), load.settings.frequency,
                                   load.cycles_done)
        elapsed = perf_counter() - started
        remaining = elapsed / (load.cycles_done - first_cycle) * (cycles - load.cycles_done)
        print(
            f"{label}: cycle {load.cycles_done}/{cycles}, settlement {residual[-1] * 1e3:.2f} mm, "
            f"peak {peak[-1] * 1e3:.2f} mm; {elapsed / 60:.1f} min, about {remaining / 60:.0f} min to go",
            flush=True)


def preload(packing: int,
            settings: LoadSettings,
            cycles: int,
            deposit_dir: str,
            output_dir: str,
            save_every_n_steps: int = SAVE_EVERY_N_STEPS,
            device: str = "cpu") -> str:
    """Load a deposited packing dry for a number of cycles, resuming from its checkpoint, and store the bed.

    A packing that is already preloaded is left alone, so that a study can be resumed.

    Args:
        packing: index of the packing.
        settings: load of the study; the preload is dry even if the cases are coupled.
        cycles: number of cycles of the preload.
        deposit_dir: directory of the results of ``ballast_cyclic_3d.py``, which holds the deposited packings.
        output_dir: directory of the results of this study.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu".

    Returns:
        A summary of the preload.

    Raises:
        FileNotFoundError: if the packing has not been deposited.
        ValueError: if the packing was preloaded, or its preload started, with another load or number of cycles.
    """
    path = preloaded_path(output_dir, packing)
    if os.path.exists(path):
        check_preload(load_pickle(path, StagedRun), settings, cycles, path)
        return f"packing {packing}: already preloaded in {path}"
    directory = preload_dir(output_dir, packing)
    checkpoint = os.path.join(directory, "checkpoint.pkl")
    if os.path.exists(checkpoint):
        run = load_pickle(checkpoint, StagedRun)
        run.load.sim.device = device
        check_preload(run, settings, cycles, checkpoint)
        resume_paraview(run.load, directory, save_every_n_steps)
        print(f"packing {packing}, preload: resumed after cycle {run.load.cycles_done}", flush=True)
    else:
        source = os.path.join(packing_dir(deposit_dir, packing), "deposit.pkl")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Packing {packing} has not been deposited: {source} does not exist; run "
                                    f"examples/ballast_cyclic_3d.py deposit first.")
        os.makedirs(directory, exist_ok=True)
        deposited = load_pickle(source, DEMSimulation)
        deposited.device = device
        # the dry case of ballast_cyclic_3d.py, so the preload repeats the first cycles of that run
        load = start_run(deposited, "dry", packing, preload_settings(settings),
                         paraview_writer(directory, save_every_n_steps))
        run = StagedRun(load, cycles)
        save_pickle(run, checkpoint)

    load_cycles(run, cycles, f"packing {packing}, preload", checkpoint, os.path.join(directory, "history.npz"))
    # the cases write ParaView files of their own, so the preloaded bed must not take this writer along
    if run.load.paraview is not None:
        run.load.sim.remove_observer(run.load.paraview)
        run.load.paraview = None
    run.load.last_frame = None
    save_pickle(run, path)
    return (f"packing {packing}: settlement after {cycles} dry cycles {run.load.displacement[-1] * 1e3:.2f} mm, "
            f"{run.load.wall_time / 3600:.2f} h of loading; preloaded bed in {path}")


def flood(preloaded: StagedRun, case: str, settings: LoadSettings, paraview: Optional[VTKWriter] = None) -> StagedRun:
    """Flood a preloaded bed to the water table of a case while the rail seat is unloaded, and let it rest.

    The bed is flooded at the end of the last cycle of the preload, when the rail seat load is at its trough. The load
    stays there for the pause, and then cycles again from its trough; the sleeper stays free on the ballast meanwhile.

    Args:
        preloaded: preloaded bed, which is flooded in place.
        case: name of the case, one of CASES.
        settings: load of the run, and whether its water is coupled.
        paraview: writer of the ParaView files of the run, from the flooding on; None writes none.

    Returns:
        The run, before its first cycle after the flooding.
    """
    sim = preloaded.load.sim
    if paraview is not None:
        sim.add_observer(paraview)
    platen = sim.platens[0]
    flooded = sim.load_time
    load = LoadRun(sim=sim,
                   case=case,
                   packing=preloaded.load.packing,
                   settings=settings,
                   water_level=water_level(sim, case),
                   sleeper_base=sleeper_base(sim),
                   crib_top=sim.bed_top(),
                   buoyancy=0.0,
                   uplift=0.0,
                   pressure_points=pressure_points(sim),
                   paraview=paraview)
    run = StagedRun(load,
                    preloaded.preload_cycles,
                    time_origin=flooded + REST_TIME,
                    displacement_origin=platen.displacement)
    # the last record before the water comes in
    record(run)
    load.snapshots[0] = sim.pos.copy()
    if load.water_level is not None:
        sim.set_water(Water(level=load.water_level, coupled=settings.coupled))
    load.uplift = sleeper_uplift(sim, load.water_level)
    platen.spec = dataclasses.replace(platen.spec,
                                      load=FloodedLoad(settings.amplitude, settings.frequency, run.time_origin,
                                                       load.uplift))
    for sample in range(FLOOD_SAMPLES):
        advance_to(run, flooded + (sample + 1) / FLOOD_SAMPLES * REST_TIME)
    # the force of the water on the grains at rest, as in ballast_cyclic_3d.py
    load.buoyancy = float(sim.fluid_force[:, 1].sum())
    return run


def run_case(packing: int,
             case: str,
             settings: LoadSettings,
             preload_cycles: int,
             cycles: int,
             output_dir: str,
             save_every_n_steps: int = SAVE_EVERY_N_STEPS,
             device: str = "cpu") -> str:
    """Flood a preloaded packing in one case and load it again, resuming from its checkpoint if there is one.

    Args:
        packing: index of the packing.
        case: name of the case, one of CASES.
        settings: load of the run.
        preload_cycles: number of cycles of the dry preload.
        cycles: number of load cycles after the flooding.
        output_dir: directory of the results.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu".

    Returns:
        A summary of the run.

    Raises:
        FileNotFoundError: if the packing has not been preloaded.
        ValueError: if a checkpoint of the run was made with another load or another preload.
        ValueError: if the packing was preloaded with another load or number of cycles.
    """
    directory = case_dir(output_dir, packing, case)
    checkpoint = os.path.join(directory, "checkpoint.pkl")
    label = f"packing {packing}, {case}"
    if os.path.exists(checkpoint):
        run = load_pickle(checkpoint, StagedRun)
        run.load.sim.device = device
        if run.load.settings != settings or run.preload_cycles != preload_cycles:
            raise ValueError(f"The checkpoint {checkpoint} was made with {run.load.settings} after a preload of "
                             f"{run.preload_cycles} cycles, not with {settings} after {preload_cycles}; remove it, or "
                             f"choose another output directory, to start the run again.")
        resume_paraview(run.load, directory, save_every_n_steps)
        print(f"{label}: resumed after cycle {run.load.cycles_done}", flush=True)
    else:
        source = preloaded_path(output_dir, packing)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Packing {packing} has not been preloaded: {source} does not exist.")
        preloaded = load_pickle(source, StagedRun)
        preloaded.load.sim.device = device
        check_preload(preloaded, settings, preload_cycles, source)
        os.makedirs(directory, exist_ok=True)
        run = flood(preloaded, case, settings, paraview_writer(directory, save_every_n_steps))
        save_pickle(run, checkpoint)
        print(f"{label}: flooded, settlement while flooded {run.load.displacement[-1] * 1e3:.3f} mm", flush=True)

    # the settlement at the end of the pause, which the records of the run count from the flooding
    flood_settlement = float(np.interp(0.0, run.load.time, run.load.displacement))
    extra = {
        "preload_cycles": np.array(run.preload_cycles),
        "preload_settlement": np.array(run.displacement_origin),
        "flood_settlement": np.array(flood_settlement),
        "pause": np.array(REST_TIME),
    }
    results = os.path.join(directory, "result.npz")
    load_cycles(run, cycles, label, checkpoint, results, extra)
    save_results(run.load, results, extra)
    sim = run.load.sim
    if run.load.last_frame is not None:
        sim.frames[:] = [run.load.last_frame]
        render_frame(sim, os.path.join(directory, "final.png"), width=7.0)
        render_frame(sim, os.path.join(directory, "final_section.png"), width=6.0, section=RAIL_SEAT)
        sim.frames.clear()
    residual, _ = per_cycle(np.array(run.load.time), np.array(run.load.displacement), settings.frequency,
                            run.load.cycles_done)
    return (f"{label}: after a preload of {run.displacement_origin * 1e3:.2f} mm, settlement while flooded "
            f"{flood_settlement * 1e3:.3f} mm, after {run.load.cycles_done} more cycles {residual[-1] * 1e3:.2f} mm, "
            f"{run.load.wall_time / 3600:.2f} h of loading; results in {directory}")


def run_all(packings: int,
            settings: LoadSettings,
            preload_cycles: int,
            cycles: int,
            deposit_dir: str,
            output_dir: str,
            workers: int,
            save_every_n_steps: int = SAVE_EVERY_N_STEPS,
            device: str = "cpu") -> None:
    """Preload every packing and load it in every case, in parallel processes on one machine.

    Args:
        packings: number of packings.
        settings: load of the runs.
        preload_cycles: number of cycles of the dry preload.
        cycles: number of load cycles after the flooding.
        deposit_dir: directory of the results of ``ballast_cyclic_3d.py``, which holds the deposited packings.
        output_dir: directory of the results of this study.
        workers: number of parallel processes.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu".
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        preloads = [
            pool.submit(preload, packing, settings, preload_cycles, deposit_dir, output_dir, save_every_n_steps, device)
            for packing in range(packings)
        ]
        for job in preloads:
            print(job.result(), flush=True)
        runs = [
            pool.submit(run_case, packing, case, settings, preload_cycles, cycles, output_dir, save_every_n_steps,
                        device) for packing in range(packings) for case in CASES
        ]
        for job in runs:
            print(job.result(), flush=True)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: arguments; None reads those of the process.

    Returns:
        The arguments, with ``packing`` and, for ``load``, ``case`` resolved from ``task``.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--grains",
                        choices=GRAINS,
                        default="spheres",
                        help="shape of the grains of the deposits, which sets the default directories (default "
                        "spheres)")
    common.add_argument("--output-dir",
                        help=f"directory of the results (default {OUTPUT_DIRS['spheres']}, or "
                        f"{OUTPUT_DIRS['polyhedra']} for polyhedra)")
    common.add_argument("--device",
                        choices=DEVICES,
                        default="cpu",
                        help="device that measures the overlaps of polyhedra; gpu needs CuPy (default cpu)")
    common.add_argument("--save-every-n-steps",
                        type=int,
                        help=f"time steps between two ParaView exports, 0 for none (default {SAVE_EVERY_N_STEPS}, or "
                        f"{POLYHEDRA_SAVE_EVERY_N_STEPS} for polyhedra)")
    common.add_argument("--amplitude",
                        type=float,
                        default=AMPLITUDE / 1e3,
                        help=f"amplitude of the rail seat load [kN], which cycles between zero and twice it "
                        f"(default {AMPLITUDE / 1e3:g})")
    common.add_argument("--frequency",
                        type=float,
                        default=FREQUENCY,
                        help=f"frequency of the load [Hz] (default {FREQUENCY:g})")
    common.add_argument("--preload-cycles",
                        type=int,
                        default=PRELOAD_CYCLES,
                        help=f"number of dry load cycles before the flooding (default {PRELOAD_CYCLES})")
    deposits = argparse.ArgumentParser(add_help=False)
    deposits.add_argument("--deposit-dir",
                          help=f"directory of the results of ballast_cyclic_3d.py, with the deposited packings "
                          f"(default {DEPOSIT_DIR}, or {DEPOSIT_DIRS['polyhedra']} for polyhedra)")
    flooding = argparse.ArgumentParser(add_help=False)
    flooding.add_argument("--cycles",
                          type=int,
                          default=CYCLES,
                          help=f"number of load cycles after the flooding; a finished run given more is continued "
                          f"(default {CYCLES})")
    flooding.add_argument("--coupled",
                          action="store_true",
                          help="solve the excess pore pressure instead of draining the water")

    preload_parser = commands.add_parser("preload",
                                         parents=[common, deposits],
                                         help="preload one deposited packing dry")
    which = preload_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--packing", type=int, help="index of the packing")
    which.add_argument("--task", type=int, help="index of an array task, the packing")

    load_parser = commands.add_parser("load",
                                      parents=[common, flooding],
                                      help="flood one preloaded packing in one case and load it again")
    which = load_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--packing", type=int, help="index of the packing, with --case")
    which.add_argument("--task",
                       type=int,
                       help=f"index k of an array task: packing k // {len(CASES)} in case k % {len(CASES)} "
                       f"({', '.join(CASES)})")
    load_parser.add_argument("--case", choices=CASES, help="case to run, with --packing")

    all_parser = commands.add_parser("all",
                                     parents=[common, deposits, flooding],
                                     help="preload every packing and load it in every case, on this machine")
    all_parser.add_argument("--packings", type=int, default=PACKINGS, help=f"number of packings (default {PACKINGS})")
    all_parser.add_argument("--workers",
                            type=int,
                            default=os.cpu_count() or 1,
                            help="number of parallel processes (default: one per core)")

    arguments = parser.parse_args(argv)
    if arguments.output_dir is None:
        arguments.output_dir = OUTPUT_DIRS[arguments.grains]
    if arguments.save_every_n_steps is None:
        polyhedra = arguments.grains == "polyhedra"
        arguments.save_every_n_steps = POLYHEDRA_SAVE_EVERY_N_STEPS if polyhedra else SAVE_EVERY_N_STEPS
    if getattr(arguments, "deposit_dir", "") is None:
        arguments.deposit_dir = DEPOSIT_DIRS[arguments.grains]
    if arguments.save_every_n_steps < 0:
        parser.error(f"--save-every-n-steps must be non-negative, got {arguments.save_every_n_steps}.")
    if arguments.preload_cycles < 1:
        parser.error(f"--preload-cycles must be at least 1, got {arguments.preload_cycles}.")
    if arguments.command in ("preload", "load") and arguments.task is not None:
        if arguments.task < 0:
            parser.error(f"--task must be non-negative, got {arguments.task}.")
        if arguments.command == "preload":
            arguments.packing = arguments.task
        else:
            if arguments.case is not None:
                parser.error("--case is given by --task; use --case with --packing instead.")
            arguments.packing, case = divmod(arguments.task, len(CASES))
            arguments.case = CASES[case]
    if arguments.command == "load" and arguments.case is None:
        parser.error("--packing needs --case.")
    if arguments.command != "preload" and arguments.cycles < 1:
        parser.error(f"--cycles must be at least 1, got {arguments.cycles}.")
    return arguments


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the command given on the command line.

    Args:
        argv: arguments; None reads those of the process.
    """
    arguments = parse_arguments(argv)
    settings = LoadSettings(arguments.amplitude * 1e3, arguments.frequency, getattr(arguments, "coupled", False))
    if arguments.command == "preload":
        print(preload(arguments.packing, settings, arguments.preload_cycles, arguments.deposit_dir,
                      arguments.output_dir, arguments.save_every_n_steps, arguments.device),
              flush=True)
        return
    if arguments.command == "load":
        print(
            run_case(arguments.packing, arguments.case, settings, arguments.preload_cycles, arguments.cycles,
                     arguments.output_dir, arguments.save_every_n_steps, arguments.device))
        return
    run_all(arguments.packings, settings, arguments.preload_cycles, arguments.cycles, arguments.deposit_dir,
            arguments.output_dir, arguments.workers, arguments.save_every_n_steps, arguments.device)


if __name__ == "__main__":
    main()
