"""Cyclic loading of a railway sleeper on ballast in 3D: dry, half submerged and fully saturated.

The three-dimensional counterpart of ``ballast_cyclic.py``. The model is a box that holds one sleeper bay of the
track, with x along the track, y vertical and z along the sleeper:

* along the track the box is one sleeper spacing, 0.6 m, wide, between smooth walls that are planes of symmetry
  halfway to the neighbouring sleepers;
* along the sleeper it runs from the track centreline, a smooth wall that is the plane of symmetry of the track,
  past the end of the sleeper, 1.3 m away, to the end wall of the box 0.4 m further, which retains the shoulder;
* the sleeper rests on 0.3 m of ballast on a rigid formation, and ballast fills the cribs on both sides of it and the
  shoulder beyond its end.

So the box holds one whole sleeper, of which only half is modelled. The sleeper is a ``LoadedPlaten`` on four faces,
its base, its two sides and its end, whose sides reach the top of the box so that the crib ballast cannot spill onto
it. It has the 150 kg of half a concrete sleeper and its own weight, and it translates vertically; it neither bends
nor tilts. It carries the load of one rail seat, which cycles between zero and twice its amplitude,
``F = A (1 - cos 2 pi f t)``; the default, A = 32.5 kN at f = 5 Hz, peaks at 65 kN, 200 kPa on the sleeper base.
Ballast grains are spheres 30 to 50 mm across, of the same material as in 2D, rolling resistance included.

The ballast is deposited dry: first the layer under the sleeper, which is then seated on it, then the cribs and the
shoulder. The same packing is loaded in three cases: dry, with the water table halfway down the ballast under the
sleeper, and fully saturated with the water at the top of the ballast. Clean ballast drains almost instantly, so the
water is drained by default: it buoys the grains below the water table, drags on them when they move and pushes up
on the sleeper base with its hydrostatic pressure. ``--coupled`` solves the excess pore pressure as well, to check
that none builds up.

A 3D run is too long for a laptop, so the study is split into jobs that a cluster can run in parallel, and that
resume where they stopped if they are killed. From the repository root::

    python examples/ballast_cyclic_3d.py deposit --packing 0
    python examples/ballast_cyclic_3d.py load --packing 0 --case dry --amplitude 32.5 --frequency 5 --cycles 50

``deposit`` rains one packing and stores it; ``load`` loads a stored packing in one case, saves a checkpoint after
every cycle and continues from it when it is run again, also to add cycles to a finished run. ``--task k`` replaces
``--packing`` and ``--case`` in array jobs: packing k in ``deposit``, and packing k // 3 in case k % 3 (dry, half,
saturated) in ``load``. ``all`` runs every job on one machine, in parallel. ``ballast_cyclic_3d.sbatch`` submits the
jobs to a SLURM cluster, and ``ballast_cyclic_3d_post.py`` post-processes the results.

Every run writes to ``output/ballast_3d/packing_<k>/<case>/``: the load history of the sleeper, the forces on the
faces of the box and the positions of the grains at selected cycles (``result.npz``), and snapshots of the final
state. Both stages also write ParaView files every ``--save-every-n-steps`` time steps, the deposition to
``packing_<k>/paraview/`` and every run to ``packing_<k>/<case>/paraview/``; run the ``paraview_setup.py`` in such a
folder in ParaView to see the grains, the force chains and the sleeper. The default, 2000 steps, is about 0.024 s, or
eight exports per load cycle at 5 Hz; an export of a packing takes about 3 MB, so a run of 50 cycles writes some
1.2 GB and a deposition some 150 MB. ``--save-every-n-steps 0`` writes none. A run resumed from its checkpoint
carries on with its ParaView files, at the interval it is resumed with. Set ``OMP_NUM_THREADS=1`` when several jobs
share a node. A packing holds about 6600 grains. On one core,
depositing it takes about half an hour and a load cycle at 5 Hz about four minutes, so 50 cycles take some three
hours per case; the time per cycle grows in proportion to the period of the load.
"""

import argparse
import dataclasses
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np
import numpy.typing as npt

from DEMcookies import (BoundaryCondition, DEMSimulation, Frame, Geometry3D, LoadedPlaten, MaterialProperties, Open,
                        RainSettings, Sinusoidal, TimeHistory, Wall, Water)
from DEMcookies.plotting import render_animation, render_frame
from DEMcookies.vtk_export import VTKWriter

OUTPUT_DIR = os.path.join("output", "ballast_3d")

# box: one sleeper spacing along the track (x) and height (y) [m]
SLEEPER_SPACING = 0.6
BOX_HEIGHT = 0.65
# sleeper: width of its base along the track, half its length along z, and the ballast depth under it [m]
SLEEPER_WIDTH = 0.25
SLEEPER_HALF_LENGTH = 1.3
BALLAST_DEPTH = 0.3
# width of the shoulder between the end of the sleeper and the end wall of the box [m]
SHOULDER_WIDTH = 0.4
# band in which the crib and shoulder ballast is rained, which sets their height [m]
CRIB_BAND = (0.47, 0.50)
# distance of the rail seat from the track centreline, where the vertical sections are drawn [m]
RAIL_SEAT = 0.75

# half of a 300 kg concrete sleeper [kg]
SLEEPER_MASS = 150.0
# rail seat load, which cycles between zero and twice the amplitude [N], its frequency [Hz] and the number of cycles
AMPLITUDE = 32.5e3
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
COLOURS = {"dry": "#b35a1f", "half": "#1590a8", "saturated": "#2c4fa0"}
LINE_STYLES = {"dry": "-", "half": "--", "saturated": "-."}
LABELS = {"dry": "dry", "half": "water table at half depth", "saturated": "saturated"}

# names of the faces of the box, in the order of build_geometry; left is x = 0 and right is x = SLEEPER_SPACING
FACE_NAMES = ("floor", "right", "left", "end wall", "centreline", "top", "sleeper base", "sleeper left",
              "sleeper right", "sleeper end")

# grains rained per square metre of the insertion band and wave, time between waves, longest settling time of each
# rain pass, and rest after flooding [s]
RAIN_INTENSITY = 200.0
WAVE_INTERVAL = 0.022
SETTLE_TIME = 0.3
REST_TIME = 0.3
# waves of each rain pass: a layer is full after about 30, after which the waves only drop the odd small grain into
# the gaps of its surface, over a large area for ever
RAIN_WAVES = 50
# safety limit of the deposition, far above what a packing needs
MAX_GRAINS = 30000
# records per load cycle of the sleeper, of the forces on the faces and of the excess pore pressure
SAMPLES_PER_CYCLE = 40
# cycles after which the positions of the grains are kept, besides the start of loading and the last cycle
SNAPSHOT_CYCLES = (1, 10, 25, 50, 100, 200, 500, 1000)
# time steps between two ParaView exports: about 0.024 s, eight per load cycle at 5 Hz; zero writes none
SAVE_EVERY_N_STEPS = 2000
# frames of the animation per load cycle, and its frame rate
ANIMATION_FRAMES_PER_CYCLE = 4
ANIMATION_FPS = 20

T = TypeVar("T")


@dataclass(frozen=True)
class LoadSettings:
    """Cyclic load of a run, and how its water is modelled.

    Attributes:
        amplitude: amplitude of the rail seat load, which cycles between zero and twice it [N].
        frequency: frequency of the load [Hz].
        coupled: whether the excess pore pressure is solved; otherwise the water is drained.
    """

    amplitude: float
    frequency: float
    coupled: bool = False

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: if the amplitude is not positive.
            ValueError: if the frequency is not positive.
        """
        if self.amplitude <= 0.0:
            raise ValueError(f"The load amplitude must be positive, got {self.amplitude}.")
        if self.frequency <= 0.0:
            raise ValueError(f"The load frequency must be positive, got {self.frequency}.")


@dataclass(eq=False)
class LoadRun:
    """Cyclic loading of one packing in one case, which is checkpointed after every cycle.

    Attributes:
        sim: simulation, with the water of the case and the load of the run on the sleeper.
        case: name of the case, one of CASES.
        packing: index of the packing.
        settings: load of the run.
        water_level: height of the water table, None when dry [m].
        sleeper_base: height of the sleeper base when loading starts [m].
        crib_top: top of the highest grain when loading starts [m].
        buoyancy: upward force of the water on the grains at rest, before loading [N].
        uplift: upward force of the water on the sleeper base [N].
        pressure_points: points under the sleeper where the excess pore pressure is sampled, shape (k, 3) [m].
        cycles_done: number of completed load cycles.
        wall_time: wall-clock time spent loading [s].
        time: time since loading started at every sample [s].
        applied: load applied to the sleeper at every sample, the rail seat load less the uplift [N].
        displacement: settlement of the sleeper at every sample [m].
        reaction: force with which the ballast pushes the sleeper up at every sample [N].
        face_forces: force of the grains on every face of the box at every sample, each of shape (faces, 3) [N].
        excess_pressure: largest excess pore pressure under the sleeper at every sample [Pa].
        snapshots: positions of the grains after selected cycles, 0 being the start of loading [m].
        last_frame: latest recorded frame.
        animation: frames of the animation.
        paraview: writer of the ParaView files of the run, which observes the simulation, or None if the run writes
            none.
    """

    sim: DEMSimulation
    case: str
    packing: int
    settings: LoadSettings
    water_level: Optional[float]
    sleeper_base: float
    crib_top: float
    buoyancy: float
    uplift: float
    pressure_points: npt.NDArray[np.float64]
    cycles_done: int = 0
    wall_time: float = 0.0
    time: List[float] = field(default_factory=list)
    applied: List[float] = field(default_factory=list)
    displacement: List[float] = field(default_factory=list)
    reaction: List[float] = field(default_factory=list)
    face_forces: List[npt.NDArray[np.float64]] = field(default_factory=list)
    excess_pressure: List[float] = field(default_factory=list)
    snapshots: Dict[int, npt.NDArray[np.float64]] = field(default_factory=dict)
    last_frame: Optional[Frame] = None
    animation: List[Frame] = field(default_factory=list)
    paraview: Optional[VTKWriter] = None


def build_geometry() -> Geometry3D:
    """Define the box with the half sleeper, whose base and sides form one loaded body.

    The box is cut by the sleeper, a block that runs from the track centreline to the end of the sleeper and from
    its base to the top of the box. The centreline face and the top face go around the block, so both are
    non-convex.

    Returns:
        The domain, with a placeholder load on the sleeper; ``start_run`` gives it the load of the run.
    """
    width, height = SLEEPER_SPACING, BOX_HEIGHT
    length = SLEEPER_HALF_LENGTH + SHOULDER_WIDTH
    left = 0.5 * (SLEEPER_SPACING - SLEEPER_WIDTH)
    right = left + SLEEPER_WIDTH
    floor = [(0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, 0.0, length), (0.0, 0.0, length)]
    rim = [(x, height, z) for x, _, z in floor]
    opening = [(left, height, 0.0), (right, height, 0.0), (right, height, SLEEPER_HALF_LENGTH),
               (left, height, SLEEPER_HALF_LENGTH)]
    base = [(x, BALLAST_DEPTH, z) for x, _, z in opening]
    # vertices 0-3 are the corners of the floor, 4-7 those of the rim, 8-11 those of the sleeper at the top of the
    # box and 12-15 those of its base
    vertices = floor + rim + opening + base
    sleeper = LoadedPlaten(load=TimeHistory([0.0], [0.0]),
                           mass=SLEEPER_MASS,
                           self_weight=True,
                           direction=(0.0, -1.0, 0.0))
    faces_and_conditions: List[Tuple[Tuple[int, ...], BoundaryCondition]] = [
        ((0, 1, 2, 3), Wall()),  # floor: the formation
        ((1, 5, 6, 2), Wall(friction=0.0)),  # right: plane of symmetry halfway to the next sleeper
        ((0, 3, 7, 4), Wall(friction=0.0)),  # left: plane of symmetry halfway to the previous sleeper
        ((3, 2, 6, 7), Wall()),  # end wall of the box, beyond the shoulder
        ((0, 4, 8, 12, 13, 9, 5, 1), Wall(friction=0.0)),  # track centreline, around the sleeper
        ((4, 7, 6, 5, 9, 10, 11, 8), Open()),  # top, around the sleeper
        ((12, 15, 14, 13), sleeper),  # sleeper base
        ((15, 12, 8, 11), sleeper),  # sleeper side facing the left crib
        ((9, 13, 14, 10), sleeper),  # sleeper side facing the right crib
        ((14, 15, 11, 10), sleeper),  # sleeper end, facing the shoulder
    ]
    faces = [face for face, _ in faces_and_conditions]
    conditions = [condition for _, condition in faces_and_conditions]
    return Geometry3D(vertices, faces, conditions)


def build_simulation() -> DEMSimulation:
    """Define the empty box with the sleeper.

    Returns:
        The simulation, with an empty box.
    """
    return DEMSimulation(build_geometry(), MATERIAL, radius_range=RADIUS_RANGE)


def sleeper_base(sim: DEMSimulation) -> float:
    """Return the height of the sleeper base [m]."""
    return float(sim.platens[0].vertices[..., 1].min())


def sleeper_footprint(sim: DEMSimulation) -> Tuple[float, float, float]:
    """Return the x coordinates of the sides of the sleeper and its length from the centreline [m]."""
    vertices = sim.platens[0].vertices.reshape(-1, 3)
    return float(vertices[:, 0].min()), float(vertices[:, 0].max()), float(vertices[:, 2].max())


def packing_dir(output_dir: str, packing: int) -> str:
    """Return the directory of the deposited packing and of its cases."""
    return os.path.join(output_dir, f"packing_{packing}")


def case_dir(output_dir: str, packing: int, case: str) -> str:
    """Return the directory of the results of one packing in one case."""
    return os.path.join(packing_dir(output_dir, packing), case)


def save_pickle(value: object, path: str) -> None:
    """Write an object to a pickle file, through a temporary file, so that a job killed meanwhile keeps the old one.

    Args:
        value: object to store.
        path: output file.
    """
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_pickle(path: str, kind: Type[T]) -> T:
    """Read an object of a given type from a pickle file written by this script.

    Args:
        path: input file.
        kind: type of the stored object.

    Returns:
        The object.

    Raises:
        TypeError: if the file holds an object of another type.
    """
    with open(path, "rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, kind):
        raise TypeError(f"{path} holds a {type(value).__name__}, not a {kind.__name__}.")
    return value


def paraview_writer(directory: str, save_every_n_steps: int) -> Optional[VTKWriter]:
    """Return a writer of ParaView files to the ``paraview`` folder of a directory.

    Args:
        directory: directory of a deposition or of a run.
        save_every_n_steps: number of time steps between two exports; zero writes none.

    Returns:
        The writer, or None if no files are written.
    """
    if save_every_n_steps == 0:
        return None
    return VTKWriter(os.path.join(directory, "paraview"), save_every_n_steps)


def deposit(packing: int, output_dir: str, save_every_n_steps: int = SAVE_EVERY_N_STEPS) -> str:
    """Deposit the ballast under the sleeper, seat the sleeper on it, fill the cribs and the shoulder, and store it.

    A packing that is already stored is left alone, so that a study can be resumed.

    Args:
        packing: index of the packing, which seeds the random positions of the grains.
        output_dir: directory of the results.
        save_every_n_steps: number of time steps between two ParaView exports of the deposition; zero writes none.

    Returns:
        A summary of the packing.

    Raises:
        RuntimeError: if a pass of the deposition stops at the largest number of grains.
    """
    directory = packing_dir(output_dir, packing)
    path = os.path.join(directory, "deposit.pkl")
    if os.path.exists(path):
        return f"packing {packing}: already deposited in {path}"
    os.makedirs(directory, exist_ok=True)
    start = perf_counter()
    sim = build_simulation()
    paraview = paraview_writer(directory, save_every_n_steps)
    if paraview is not None:
        sim.add_observer(paraview)
    steps_between = round(WAVE_INTERVAL / sim.dt)
    settle_steps = round(SETTLE_TIME / sim.dt)
    area = SLEEPER_SPACING * (SLEEPER_HALF_LENGTH + SHOULDER_WIDTH)
    # the grains of the first pass must fit under the sleeper, which is seated on them afterwards
    under = (BALLAST_DEPTH - 2.5 * RADIUS_RANGE[1], BALLAST_DEPTH - 1.2 * RADIUS_RANGE[1])
    passes = [(under, area, settle_steps, 2 * packing),
              (CRIB_BAND, area - SLEEPER_WIDTH * SLEEPER_HALF_LENGTH, 2 * settle_steps, 2 * packing + 1)]
    for number, (band, band_area, settle, seed) in enumerate(passes):
        if number == 1:
            sim.seat_platens()
        sim.rain_until_full(
            RainSettings(batch_size=round(RAIN_INTENSITY * band_area),
                         steps_between=steps_between,
                         insertion_band=band,
                         settle_steps=settle,
                         record_every=settle_steps,
                         max_particles=MAX_GRAINS,
                         max_steps=RAIN_WAVES * steps_between,
                         seed=seed))
        if sim.n >= MAX_GRAINS:
            raise RuntimeError(f"The deposition of packing {packing} stopped at {MAX_GRAINS} grains.")
        print(
            f"packing {packing}: pass {number + 1} done, {sim.n} grains up to {sim.bed_top():.3f} m, "
            f"{(perf_counter() - start) / 60:.1f} min",
            flush=True)

    render_frame(sim, os.path.join(directory, "deposit.png"), width=7.0)
    render_frame(sim, os.path.join(directory, "deposit_section.png"), width=6.0, section=RAIL_SEAT)
    # the frames of the deposition are not needed by the cases, and would make every checkpoint large
    sim.frames.clear()
    # the cases write ParaView files of their own, so the stored packing must not take this writer along
    if paraview is not None:
        sim.remove_observer(paraview)
    save_pickle(sim, path)
    return (f"packing {packing}: {sim.n} grains, sleeper base {sleeper_base(sim):.3f} m, crib top "
            f"{sim.bed_top():.3f} m, deposited in {(perf_counter() - start) / 60:.1f} min")


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


def sleeper_load(settings: LoadSettings, uplift: float) -> Sinusoidal:
    """Return the load on the sleeper: the rail seat load, from zero to twice its amplitude, less the uplift.

    Args:
        settings: load of the run.
        uplift: upward force of the water on the sleeper base [N].

    Returns:
        The load history, which starts at the trough of the rail seat load.
    """
    return Sinusoidal(amplitude=settings.amplitude,
                      frequency=settings.frequency,
                      mean=settings.amplitude - uplift,
                      phase=-0.5 * math.pi)


def sleeper_uplift(sim: DEMSimulation, level: Optional[float]) -> float:
    """Return the upward force of the water on the sleeper base [N].

    Args:
        sim: simulation with its water, if the case has any.
        level: height of the water table, None when dry [m].

    Returns:
        The uplift, zero when dry or when the water table lies below the sleeper base.
    """
    base = sleeper_base(sim)
    if sim.water is None or level is None or level <= base:
        return 0.0
    left, right, length = sleeper_footprint(sim)
    # the pore water under the base is at hydrostatic pressure, since it drains freely
    return sim.water.density * sim.settings.gravity * (level - base) * (right - left) * length


def pressure_points(sim: DEMSimulation) -> npt.NDArray[np.float64]:
    """Return the points under the sleeper at which the excess pore pressure is sampled, shape (k, 3) [m]."""
    left, right, length = sleeper_footprint(sim)
    base = sleeper_base(sim)
    x = left + (right - left) * (np.arange(5) + 0.5) / 5
    y = np.linspace(0.05, base - 0.03, 4)
    z = length * (np.arange(7) + 0.5) / 7
    grid = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([axis.ravel() for axis in grid])


def start_run(sim: DEMSimulation,
              case: str,
              packing: int,
              settings: LoadSettings,
              paraview: Optional[VTKWriter] = None) -> LoadRun:
    """Flood a deposited packing, if the case has water, let it rest and put the load of the run on the sleeper.

    Args:
        sim: deposited simulation.
        case: name of the case, one of CASES.
        packing: index of the packing.
        settings: load of the run.
        paraview: writer of the ParaView files of the run, from the flooding on; None writes none.

    Returns:
        The run, before its first cycle.
    """
    if paraview is not None:
        sim.add_observer(paraview)
    level = water_level(sim, case)
    if level is not None:
        sim.set_water(Water(level=level, coupled=settings.coupled))
    # the dry bed rests as long as the flooded ones, so all cases start loading after the same history
    sim.rest(REST_TIME, record_every=round(REST_TIME / sim.dt))
    sim.frames.clear()
    base = sleeper_base(sim)
    uplift = sleeper_uplift(sim, level)
    platen = sim.platens[0]
    # the deposited packing is shared by every load, so the sleeper only gets the load of the run now
    platen.spec = dataclasses.replace(platen.spec, load=sleeper_load(settings, uplift))
    run = LoadRun(sim,
                  case,
                  packing,
                  settings,
                  level,
                  base,
                  sim.bed_top(),
                  float(sim.fluid_force[:, 1].sum()),
                  uplift,
                  pressure_points(sim),
                  paraview=paraview)
    record_sample(run, displacement=0.0)
    run.snapshots[0] = sim.pos.copy()
    return run


def record_sample(run: LoadRun, displacement: Optional[float] = None) -> None:
    """Append the current state of the sleeper, the forces on the faces and the excess pore pressure to a run.

    Args:
        run: run to record.
        displacement: settlement to record instead of that of the sleeper, which is only defined once it has been
            released [m].
    """
    sim = run.sim
    platen = sim.platens[0]
    run.time.append(sim.load_time)
    run.applied.append(platen.spec.load(sim.load_time))
    run.displacement.append(platen.displacement if displacement is None else displacement)
    run.reaction.append(platen.reaction)
    run.face_forces.append(sim.face_forces.copy())
    run.excess_pressure.append(float(np.max(np.abs(sim.excess_pore_pressure(run.pressure_points)))))


def per_cycle(time: npt.NDArray[np.float64], displacement: npt.NDArray[np.float64], frequency: float,
              cycles: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the settlement at the end of every cycle, when the load is back to zero, and the largest one in it.

    Args:
        time: time since loading started [s].
        displacement: settlement of the sleeper at every time [m].
        frequency: frequency of the load [Hz].
        cycles: number of cycles.

    Returns:
        The permanent and the peak settlement of every cycle [m].
    """
    period = 1.0 / frequency
    residual = np.interp(period * np.arange(1, cycles + 1), time, displacement)
    cycle = np.clip((time / period).astype(np.int64), 0, cycles - 1)
    peak = np.full(cycles, -np.inf)
    np.maximum.at(peak, cycle, displacement)
    return residual, peak


def save_results(run: LoadRun, path: str, extra: Optional[Dict[str, npt.NDArray[Any]]] = None) -> None:
    """Save the load history, the forces on the faces and the positions of the grains of a run.

    Args:
        run: run to save.
        path: output file [npz]. Scalars describe the run in SI units; ``time``, ``applied``, ``displacement``,
            ``reaction``, ``excess_pressure`` and ``face_forces`` (samples, faces, 3) are its records, and
            ``positions_<cycle>`` the positions of the grains after that cycle, one of ``snapshot_cycles``.
        extra: further arrays to store, by name, e.g. those of a study that builds on this one.
    """
    sim = run.sim
    left, right, length = sleeper_footprint(sim)
    low, high = sim.geometry.bounds
    snapshots = dict(run.snapshots)
    snapshots[run.cycles_done] = sim.pos.copy()
    # typed loosely because the numpy stub of savez mistakes keyword arrays for its allow_pickle flag
    arrays: Dict[str, Any] = {
        "case": np.array(run.case),
        "packing": np.array(run.packing),
        "amplitude": np.array(run.settings.amplitude),
        "frequency": np.array(run.settings.frequency),
        "coupled": np.array(run.settings.coupled),
        "cycles": np.array(run.cycles_done),
        "dt": np.array(sim.dt),
        "wall_time": np.array(run.wall_time),
        "n_grains": np.array(sim.n),
        "n_removed": np.array(sim.n_removed),
        "water_level": np.array(np.nan if run.water_level is None else run.water_level),
        "sleeper_base": np.array(run.sleeper_base),
        "crib_top": np.array(run.crib_top),
        "buoyancy": np.array(run.buoyancy),
        "uplift": np.array(run.uplift),
        "sleeper_mass": np.array(sim.platens[0].spec.mass),
        "sleeper_footprint": np.array([left, right, length]),
        "box": np.array(high - low),
        "rail_seat": np.array(RAIL_SEAT),
        "face_names": np.array(FACE_NAMES),
        "time": np.array(run.time),
        "applied": np.array(run.applied),
        "displacement": np.array(run.displacement),
        "reaction": np.array(run.reaction),
        "excess_pressure": np.array(run.excess_pressure),
        "face_forces": np.array(run.face_forces),
        "radii": sim.radius.copy(),
        "snapshot_cycles": np.array(sorted(snapshots)),
    }
    for cycle, positions in snapshots.items():
        arrays[f"positions_{cycle}"] = positions
    arrays.update(extra or {})
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def resume_paraview(run: LoadRun, directory: str, save_every_n_steps: int) -> None:
    """Make a resumed run write ParaView files at a given interval, or stop writing them.

    A run checkpointed with a writer carries on with its series; one checkpointed without, e.g. by an earlier version
    of this script, starts them.

    Args:
        run: run loaded from its checkpoint.
        directory: directory of the run.
        save_every_n_steps: number of time steps between two exports; zero writes none.
    """
    if run.paraview is None:
        run.paraview = paraview_writer(directory, save_every_n_steps)
        if run.paraview is not None:
            run.sim.add_observer(run.paraview)
    elif save_every_n_steps == 0:
        run.sim.remove_observer(run.paraview)
        run.paraview = None
    else:
        run.paraview.save_every_n_steps = save_every_n_steps


def run_case(packing: int,
             case: str,
             settings: LoadSettings,
             cycles: int,
             output_dir: str,
             animate: bool,
             save_every_n_steps: int = SAVE_EVERY_N_STEPS) -> str:
    """Load a deposited packing cyclically in one case, resuming from its checkpoint if there is one.

    Args:
        packing: index of the packing.
        case: name of the case, one of CASES.
        settings: load of the run.
        cycles: number of load cycles.
        output_dir: directory of the results.
        animate: whether an animation of a section at the rail seat is written as well.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.

    Returns:
        A summary of the run.

    Raises:
        FileNotFoundError: if the packing has not been deposited.
        ValueError: if a checkpoint of the run was made with another load.
    """
    directory = case_dir(output_dir, packing, case)
    checkpoint = os.path.join(directory, "checkpoint.pkl")
    if os.path.exists(checkpoint):
        run = load_pickle(checkpoint, LoadRun)
        if run.settings != settings:
            raise ValueError(f"The checkpoint {checkpoint} was made with {run.settings}, not {settings}; remove it, or "
                             f"choose another output directory, to start the run again.")
        resume_paraview(run, directory, save_every_n_steps)
        print(f"packing {packing}, {case}: resumed after cycle {run.cycles_done}", flush=True)
    else:
        source = os.path.join(packing_dir(output_dir, packing), "deposit.pkl")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Packing {packing} has not been deposited: {source} does not exist.")
        os.makedirs(directory, exist_ok=True)
        run = start_run(load_pickle(source, DEMSimulation), case, packing, settings,
                        paraview_writer(directory, save_every_n_steps))
        save_pickle(run, checkpoint)

    sim = run.sim
    period = 1.0 / settings.frequency
    animation_stride = max(1, SAMPLES_PER_CYCLE // ANIMATION_FRAMES_PER_CYCLE)
    results = os.path.join(directory, "result.npz")
    started = perf_counter()
    first_cycle = run.cycles_done
    while run.cycles_done < cycles:
        cycle_start = perf_counter()
        for sample in range(SAMPLES_PER_CYCLE):
            # every sample is aimed at its own phase of the load, so rounding to whole steps never accumulates
            target = (run.cycles_done + (sample + 1) / SAMPLES_PER_CYCLE) * period
            steps = max(1, round((target - sim.load_time) / sim.dt))
            # one frame and one load record per call; only the latest frame is kept, and those of the animation
            sim.compress(steps * sim.dt, record_every=steps)
            record_sample(run)
            run.last_frame = sim.frames[-1]
            if animate and sample % animation_stride == 0:
                run.animation.append(run.last_frame)
            sim.frames.clear()
        run.cycles_done += 1
        run.wall_time += perf_counter() - cycle_start
        if run.cycles_done in SNAPSHOT_CYCLES:
            run.snapshots[run.cycles_done] = sim.pos.copy()
        save_pickle(run, checkpoint)
        save_results(run, results)

        residual, peak = per_cycle(np.array(run.time), np.array(run.displacement), settings.frequency, run.cycles_done)
        elapsed = perf_counter() - started
        remaining = elapsed / (run.cycles_done - first_cycle) * (cycles - run.cycles_done)
        print(
            f"packing {packing}, {case}: cycle {run.cycles_done}/{cycles}, settlement {residual[-1] * 1e3:.2f} mm, "
            f"peak {peak[-1] * 1e3:.2f} mm; {elapsed / 60:.1f} min, about {remaining / 60:.0f} min to go",
            flush=True)

    save_results(run, results)
    if run.last_frame is not None:
        sim.frames[:] = [run.last_frame]
        render_frame(sim, os.path.join(directory, "final.png"), width=7.0)
        render_frame(sim, os.path.join(directory, "final_section.png"), width=6.0, section=RAIL_SEAT)
    if animate and run.animation:
        sim.frames[:] = run.animation
        render_animation(sim, os.path.join(directory, "animation.gif"), fps=ANIMATION_FPS, section=RAIL_SEAT)
    sim.frames.clear()
    residual, _ = per_cycle(np.array(run.time), np.array(run.displacement), settings.frequency, run.cycles_done)
    return (f"packing {packing}, {case}: settlement after {run.cycles_done} cycles {residual[-1] * 1e3:.2f} mm, "
            f"{run.wall_time / 3600:.2f} h of loading; results in {directory}")


def run_all(packings: int,
            settings: LoadSettings,
            cycles: int,
            output_dir: str,
            workers: int,
            animate: bool,
            save_every_n_steps: int = SAVE_EVERY_N_STEPS) -> None:
    """Deposit every packing and load it in every case, in parallel processes on one machine.

    Args:
        packings: number of packings.
        settings: load of the runs.
        cycles: number of load cycles.
        output_dir: directory of the results.
        workers: number of parallel processes.
        animate: whether the first packing of every case is animated as well.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary in pool.map(deposit, range(packings), [output_dir] * packings, [save_every_n_steps] * packings):
            print(summary, flush=True)
        runs = [
            pool.submit(run_case, packing, case, settings, cycles, output_dir, animate and packing == 0,
                        save_every_n_steps) for packing in range(packings) for case in CASES
        ]
        for run in runs:
            print(run.result(), flush=True)


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
    common.add_argument("--output-dir", default=OUTPUT_DIR, help=f"directory of the results (default {OUTPUT_DIR})")
    common.add_argument("--save-every-n-steps",
                        type=int,
                        default=SAVE_EVERY_N_STEPS,
                        help=f"time steps between two ParaView exports, 0 for none (default {SAVE_EVERY_N_STEPS})")
    loading = argparse.ArgumentParser(add_help=False)
    loading.add_argument("--amplitude",
                         type=float,
                         default=AMPLITUDE / 1e3,
                         help=f"amplitude of the rail seat load [kN], which cycles between zero and twice it "
                         f"(default {AMPLITUDE / 1e3:g})")
    loading.add_argument("--frequency",
                         type=float,
                         default=FREQUENCY,
                         help=f"frequency of the load [Hz] (default {FREQUENCY:g})")
    loading.add_argument("--cycles",
                         type=int,
                         default=CYCLES,
                         help=f"number of load cycles; a finished run given more is continued (default {CYCLES})")
    loading.add_argument("--coupled",
                         action="store_true",
                         help="solve the excess pore pressure instead of draining the water")
    loading.add_argument("--animate", action="store_true", help="also animate a section at the rail seat")

    deposit_parser = commands.add_parser("deposit", parents=[common], help="deposit the ballast of one packing")
    which = deposit_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--packing", type=int, help="index of the packing")
    which.add_argument("--task", type=int, help="index of an array task, the packing")

    load_parser = commands.add_parser("load", parents=[common, loading], help="load one packing in one case")
    which = load_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--packing", type=int, help="index of the packing, with --case")
    which.add_argument("--task",
                       type=int,
                       help=f"index k of an array task: packing k // {len(CASES)} in case k % {len(CASES)} "
                       f"({', '.join(CASES)})")
    load_parser.add_argument("--case", choices=CASES, help="case to run, with --packing")

    all_parser = commands.add_parser("all",
                                     parents=[common, loading],
                                     help="deposit every packing and load it in every case, on this machine")
    all_parser.add_argument("--packings", type=int, default=PACKINGS, help=f"number of packings (default {PACKINGS})")
    all_parser.add_argument("--workers",
                            type=int,
                            default=os.cpu_count() or 1,
                            help="number of parallel processes (default: one per core)")

    arguments = parser.parse_args(argv)
    if arguments.save_every_n_steps < 0:
        parser.error(f"--save-every-n-steps must be non-negative, got {arguments.save_every_n_steps}.")
    if arguments.command in ("deposit", "load") and arguments.task is not None:
        if arguments.task < 0:
            parser.error(f"--task must be non-negative, got {arguments.task}.")
        if arguments.command == "deposit":
            arguments.packing = arguments.task
        else:
            if arguments.case is not None:
                parser.error("--case is given by --task; use --case with --packing instead.")
            arguments.packing, case = divmod(arguments.task, len(CASES))
            arguments.case = CASES[case]
    if arguments.command == "load" and arguments.case is None:
        parser.error("--packing needs --case.")
    if arguments.command != "deposit" and arguments.cycles < 1:
        parser.error(f"--cycles must be at least 1, got {arguments.cycles}.")
    return arguments


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the command given on the command line.

    Args:
        argv: arguments; None reads those of the process.
    """
    arguments = parse_arguments(argv)
    if arguments.command == "deposit":
        print(deposit(arguments.packing, arguments.output_dir, arguments.save_every_n_steps), flush=True)
        return
    settings = LoadSettings(arguments.amplitude * 1e3, arguments.frequency, arguments.coupled)
    if arguments.command == "load":
        print(
            run_case(arguments.packing, arguments.case, settings, arguments.cycles, arguments.output_dir,
                     arguments.animate, arguments.save_every_n_steps))
        return
    run_all(arguments.packings, settings, arguments.cycles, arguments.output_dir, arguments.workers, arguments.animate,
            arguments.save_every_n_steps)


if __name__ == "__main__":
    main()
