"""Hanging sleeper test of a G44 concrete sleeper on 3D ballast, after the laboratory test of Esen et al. (2026).

Esen, Lojda, Laghrouche, van Belkom and Connolly, "The effect of different railway sleeper types on hanging sleeper
development", Transportation Geotechnics 62 (2026) 102151, loaded single sleepers on ballast in the GRAFT II box at
Heriot-Watt University for 600,000 cycles in displacement control: the sleeper keeps cycling between the same two
positions while the ballast settles away from it. They measured how the force on the rails falls, how large a gap
opens under the sleeper and how the pressure under the ballast changes. This script models their benchmark, the G44
concrete sleeper, in the frame of ``ballast_cyclic_3d.py``: x along the track, y vertical and z along the sleeper,
with half of the track modelled, from its centreline, a smooth plane of symmetry, to the end wall of the box.
Following Figs. 2 and 7 of the paper:

* along the track the ballast lies between wooden boards 0.65 m apart, one sleeper spacing, which are walls with the
  friction of the grains;
* along the sleeper the box is 3.8 m long, 1.9 m from the centreline to the end wall, so the shoulder beyond the end
  of the sleeper is 0.65 m wide;
* the floor is the top of the compacted sand subgrade, which is rigid here. The ballast is 0.4 m deep under the
  sleeper and fills the cribs and the shoulder up to the top of the box, 0.52 m above the subgrade and 0.12 m up the
  sides of the sleeper. Fig. 7 sketches the shoulders sloping down to the end walls; here they are filled level, like
  the cribs.

The sleeper is a G44: 2.5 m long and 0.2 m deep, with a trapezoidal section 285 mm wide at its base and 200 mm at its
top, and 309 kg, of which the model holds half. It is rigid and translates vertically. The ballast is that of
``ballast_cyclic_3d.py``: spheres 30 to 50 mm across, whose median of 40 mm is about that of the micro-granite of the
test (Fig. 3), with rolling resistance, or polyhedral stones with ``--grains polyhedra``. The grains are rained, not
compacted in layers as in the laboratory.

Every load is that of one actuator, on one rail seat, and the protocol follows the paper:

1. bedding, in force control: 1 kN, raised in steps of 10 kN to 41.7 kN, then to 51.7 and 61.3 kN, the rail seat
   loads of 17 t and 25 t axles, and lowered through the same steps back to 1 kN. The laboratory held every step for
   2 min; here it is ramped in 0.05 s and held for 0.1 s, about six periods of the sleeper bouncing on the ballast,
   since the grains do not creep;
2. calibration, in force control: the first cycle rises as a half sine from 1 kN to the target, 41.7 kN, the rail
   seat load of a 17 t axle of which the sleeper carries half. The settlement of the sleeper over the rise is the
   stroke;
3. fatigue, in displacement control: from the top of the first cycle on, the sleeper follows a sinusoid between the
   position at which that cycle started and the stroke below it, at 2.5 Hz. As the ballast settles, the force needed
   to reach the bottom of the stroke falls, and once the ballast has settled away from the sleeper, the sleeper hangs
   in the air for the first part of every downstroke.

The sleeper is a ``LoadedPlaten`` in the first two stages and becomes a ``PrescribedPlaten`` at the top of the first
cycle, whose path starts from where the sleeper is, so the switch makes no jump.

Every run records, 100 times per cycle, the force of the actuator on the rail seat (the load in force control; the
reaction of the ballast less the weight and the inertia of the half sleeper in displacement control), the settlement
of the sleeper, the force of the grains on the sleeper and on its base, and the pressure on the two pressure cells
under the ballast, the 9 in plates of the test: one under the rail seat and one under the centre of the sleeper, which
the plane of symmetry halves. From these it derives, for every cycle, as the paper does:

* the peak rail seat force (Fig. 12 of the paper);
* the gap: how far the sleeper travels down from the top of its stroke before the ballast pushes back on it with more
  than 2 % of the target load, in absolute terms and as a fraction of the stroke (Fig. 11). Like the laboratory, which
  sampled at 250 Hz, the records resolve it to within a few per cent of the stroke;
* the peak pressure on each cell (Fig. 13), over the pressure at rest before loading. The cells of the test were
  zeroed: the centre one reads 4.9 kPa at the end, less than the weight of the 0.4 m of ballast above it.

The model sets its stroke as the laboratory did, from the force-controlled rise, so the stroke reflects the stiffness
of the model, whose grains are ten times softer than granite and whose sleeper and subgrade are rigid. The laboratory
stroke, 1.77 mm, also includes the rail pads and the bending of the sleeper, so the gap is best compared as a fraction
of the stroke. The laboratory ran 600,000 cycles; a DEM run reaches some hundreds, so it is compared with the
snapshots of the test at 1, 10 and 100 cycles.

Like ``ballast_cyclic_3d.py``, the study is split into jobs that resume where they stopped if they are killed. From
the repository root::

    python examples/hanging_sleeper_g44.py deposit --packing 0
    python examples/hanging_sleeper_g44.py load --packing 0 --cycles 100
    python examples/hanging_sleeper_g44.py plot

``deposit`` rains one packing and stores it. ``load`` runs the test on a stored packing, saves a checkpoint after every
bedding step and every cycle, and continues from it when it is run again, also to add cycles to a finished run.
``--task k`` replaces ``--packing`` in array jobs. ``all`` runs every job on one machine, in parallel, and ``plot``
compares the runs with the test, in ``comparison.png`` and ``per_cycle.csv``. The deposition writes to
``output/hanging_sleeper_g44/packing_<k>/`` and the test to its ``fatigue/`` folder: the records and the per-cycle
results (``result.npz``), snapshots of the final state, and ParaView files every ``--save-every-n-steps`` time steps,
by default 8000, about 0.1 s or four per cycle.

A packing holds about 9000 grains, a third more than in ``ballast_cyclic_3d.py``, which seat the sleeper 0.40 m above
the floor and fill the cribs to 0.52 m. On a laptop, with two threads, depositing one took 72 min, a time step takes
21 ms, so a cycle at 2.5 Hz takes about 12 min, and the bedding, as long as five cycles, about an hour: 100 cycles take
some 21 h. Time the first cycles on your own machine before planning a study. ``--frequency 5`` halves the cost, at the
price of the rate of the test.
"""

import argparse
import csv
import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ballast_cyclic_3d import (GRAINS, MATERIAL, MAX_GRAINS, POLYHEDRA_SAVE_EVERY_N_STEPS, POLYHEDRAL_MATERIAL,
                               RADIUS_RANGE, RAIL_SEAT, RAIN_INTENSITY, SETTLE_TIME, STONES, WAVE_INTERVAL, load_pickle,
                               packing_dir, paraview_writer, save_pickle, sleeper_base)
from DEMcookies import (BoundaryCondition, DEMSimulation, Frame, Geometry3D, LoadedPlaten, Open, PrescribedPlaten,
                        RainSettings, TimeHistory, Wall)
from DEMcookies.backend import DEVICES
from DEMcookies.loads import LoadFunction
from DEMcookies.plotting import render_frame
from DEMcookies.vtk_export import VTKWriter

# directory of the results for each shape of the grains
OUTPUT_DIRS = {
    "spheres": os.path.join("output", "hanging_sleeper_g44"),
    "polyhedra": os.path.join("output", "hanging_sleeper_g44_polyhedra")
}

# box: distance between the wooden boards along the track (x), and half its length along the sleeper (z) [m]
BOX_WIDTH = 0.65
BOX_HALF_LENGTH = 1.9
# G44 sleeper: half its length, its depth, and the width of its base and of its top along the track [m]
SLEEPER_HALF_LENGTH = 1.25
SLEEPER_DEPTH = 0.2
SLEEPER_BASE_WIDTH = 0.285
SLEEPER_TOP_WIDTH = 0.2
# half of the 309 kg of a G44 [kg]
SLEEPER_MASS = 154.5
# height of the sleeper base before it is seated on the ballast; seating lowers it by about a centimetre, onto the
# 0.4 m of ballast of the test [m]
BALLAST_DEPTH = 0.41
# band in which the crib and shoulder ballast is rained, which fills them to about 0.52 m, the top of the box of the
# test, as it does in ballast_cyclic_3d.py [m]
CRIB_BAND = (0.47, 0.50)

# waves of each rain pass; the layer under the sleeper is a third deeper than in ballast_cyclic_3d.py and needs about
# 40 waves to fill, and polyhedra, which a wave places two thirds as many of, half as many again
RAIN_WAVES = 70
POLYHEDRA_RAIN_WAVES = 105

# pressure cells: 9 in plates on the subgrade, under the middle of the sleeper, at the rail seat and at the track
# centreline, with the fraction of each plate in the model, since the plane of symmetry halves the centre one
PRESSURE_CELL_RADIUS = 0.5 * 0.2286
PRESSURE_CELLS = ((RAIL_SEAT, 1.0), (0.0, 0.5))
CELL_NAMES = ("rail seat", "centre")

# bedding: rail seat load of every step on the way up, which is retraced on the way down [N], the time of the ramp to
# every step and the time it is held [s], and the records per step
BEDDING_LEVELS = (1.0e3, 11.0e3, 21.0e3, 31.0e3, 41.7e3, 51.7e3, 61.3e3)
BEDDING_RAMP = 0.05
BEDDING_HOLD = 0.1
BEDDING_SAMPLES = 15

# target rail seat load of the first cycle, which sets the stroke [N], frequency of the cycles [Hz] and their number
TARGET = 41.7e3
FREQUENCY = 2.5
CYCLES = 100
PACKINGS = 5
# records per cycle; even, so that the top of the first cycle, where the control switches, is a record
SAMPLES_PER_CYCLE = 100
# reaction of the ballast, as a fraction of the target load, above which the sleeper is no longer in the air
GAP_FORCE_FRACTION = 0.02
# cycles after which the positions of the grains are kept, besides the end of the bedding and the last cycle
SNAPSHOT_CYCLES = (1, 10, 100, 1000)
# time steps between two ParaView exports: about 0.1 s, four per cycle at 2.5 Hz; zero writes none
SAVE_EVERY_N_STEPS = 8000

# names of the faces of the box, in the order of build_geometry; the left board is at x = 0
FACE_NAMES = ("floor", "right board", "left board", "end wall", "centreline", "top", "sleeper base", "sleeper left",
              "sleeper right", "sleeper end")
FLOOR_FACE = FACE_NAMES.index("floor")
SLEEPER_BASE_FACE = FACE_NAMES.index("sleeper base")

# the G44 in the laboratory, read off the paper at its snapshot cycles: the peak rail seat force, averaged over both
# rails, of Fig. 12a [kN] (the text gives 41.7 kN at cycle 1), the gap, averaged over both rails, of Fig. 11, out of a
# stroke of 1.77 mm [mm], and the peak pressure on the cells under the left and the right rail seat and under the
# centre of Fig. 13a [kPa]
PAPER_CYCLES = (1, 10, 100, 1000, 10000, 100000, 300000, 600000)
PAPER_FORCE = (40.0, 38.1, 35.0, 29.6, 25.7, 24.2, 21.5, 18.0)
PAPER_GAP = (0.0, 0.08, 0.22, 0.35, 0.59, 0.67, 0.71, 0.88)
PAPER_STROKE = 1.77
PAPER_PRESSURE_LEFT = (54.1, 51.1, 48.5, 40.6, 35.8, 30.8, 31.2, 21.9)
PAPER_PRESSURE_RIGHT = (35.2, 35.0, 33.8, 31.5, 26.0, 24.2, 25.3, 26.0)
PAPER_PRESSURE_CENTRE = (29.5, 29.9, 30.2, 25.3, 22.2, 21.2, 10.5, 4.9)
PAPER_LABEL = "laboratory, Esen et al. (2026)"

# colours of the two places measured, the rail seat and the centre of the sleeper
RAIL_SEAT_COLOUR = "#2a78d6"
CENTRE_COLOUR = "#eb6834"


@dataclass(frozen=True)
class FatigueSettings:
    """Load of a test.

    Attributes:
        target: rail seat load at the top of the first cycle, which sets the stroke [N].
        frequency: frequency of the cycles [Hz].
    """

    target: float
    frequency: float

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: if the target load is not above the load at the end of the bedding.
            ValueError: if the frequency is not positive.
        """
        if self.target <= BEDDING_LEVELS[0]:
            raise ValueError(f"The target load must be above the {BEDDING_LEVELS[0] / 1e3:g} kN at the end of the "
                             f"bedding, got {self.target / 1e3:g} kN.")
        if self.frequency <= 0.0:
            raise ValueError(f"The frequency must be positive, got {self.frequency}.")


@dataclass(frozen=True)
class ForceControl:
    """Rail seat load of the force-controlled stages: the bedding steps, then the rise of the first cycle.

    It is a class rather than a closure so that it can be pickled with the checkpoints.

    Attributes:
        bedding: load of the bedding, which ends at ``start``.
        start: load time at which the first cycle starts [s].
        low: rail seat load at the start of the first cycle [N].
        high: target rail seat load, reached half a period later and held after it [N].
        frequency: frequency of the cycles [Hz].
    """

    bedding: TimeHistory
    start: float
    low: float
    high: float
    frequency: float

    def __call__(self, time: float) -> float:
        """Evaluate the load.

        Args:
            time: time since loading started [s].

        Returns:
            The rail seat load, positive downwards [N].
        """
        if time <= self.start:
            return self.bedding(time)
        elapsed = min(time - self.start, 0.5 / self.frequency)
        return self.low + 0.5 * (self.high - self.low) * (1.0 - math.cos(2.0 * math.pi * self.frequency * elapsed))


@dataclass(frozen=True)
class Stroke:
    """Settlement of the sleeper in displacement control: a sinusoid between two fixed positions.

    ``u = u0 + s (1 - cos 2 pi f (t - t0)) / 2`` from the start ``t0`` of the first cycle on, measured like the
    settlement of the sleeper, from where it was seated before the bedding. Before ``t0`` the sleeper is
    force-controlled and the history is never evaluated; it is zero there only because a ``PrescribedPlaten`` must
    start from zero.

    Attributes:
        start: load time at which the first cycle starts [s].
        origin: settlement of the sleeper then, the top of the stroke [m].
        stroke: distance from the top to the bottom of the stroke [m].
        frequency: frequency of the cycles [Hz].
    """

    start: float
    origin: float
    stroke: float
    frequency: float

    def __call__(self, time: float) -> float:
        """Evaluate the settlement.

        Args:
            time: time since loading started [s].

        Returns:
            The settlement of the sleeper, positive downwards [m].
        """
        if time < self.start:
            return 0.0
        phase = 2.0 * math.pi * self.frequency * (time - self.start)
        return self.origin + 0.5 * self.stroke * (1.0 - math.cos(phase))

    def acceleration(self, time: float) -> float:
        """Return the acceleration of the sleeper, positive downwards [m/s^2].

        Args:
            time: time since loading started [s].
        """
        if time < self.start:
            return 0.0
        omega = 2.0 * math.pi * self.frequency
        return 0.5 * self.stroke * omega**2 * math.cos(omega * (time - self.start))


@dataclass(eq=False)
class HangingRun:
    """The test of one packing, checkpointed after every bedding step and every cycle.

    Attributes:
        sim: simulation, with the load of the test on the sleeper.
        packing: index of the packing.
        settings: load of the test.
        sleeper_base: height of the sleeper base when loading starts [m].
        crib_top: top of the highest grain when loading starts [m].
        cycle_start: load time at which the bedding ends and the first cycle starts [s].
        rest_pressure: pressure on every pressure cell at rest, before loading, which the cells are zeroed at [Pa].
        origin: settlement of the sleeper at the start of the first cycle, the top of the stroke, once the bedding is
            done [m].
        control: prescribed path of the sleeper, once it is in displacement control.
        steps_done: number of completed bedding steps.
        cycles_done: number of completed cycles.
        wall_time: wall-clock time spent loading [s].
        cycle: cycle of every record, counted from zero, -1 for the bedding.
        time: time since loading started at every record [s].
        rail_seat_force: force of the actuator on the half sleeper at every record [N].
        displacement: settlement of the sleeper at every record [m].
        reaction: force with which the grains push the sleeper up at every record [N].
        base_force: upward force of the grains on the sleeper base at every record [N].
        cell_pressure: pressure on every pressure cell at every record, each of shape (cells,) [Pa].
        face_forces: force of the grains on every face of the box at every record, each of shape (faces, 3) [N].
        snapshots: positions of the grains after selected cycles, 0 being the end of the bedding [m].
        last_frame: latest recorded frame.
        paraview: writer of the ParaView files of the run, which observes the simulation, or None if the run writes
            none.
    """

    sim: DEMSimulation
    packing: int
    settings: FatigueSettings
    sleeper_base: float
    crib_top: float
    cycle_start: float
    rest_pressure: npt.NDArray[np.float64]
    origin: Optional[float] = None
    control: Optional[Stroke] = None
    steps_done: int = 0
    cycles_done: int = 0
    wall_time: float = 0.0
    cycle: List[int] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    rail_seat_force: List[float] = field(default_factory=list)
    displacement: List[float] = field(default_factory=list)
    reaction: List[float] = field(default_factory=list)
    base_force: List[float] = field(default_factory=list)
    cell_pressure: List[npt.NDArray[np.float64]] = field(default_factory=list)
    face_forces: List[npt.NDArray[np.float64]] = field(default_factory=list)
    snapshots: Dict[int, npt.NDArray[np.float64]] = field(default_factory=dict)
    last_frame: Optional[Frame] = None
    paraview: Optional[VTKWriter] = None


@dataclass(frozen=True)
class CycleResults:
    """Results of every cycle of a test.

    Attributes:
        peak_force: largest rail seat force of every cycle [N].
        gap: distance the sleeper travels down from the top of its stroke before it touches the ballast, in every
            cycle [m].
        peak_pressure: largest pressure on every pressure cell in every cycle, over its pressure at rest before
            loading, shape (cycles, cells) [Pa].
    """

    peak_force: npt.NDArray[np.float64]
    gap: npt.NDArray[np.float64]
    peak_pressure: npt.NDArray[np.float64]


def sleeper_platen(load: LoadFunction) -> LoadedPlaten:
    """Return the half sleeper in force control, under a given rail seat load and its own weight.

    Args:
        load: rail seat load as a function of the time since loading started [N].
    """
    return LoadedPlaten(load=load, mass=SLEEPER_MASS, self_weight=True, direction=(0.0, -1.0, 0.0))


def build_geometry() -> Geometry3D:
    """Define the box with the half sleeper, whose base, sides and end form one moving body.

    The sleeper is a trapezoidal prism from the track centreline to its end, whose top is the top of the box. The
    centreline face and the top face go around it, so both are non-convex.

    Returns:
        The domain, with the sleeper under a placeholder load; ``start_run`` gives it the load of the test.
    """
    width, height, length = BOX_WIDTH, BALLAST_DEPTH + SLEEPER_DEPTH, BOX_HALF_LENGTH
    middle = 0.5 * BOX_WIDTH
    top_left, top_right = middle - 0.5 * SLEEPER_TOP_WIDTH, middle + 0.5 * SLEEPER_TOP_WIDTH
    base_left, base_right = middle - 0.5 * SLEEPER_BASE_WIDTH, middle + 0.5 * SLEEPER_BASE_WIDTH
    floor = [(0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, 0.0, length), (0.0, 0.0, length)]
    rim = [(x, height, z) for x, _, z in floor]
    top = [(top_left, height, 0.0), (top_right, height, 0.0), (top_right, height, SLEEPER_HALF_LENGTH),
           (top_left, height, SLEEPER_HALF_LENGTH)]
    base = [(base_left, BALLAST_DEPTH, 0.0), (base_right, BALLAST_DEPTH, 0.0),
            (base_right, BALLAST_DEPTH, SLEEPER_HALF_LENGTH), (base_left, BALLAST_DEPTH, SLEEPER_HALF_LENGTH)]
    # vertices 0-3 are the corners of the floor, 4-7 those of the rim, 8-11 those of the top of the sleeper and 12-15
    # those of its base
    vertices = floor + rim + top + base
    sleeper = sleeper_platen(TimeHistory([0.0], [0.0]))
    faces_and_conditions: List[Tuple[Tuple[int, ...], BoundaryCondition]] = [
        ((0, 1, 2, 3), Wall()),  # floor: the top of the subgrade
        ((1, 5, 6, 2), Wall()),  # right wooden board
        ((0, 3, 7, 4), Wall()),  # left wooden board
        ((3, 2, 6, 7), Wall()),  # end wall of the box, beyond the shoulder
        ((0, 4, 8, 12, 13, 9, 5, 1), Wall(friction=0.0)),  # track centreline, around the sleeper
        ((4, 7, 6, 5, 9, 10, 11, 8), Open()),  # top, around the sleeper
        ((12, 15, 14, 13), sleeper),  # sleeper base
        ((15, 12, 8, 11), sleeper),  # sloping side of the sleeper facing the left crib
        ((9, 13, 14, 10), sleeper),  # sloping side of the sleeper facing the right crib
        ((14, 15, 11, 10), sleeper),  # sleeper end, facing the shoulder
    ]
    faces = [face for face, _ in faces_and_conditions]
    conditions = [condition for _, condition in faces_and_conditions]
    return Geometry3D(vertices, faces, conditions)


def build_simulation(grains: str = "spheres", device: str = "cpu") -> DEMSimulation:
    """Define the empty box with the sleeper.

    Args:
        grains: shape of the grains, one of GRAINS.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu".

    Returns:
        The simulation, with an empty box.

    Raises:
        ValueError: if the shape of the grains is unknown.
    """
    if grains == "spheres":
        return DEMSimulation(build_geometry(), MATERIAL, radius_range=RADIUS_RANGE, device=device)
    if grains == "polyhedra":
        return DEMSimulation(build_geometry(),
                             POLYHEDRAL_MATERIAL,
                             radius_range=RADIUS_RANGE,
                             shapes=STONES,
                             device=device)
    raise ValueError(f"Unknown grains {grains!r}; expected one of {GRAINS}.")


def fatigue_dir(output_dir: str, packing: int) -> str:
    """Return the directory of the test of a packing."""
    return os.path.join(packing_dir(output_dir, packing), "fatigue")


def deposit(packing: int,
            output_dir: str,
            save_every_n_steps: int = SAVE_EVERY_N_STEPS,
            grains: str = "spheres",
            device: str = "cpu") -> str:
    """Deposit the ballast under the sleeper, seat the sleeper on it, fill the cribs and the shoulder, and store it.

    A packing that is already stored is left alone, so that a study can be resumed.

    Args:
        packing: index of the packing, which seeds the random positions of the grains.
        output_dir: directory of the results.
        save_every_n_steps: number of time steps between two ParaView exports of the deposition; zero writes none.
        grains: shape of the grains, one of GRAINS.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu".

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
    sim = build_simulation(grains, device)
    paraview = paraview_writer(directory, save_every_n_steps)
    if paraview is not None:
        sim.add_observer(paraview)
    steps_between = round(WAVE_INTERVAL / sim.dt)
    settle_steps = round(SETTLE_TIME / sim.dt)
    waves = POLYHEDRA_RAIN_WAVES if grains == "polyhedra" else RAIN_WAVES
    area = BOX_WIDTH * BOX_HALF_LENGTH
    # the grains of the first pass must fit under the sleeper, which is seated on them afterwards
    under = (BALLAST_DEPTH - 2.5 * RADIUS_RANGE[1], BALLAST_DEPTH - 1.2 * RADIUS_RANGE[1])
    passes = [(under, area, settle_steps, 2 * packing),
              (CRIB_BAND, area - SLEEPER_BASE_WIDTH * SLEEPER_HALF_LENGTH, 2 * settle_steps, 2 * packing + 1)]
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
                         max_steps=waves * steps_between,
                         seed=seed))
        if sim.n >= MAX_GRAINS:
            raise RuntimeError(f"The deposition of packing {packing} stopped at {MAX_GRAINS} grains.")
        print(
            f"packing {packing}: pass {number + 1} done, {sim.n} grains up to {sim.bed_top():.3f} m, "
            f"{(perf_counter() - start) / 60:.1f} min",
            flush=True)

    render_frame(sim, os.path.join(directory, "deposit.png"), width=7.0)
    render_frame(sim, os.path.join(directory, "deposit_section.png"), width=6.0, section=RAIL_SEAT)
    # the frames of the deposition are not needed by the test, and would make every checkpoint large
    sim.frames.clear()
    # the test writes ParaView files of its own, so the stored packing must not take this writer along
    if paraview is not None:
        sim.remove_observer(paraview)
    save_pickle(sim, path)
    return (f"packing {packing}: {sim.n} grains, sleeper base {sleeper_base(sim):.3f} m, crib top "
            f"{sim.bed_top():.3f} m, deposited in {(perf_counter() - start) / 60:.1f} min")


def bedding_levels() -> Tuple[float, ...]:
    """Return the rail seat load of every bedding step, up to the largest and back down to the first [N]."""
    return BEDDING_LEVELS + BEDDING_LEVELS[-2::-1]


def bedding_load() -> TimeHistory:
    """Return the load of the bedding: every step is ramped to from the previous one, from zero, and then held."""
    times, values = [0.0], [0.0]
    for level in bedding_levels():
        times += [times[-1] + BEDDING_RAMP, times[-1] + BEDDING_RAMP + BEDDING_HOLD]
        values += [level, level]
    return TimeHistory(times, values)


def start_run(sim: DEMSimulation,
              packing: int,
              settings: FatigueSettings,
              paraview: Optional[VTKWriter] = None) -> HangingRun:
    """Put the force-controlled load of the test on the sleeper of a deposited packing.

    Args:
        sim: deposited simulation.
        packing: index of the packing.
        settings: load of the test.
        paraview: writer of the ParaView files of the run; None writes none.

    Returns:
        The run, before its first bedding step.
    """
    if paraview is not None:
        sim.add_observer(paraview)
    start = len(bedding_levels()) * (BEDDING_RAMP + BEDDING_HOLD)
    # the deposited packing is shared by every load, so the sleeper only gets the load of the test now
    sim.platens[0].spec = sleeper_platen(
        ForceControl(bedding_load(), start, BEDDING_LEVELS[0], settings.target, settings.frequency))
    return HangingRun(sim,
                      packing,
                      settings,
                      sleeper_base(sim),
                      sim.bed_top(),
                      start,
                      cell_pressures(sim),
                      paraview=paraview)


def cell_pressures(sim: DEMSimulation) -> npt.NDArray[np.float64]:
    """Return the pressure of the grains on every pressure cell, their load on the part of the plate in the model [Pa].

    Args:
        sim: simulation.

    Returns:
        The pressure on every cell, in the order of PRESSURE_CELLS.
    """
    _, walls = sim.contacts()
    on_floor = walls.second == FLOOR_FACE
    point = walls.point[on_floor]
    # the upward push of the floor on a grain is as large as the load of the grain on the floor
    load = walls.force[on_floor, 1]
    pressures = np.empty(len(PRESSURE_CELLS))
    for k, (z, fraction) in enumerate(PRESSURE_CELLS):
        inside = (point[:, 0] - 0.5 * BOX_WIDTH)**2 + (point[:, 2] - z)**2 <= PRESSURE_CELL_RADIUS**2
        pressures[k] = load[inside].sum() / (fraction * math.pi * PRESSURE_CELL_RADIUS**2)
    return pressures


def rail_seat_force(run: HangingRun) -> float:
    """Return the force of the actuator on the half sleeper, positive downwards [N].

    In force control it is the load. In displacement control the sleeper follows its path, so the actuator supplies
    whatever the reaction of the grains, the weight of the sleeper and its inertia leave over.

    Args:
        run: run, with its sleeper released.
    """
    sim = run.sim
    platen = sim.platens[0]
    if run.control is None:
        return platen.applied_load(sim.load_time)
    return platen.reaction - SLEEPER_MASS * (sim.settings.gravity - run.control.acceleration(sim.load_time))


def record_sample(run: HangingRun, cycle: int) -> None:
    """Append the current state of the sleeper, the pressure cells and the faces of the box to a run.

    Args:
        run: run to record.
        cycle: cycle of the record, counted from zero, -1 for the bedding.
    """
    sim = run.sim
    platen = sim.platens[0]
    run.cycle.append(cycle)
    run.time.append(sim.load_time)
    run.rail_seat_force.append(rail_seat_force(run))
    run.displacement.append(platen.displacement)
    run.reaction.append(platen.reaction)
    run.base_force.append(float(sim.face_forces[SLEEPER_BASE_FACE, 1]))
    run.cell_pressure.append(cell_pressures(sim))
    run.face_forces.append(sim.face_forces.copy())


def advance_to(run: HangingRun, load_time: float, cycle: int) -> None:
    """Advance a run to a load time, to the nearest time step, and record it.

    Args:
        run: run to advance.
        load_time: load time to reach [s].
        cycle: cycle of the record, counted from zero, -1 for the bedding.
    """
    sim = run.sim
    steps = max(1, round((load_time - sim.load_time) / sim.dt))
    # one frame and one load record per call; only the latest frame is kept
    sim.compress(steps * sim.dt, record_every=steps)
    record_sample(run, cycle)
    run.last_frame = sim.frames[-1]
    sim.frames.clear()


def bed(run: HangingRun, checkpoint: str) -> None:
    """Run the bedding steps that a run has not done yet, checkpointing it after every one.

    Args:
        run: run to bed.
        checkpoint: file of the checkpoint [pkl].
    """
    levels = bedding_levels()
    duration = BEDDING_RAMP + BEDDING_HOLD
    while run.steps_done < len(levels):
        started = perf_counter()
        for sample in range(BEDDING_SAMPLES):
            advance_to(run, (run.steps_done + (sample + 1) / BEDDING_SAMPLES) * duration, -1)
        run.steps_done += 1
        run.wall_time += perf_counter() - started
        if run.steps_done == len(levels):
            run.origin = run.sim.platens[0].displacement
            run.snapshots[0] = run.sim.pos.copy()
        save_pickle(run, checkpoint)
        print(
            f"packing {run.packing}: bedding step {run.steps_done}/{len(levels)} at "
            f"{levels[run.steps_done - 1] / 1e3:.1f} kN, settlement {run.displacement[-1] * 1e3:.2f} mm, "
            f"{run.wall_time / 60:.1f} min",
            flush=True)


def switch_to_displacement_control(run: HangingRun) -> None:
    """Take the stroke from the settlement over the rise of the first cycle and prescribe the path of the sleeper.

    Args:
        run: run at the top of its first cycle, with its bedding done.

    Raises:
        RuntimeError: if the bedding is not done.
    """
    if run.origin is None:
        raise RuntimeError("The stroke is set at the top of the first cycle, after the bedding.")
    platen = run.sim.platens[0]
    run.control = Stroke(run.cycle_start, run.origin, platen.displacement - run.origin, run.settings.frequency)
    platen.spec = PrescribedPlaten(displacement=run.control, seat_on_bed=False, direction=(0.0, -1.0, 0.0))


def load_cycle(run: HangingRun) -> None:
    """Run the next cycle of a run whose bedding is done, switching to displacement control in the first.

    Args:
        run: run to load.
    """
    period = 1.0 / run.settings.frequency
    cycle = run.cycles_done
    for sample in range(SAMPLES_PER_CYCLE):
        # every sample is aimed at its own phase of the cycle, so rounding to whole steps never accumulates
        advance_to(run, run.cycle_start + (cycle + (sample + 1) / SAMPLES_PER_CYCLE) * period, cycle)
        if run.control is None and sample + 1 == SAMPLES_PER_CYCLE // 2:
            switch_to_displacement_control(run)
    run.cycles_done += 1


def downstroke_gap(travel: npt.NDArray[np.float64], reaction: npt.NDArray[np.float64], threshold: float) -> float:
    """Return how far the sleeper travels down from the top of its stroke before it touches the ballast [m].

    Args:
        travel: distance of the sleeper below the top of its stroke at every record of a downstroke, from its top [m].
        reaction: force with which the grains push the sleeper up at every record [N].
        threshold: reaction above which the sleeper touches the ballast [N].

    Returns:
        The gap, interpolated between the records on either side of the first touch; zero if the sleeper touches the
        ballast at the top of its stroke, and the whole downstroke if it never does.
    """
    touching = reaction > threshold
    if touching[0]:
        return 0.0
    if not np.any(touching):
        return float(travel[-1])
    k = int(np.argmax(touching))
    weight = (threshold - reaction[k - 1]) / (reaction[k] - reaction[k - 1])
    return float(travel[k - 1] + weight * (travel[k] - travel[k - 1]))


def cycle_results(cycle: npt.NDArray[np.int64], rail_force: npt.NDArray[np.float64],
                  displacement: npt.NDArray[np.float64], reaction: npt.NDArray[np.float64],
                  cell_pressure: npt.NDArray[np.float64], origin: float, threshold: float) -> CycleResults:
    """Return the peak force, the gap and the peak pressures of every completed cycle of a test.

    Args:
        cycle: cycle of every record, counted from zero, -1 for the bedding.
        rail_force: rail seat force at every record [N].
        displacement: settlement of the sleeper at every record [m].
        reaction: force with which the grains push the sleeper up at every record [N].
        cell_pressure: pressure on every cell at every record, over its pressure at rest before loading, shape
            (records, cells) [Pa].
        origin: settlement of the sleeper at the top of its stroke [m].
        threshold: reaction above which the sleeper touches the ballast [N].

    Returns:
        The results of the cycles whose records are complete.
    """
    cycles = int(cycle.max()) + 1 if cycle.size else 0
    peak_force = np.empty(cycles)
    gap = np.empty(cycles)
    peak_pressure = np.empty((cycles, cell_pressure.shape[1]))
    for c in range(cycles):
        rows = np.flatnonzero(cycle == c)
        peak_force[c] = rail_force[rows].max()
        peak_pressure[c] = cell_pressure[rows].max(axis=0)
        # the downstroke runs from the last record of the previous cycle, or of the bedding, at the top of the
        # stroke, to the record half way through this one, at its bottom
        down = np.concatenate(([rows[0] - 1], rows[:rows.size // 2]))
        gap[c] = downstroke_gap(displacement[down] - origin, reaction[down], threshold)
    return CycleResults(peak_force, gap, peak_pressure)


def save_results(run: HangingRun, path: str) -> Optional[CycleResults]:
    """Save the records, the per-cycle results and the positions of the grains of a run.

    Args:
        run: run to save.
        path: output file [npz]. Scalars describe the test in SI units; ``cycle``, ``time``, ``rail_seat_force``,
            ``displacement``, ``reaction``, ``base_force``, ``cell_pressure`` (records, cells) and ``face_forces``
            (records, faces, 3) are its records, ``peak_force``, ``gap`` and ``peak_pressure`` (cycles, cells) the
            results of every cycle, the last over ``cell_pressure_at_rest``, the pressure on the cells before
            loading, and ``positions_<cycle>`` the positions of the grains after that cycle, one of
            ``snapshot_cycles``, 0 being the end of the bedding.

    Returns:
        The results of every cycle, or None before the bedding is done.
    """
    sim = run.sim
    snapshots = dict(run.snapshots)
    if run.cycles_done:
        snapshots[run.cycles_done] = sim.pos.copy()
    records = {
        "cycle": np.array(run.cycle, dtype=np.int64),
        "time": np.array(run.time),
        "rail_seat_force": np.array(run.rail_seat_force),
        "displacement": np.array(run.displacement),
        "reaction": np.array(run.reaction),
        "base_force": np.array(run.base_force),
        "cell_pressure": np.array(run.cell_pressure).reshape(-1, len(PRESSURE_CELLS)),
        "face_forces": np.array(run.face_forces).reshape(-1, len(FACE_NAMES), 3),
    }
    threshold = GAP_FORCE_FRACTION * run.settings.target
    results = None
    if run.origin is not None:
        results = cycle_results(records["cycle"], records["rail_seat_force"], records["displacement"],
                                records["reaction"], records["cell_pressure"] - run.rest_pressure, run.origin,
                                threshold)
    # typed loosely because the numpy stub of savez mistakes keyword arrays for its allow_pickle flag
    arrays: Dict[str, Any] = {
        "packing": np.array(run.packing),
        "grains": np.array("polyhedra" if np.any(sim.polytope) else "spheres"),
        "target": np.array(run.settings.target),
        "frequency": np.array(run.settings.frequency),
        "cycles": np.array(run.cycles_done),
        "samples_per_cycle": np.array(SAMPLES_PER_CYCLE),
        "dt": np.array(sim.dt),
        "wall_time": np.array(run.wall_time),
        "n_grains": np.array(sim.n),
        "n_removed": np.array(sim.n_removed),
        "sleeper_base": np.array(run.sleeper_base),
        "crib_top": np.array(run.crib_top),
        "sleeper_mass": np.array(SLEEPER_MASS),
        "box": np.array([BOX_WIDTH, BALLAST_DEPTH + SLEEPER_DEPTH, BOX_HALF_LENGTH]),
        "sleeper": np.array([SLEEPER_BASE_WIDTH, SLEEPER_TOP_WIDTH, SLEEPER_DEPTH, SLEEPER_HALF_LENGTH]),
        "bedding_levels": np.array(bedding_levels()),
        "cycle_start": np.array(run.cycle_start),
        "origin": np.array(np.nan if run.origin is None else run.origin),
        "stroke": np.array(np.nan if run.control is None else run.control.stroke),
        "gap_threshold": np.array(threshold),
        "cell_positions": np.array([z for z, _ in PRESSURE_CELLS]),
        "cell_radius": np.array(PRESSURE_CELL_RADIUS),
        "cell_pressure_at_rest": run.rest_pressure.copy(),
        "cell_names": np.array(CELL_NAMES),
        "face_names": np.array(FACE_NAMES),
        "radii": sim.radius.copy(),
        "snapshot_cycles": np.array(sorted(snapshots), dtype=np.int64),
    }
    arrays.update(records)
    if results is not None:
        arrays.update(peak_force=results.peak_force, gap=results.gap, peak_pressure=results.peak_pressure)
    for cycle, positions in snapshots.items():
        arrays[f"positions_{cycle}"] = positions
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)
    return results


def resume_paraview(run: HangingRun, directory: str, save_every_n_steps: int) -> None:
    """Make a resumed run write ParaView files at a given interval, or stop writing them.

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


def run_test(packing: int,
             settings: FatigueSettings,
             cycles: int,
             output_dir: str,
             save_every_n_steps: int = SAVE_EVERY_N_STEPS,
             device: str = "cpu") -> str:
    """Run the test on a deposited packing, resuming from its checkpoint if there is one.

    Args:
        packing: index of the packing.
        settings: load of the test.
        cycles: number of cycles, the first of which sets the stroke.
        output_dir: directory of the results.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu", whatever the run was checkpointed on.

    Returns:
        A summary of the run.

    Raises:
        FileNotFoundError: if the packing has not been deposited.
        ValueError: if a checkpoint of the run was made with another load.
    """
    directory = fatigue_dir(output_dir, packing)
    checkpoint = os.path.join(directory, "checkpoint.pkl")
    results_path = os.path.join(directory, "result.npz")
    if os.path.exists(checkpoint):
        run = load_pickle(checkpoint, HangingRun)
        run.sim.device = device
        if run.settings != settings:
            raise ValueError(f"The checkpoint {checkpoint} was made with {run.settings}, not {settings}; remove it, or "
                             f"choose another output directory, to start the test again.")
        resume_paraview(run, directory, save_every_n_steps)
        print(f"packing {packing}: resumed after bedding step {run.steps_done} and cycle {run.cycles_done}", flush=True)
    else:
        source = os.path.join(packing_dir(output_dir, packing), "deposit.pkl")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Packing {packing} has not been deposited: {source} does not exist.")
        os.makedirs(directory, exist_ok=True)
        deposited = load_pickle(source, DEMSimulation)
        deposited.device = device
        run = start_run(deposited, packing, settings, paraview_writer(directory, save_every_n_steps))
        save_pickle(run, checkpoint)

    bed(run, checkpoint)
    started = perf_counter()
    first_cycle = run.cycles_done
    results = None
    while run.cycles_done < cycles:
        cycle_start = perf_counter()
        load_cycle(run)
        run.wall_time += perf_counter() - cycle_start
        if run.cycles_done in SNAPSHOT_CYCLES:
            run.snapshots[run.cycles_done] = run.sim.pos.copy()
        save_pickle(run, checkpoint)
        results = save_results(run, results_path)
        # the bedding is done and the first cycle has set the stroke
        assert results is not None and run.control is not None
        elapsed = perf_counter() - started
        remaining = elapsed / (run.cycles_done - first_cycle) * (cycles - run.cycles_done)
        if run.cycles_done == 1:
            print(f"packing {packing}: stroke {run.control.stroke * 1e3:.2f} mm at {settings.target / 1e3:g} kN "
                  f"(laboratory {PAPER_STROKE:.2f} mm)")
        pressures = "/".join(f"{p / 1e3:.1f}" for p in results.peak_pressure[-1])
        print(
            f"packing {packing}: cycle {run.cycles_done}/{cycles}, peak force {results.peak_force[-1] / 1e3:.1f} kN, "
            f"gap {results.gap[-1] * 1e3:.3f} mm ({100 * results.gap[-1] / run.control.stroke:.0f} % of the stroke), "
            f"cell pressure increases {pressures} kPa; {elapsed / 60:.1f} min, about {remaining / 60:.0f} min to go",
            flush=True)

    results = save_results(run, results_path)
    if run.last_frame is not None:
        run.sim.frames[:] = [run.last_frame]
        render_frame(run.sim, os.path.join(directory, "final.png"), width=7.0)
        render_frame(run.sim, os.path.join(directory, "final_section.png"), width=6.0, section=RAIL_SEAT)
        run.sim.frames.clear()
    if results is None or run.control is None:
        return f"packing {packing}: bedded; results in {directory}"
    return (f"packing {packing}: after {run.cycles_done} cycles peak force {results.peak_force[-1] / 1e3:.1f} kN, "
            f"gap {results.gap[-1] * 1e3:.3f} mm of a {run.control.stroke * 1e3:.2f} mm stroke, "
            f"{run.wall_time / 3600:.2f} h of loading; results in {directory}")


def run_all(packings: int,
            settings: FatigueSettings,
            cycles: int,
            output_dir: str,
            workers: int,
            save_every_n_steps: int = SAVE_EVERY_N_STEPS,
            grains: str = "spheres",
            device: str = "cpu") -> None:
    """Deposit every packing and test it, in parallel processes on one machine.

    Args:
        packings: number of packings.
        settings: load of the tests.
        cycles: number of cycles.
        output_dir: directory of the results.
        workers: number of parallel processes.
        save_every_n_steps: number of time steps between two ParaView exports; zero writes none.
        grains: shape of the grains, one of GRAINS.
        device: device that measures the overlaps of polyhedra, "cpu" or "gpu".
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary in pool.map(deposit, range(packings), [output_dir] * packings, [save_every_n_steps] * packings,
                                [grains] * packings, [device] * packings):
            print(summary, flush=True)
        runs = [
            pool.submit(run_test, packing, settings, cycles, output_dir, save_every_n_steps, device)
            for packing in range(packings)
        ]
        for run in runs:
            print(run.result(), flush=True)


def style_axes(ax: Any, ylabel: str, title: str, legend: str = "lower left") -> None:
    """Give an axes of the comparison its labels, a logarithmic cycle axis, a recessive grid and a legend.

    Args:
        ax: matplotlib axes.
        ylabel: label of the vertical axis.
        title: title of the panel.
        legend: location of the legend, clear of the data.
    """
    ax.set_xscale("log")
    ax.set_xlim(0.7, 1.5 * PAPER_CYCLES[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("load cycle")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, which="major", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc=legend)


def plot(output_dir: str) -> str:
    """Compare the tests in a directory with the laboratory test, and write the results of every cycle.

    Writes ``comparison.png``, with the peak rail seat force, the gap as a fraction of the stroke and the peak
    increase of the pressure on the cells against the cycle, and ``per_cycle.csv`` to the directory. Runs still in
    their bedding are left out.

    Args:
        output_dir: directory of the results.

    Returns:
        A summary of the comparison.

    Raises:
        FileNotFoundError: if no test in the directory has completed a cycle.
    """
    runs = []
    for path in sorted(glob.glob(os.path.join(output_dir, "packing_*", "fatigue", "result.npz"))):
        with np.load(path) as stored:
            if int(stored["cycles"]) > 0:
                runs.append(
                    {key: stored[key]
                     for key in ("packing", "cycles", "stroke", "peak_force", "gap", "peak_pressure")})
    if not runs:
        raise FileNotFoundError(f"No test in {output_dir} has completed a cycle.")

    with open(os.path.join(output_dir, "per_cycle.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "packing", "cycle", "peak_force_kN", "gap_mm", "gap_percent_of_stroke", "rail_seat_increase_kPa",
            "centre_increase_kPa"
        ])
        for run in runs:
            for c in range(int(run["cycles"])):
                writer.writerow([
                    int(run["packing"]), c + 1, f"{run['peak_force'][c] / 1e3:.3f}", f"{run['gap'][c] * 1e3:.4f}",
                    f"{100 * run['gap'][c] / run['stroke']:.2f}", f"{run['peak_pressure'][c, 0] / 1e3:.2f}",
                    f"{run['peak_pressure'][c, 1] / 1e3:.2f}"
                ])

    # the mean runs over the cycles that every packing has completed
    common = min(int(run["cycles"]) for run in runs)
    axis = np.arange(1, common + 1)
    force = np.array([run["peak_force"][:common] for run in runs]) / 1e3
    gap_percent = np.array([100 * run["gap"][:common] / run["stroke"] for run in runs])
    pressure = np.array([run["peak_pressure"][:common] for run in runs]) / 1e3
    mean_label = f"DEM, mean of {len(runs)} packings" if len(runs) > 1 else "DEM"
    figure, (force_ax, gap_ax, cell_ax) = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    panels = ((force_ax, force, RAIL_SEAT_COLOUR), (gap_ax, gap_percent, RAIL_SEAT_COLOUR),
              (cell_ax, pressure[:, :, 0], RAIL_SEAT_COLOUR), (cell_ax, pressure[:, :, 1], CENTRE_COLOUR))
    for ax, values, colour in panels:
        if len(runs) == 1:
            continue
        for k, row in enumerate(values):
            # the pressure panel shows two places, so its thin lines stay out of its legend
            label = "DEM, each packing" if k == 0 and ax is not cell_ax else None
            ax.plot(axis, row, color=colour, linewidth=0.8, alpha=0.3, label=label)
    force_ax.plot(axis, force.mean(axis=0), color=RAIL_SEAT_COLOUR, linewidth=2.0, label=mean_label)
    gap_ax.plot(axis, gap_percent.mean(axis=0), color=RAIL_SEAT_COLOUR, linewidth=2.0, label=mean_label)
    cell_ax.plot(axis, pressure[:, :, 0].mean(axis=0), color=RAIL_SEAT_COLOUR, linewidth=2.0, label="rail seat, DEM")
    cell_ax.plot(axis, pressure[:, :, 1].mean(axis=0), color=CENTRE_COLOUR, linewidth=2.0, label="centre, DEM")

    lab: Dict[str, Any] = dict(linestyle="none", markerfacecolor="none", markersize=7, markeredgewidth=1.5)
    force_ax.plot(PAPER_CYCLES, PAPER_FORCE, marker="o", color=RAIL_SEAT_COLOUR, label=PAPER_LABEL, **lab)
    gap_ax.plot(PAPER_CYCLES,
                100 * np.array(PAPER_GAP) / PAPER_STROKE,
                marker="o",
                color=RAIL_SEAT_COLOUR,
                label=PAPER_LABEL,
                **lab)
    cell_ax.plot(PAPER_CYCLES,
                 PAPER_PRESSURE_LEFT,
                 marker="o",
                 color=RAIL_SEAT_COLOUR,
                 label="left rail seat, lab",
                 **lab)
    cell_ax.plot(PAPER_CYCLES,
                 PAPER_PRESSURE_RIGHT,
                 marker="s",
                 color=RAIL_SEAT_COLOUR,
                 label="right rail seat, lab",
                 **lab)
    cell_ax.plot(PAPER_CYCLES, PAPER_PRESSURE_CENTRE, marker="o", color=CENTRE_COLOUR, label="centre, lab", **lab)
    style_axes(force_ax, "peak rail seat force [kN]", "(a) force on the rail seat")
    style_axes(gap_ax, "gap under the sleeper [% of the stroke]", "(b) gap under the sleeper", "upper left")
    style_axes(cell_ax, "peak pressure increase on the subgrade [kPa]", "(c) pressure under the ballast", "upper right")
    figure.suptitle("G44 concrete sleeper in displacement control", fontsize=11)
    figure.savefig(os.path.join(output_dir, "comparison.png"), dpi=200)
    plt.close(figure)

    lines = [f"{len(runs)} packings, {common} cycles completed by all; laboratory stroke {PAPER_STROKE:.2f} mm"]
    for run in runs:
        lines.append(f"packing {int(run['packing'])}: stroke {run['stroke'] * 1e3:.2f} mm, {int(run['cycles'])} cycles")
    for cycle in (1, 10, 100, 1000):
        if cycle > common:
            break
        lab_row = PAPER_CYCLES.index(cycle)
        c = cycle - 1
        lines.append(
            f"cycle {cycle}: peak force {force[:, c].mean():.1f} kN (lab {PAPER_FORCE[lab_row]:.1f}), gap "
            f"{gap_percent[:, c].mean():.1f} % of the stroke (lab {100 * PAPER_GAP[lab_row] / PAPER_STROKE:.1f}), "
            f"rail seat {pressure[:, c, 0].mean():.1f} kPa (lab {PAPER_PRESSURE_LEFT[lab_row]:.1f}/"
            f"{PAPER_PRESSURE_RIGHT[lab_row]:.1f}), centre {pressure[:, c, 1].mean():.1f} kPa "
            f"(lab {PAPER_PRESSURE_CENTRE[lab_row]:.1f})")
    lines.append(f"figure and per-cycle table in {output_dir}")
    return "\n".join(lines)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: arguments; None reads those of the process.

    Returns:
        The arguments, with ``packing`` resolved from ``task`` and the defaults of the shape of the grains filled in.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--grains",
                        choices=GRAINS,
                        default="spheres",
                        help="shape of the grains, which also sets the default output directory (default spheres)")
    common.add_argument("--output-dir",
                        help=f"directory of the results (default {OUTPUT_DIRS['spheres']}, or "
                        f"{OUTPUT_DIRS['polyhedra']} for polyhedra)")
    running = argparse.ArgumentParser(add_help=False)
    running.add_argument("--device",
                         choices=DEVICES,
                         default="cpu",
                         help="device that measures the overlaps of polyhedra; gpu needs CuPy (default cpu)")
    running.add_argument("--save-every-n-steps",
                         type=int,
                         help=f"time steps between two ParaView exports, 0 for none (default {SAVE_EVERY_N_STEPS}, "
                         f"or {POLYHEDRA_SAVE_EVERY_N_STEPS} for polyhedra)")
    loading = argparse.ArgumentParser(add_help=False)
    loading.add_argument("--target",
                         type=float,
                         default=TARGET / 1e3,
                         help=f"rail seat load at the top of the first cycle, which sets the stroke [kN] "
                         f"(default {TARGET / 1e3:g})")
    loading.add_argument("--frequency",
                         type=float,
                         default=FREQUENCY,
                         help=f"frequency of the cycles [Hz] (default {FREQUENCY:g})")
    loading.add_argument("--cycles",
                         type=int,
                         default=CYCLES,
                         help=f"number of cycles; a finished run given more is continued (default {CYCLES})")

    deposit_parser = commands.add_parser("deposit", parents=[common, running], help="deposit one packing")
    which = deposit_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--packing", type=int, help="index of the packing")
    which.add_argument("--task", type=int, help="index of an array task, the packing")

    load_parser = commands.add_parser("load", parents=[common, running, loading], help="test one packing")
    which = load_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--packing", type=int, help="index of the packing")
    which.add_argument("--task", type=int, help="index of an array task, the packing")

    all_parser = commands.add_parser("all",
                                     parents=[common, running, loading],
                                     help="deposit and test every packing, on this machine")
    all_parser.add_argument("--packings", type=int, default=PACKINGS, help=f"number of packings (default {PACKINGS})")
    all_parser.add_argument("--workers",
                            type=int,
                            default=os.cpu_count() or 1,
                            help="number of parallel processes (default: one per core)")

    commands.add_parser("plot", parents=[common], help="compare the tests with the laboratory test")

    arguments = parser.parse_args(argv)
    if arguments.output_dir is None:
        arguments.output_dir = OUTPUT_DIRS[arguments.grains]
    if arguments.command == "plot":
        return arguments
    if arguments.save_every_n_steps is None:
        polyhedra = arguments.grains == "polyhedra"
        arguments.save_every_n_steps = POLYHEDRA_SAVE_EVERY_N_STEPS if polyhedra else SAVE_EVERY_N_STEPS
    if arguments.save_every_n_steps < 0:
        parser.error(f"--save-every-n-steps must be non-negative, got {arguments.save_every_n_steps}.")
    if arguments.command in ("deposit", "load") and arguments.task is not None:
        if arguments.task < 0:
            parser.error(f"--task must be non-negative, got {arguments.task}.")
        arguments.packing = arguments.task
    if arguments.command != "deposit" and arguments.cycles < 1:
        parser.error(f"--cycles must be at least 1, got {arguments.cycles}.")
    return arguments


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the command given on the command line.

    Args:
        argv: arguments; None reads those of the process.
    """
    arguments = parse_arguments(argv)
    if arguments.command == "plot":
        print(plot(arguments.output_dir))
        return
    if arguments.command == "deposit":
        print(deposit(arguments.packing, arguments.output_dir, arguments.save_every_n_steps, arguments.grains,
                      arguments.device),
              flush=True)
        return
    settings = FatigueSettings(arguments.target * 1e3, arguments.frequency)
    if arguments.command == "load":
        print(
            run_test(arguments.packing, settings, arguments.cycles, arguments.output_dir, arguments.save_every_n_steps,
                     arguments.device))
        return
    run_all(arguments.packings, settings, arguments.cycles, arguments.output_dir, arguments.workers,
            arguments.save_every_n_steps, arguments.grains, arguments.device)


if __name__ == "__main__":
    main()
