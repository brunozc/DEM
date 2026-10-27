"""Benchmark of polyhedral stones on the CPU and on a GPU, to plan the polyhedral runs of the 3D ballast study.

A force evaluation of a bed of polyhedra is most of the cost of a time step, and it grows with the number of contacts.
This script packs the stones of ``ballast_cyclic_3d.py`` on a cubic lattice tight enough that each touches several of
its neighbours, as in the ballast bed, evaluates the forces on every device asked for, checks that the devices agree,
and prints the time per evaluation and the time a load cycle would take at that rate. Run it on the machine that is
to run the study, e.g. on a GPU node of the cluster::

    python examples/benchmark_polyhedra.py --devices cpu gpu

The GPU needs CuPy. The first evaluation on a device also compiles its kernels, so it is left out of the timing. The
time per load cycle counts the force evaluations alone; the rest of a time step adds a little.
"""

import argparse
import math
from time import perf_counter
from typing import List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ballast_cyclic_3d import FREQUENCY, POLYHEDRAL_MATERIAL, RADIUS_RANGE, STONES
from DEMcookies import DEMSimulation, Geometry3D, Particle3D
from DEMcookies.backend import DEVICES
from DEMcookies.polytopes import random_orientations

# spacing of the lattice over the mean diameter of the stones: tight enough for several contacts per stone
LATTICE_SPACING = 0.95


def lattice_bed(count: int, seed: int = 0) -> List[Particle3D]:
    """Pack stones of random shape, size and orientation on a cubic lattice.

    Args:
        count: number of stones.
        seed: seed of the random generator.

    Returns:
        The stones, one lattice spacing away from the walls of a box with a corner at the origin.
    """
    rng = np.random.default_rng(seed)
    spacing = LATTICE_SPACING * (RADIUS_RANGE[0] + RADIUS_RANGE[1])
    side = math.ceil(count**(1.0 / 3.0))
    radii = rng.uniform(*RADIUS_RANGE, count)
    shapes = rng.integers(len(STONES), size=count)
    orientations = random_orientations(rng, count, 3)
    stones = []
    for k in range(count):
        x, y, z = (spacing * (index + 1.0) for index in (k % side, (k // side) % side, k // (side * side)))
        w, qx, qy, qz = (float(value) for value in orientations[k])
        stones.append(Particle3D(x, y, z, float(radii[k]), shape=STONES[shapes[k]], orientation=(w, qx, qy, qz)))
    return stones


def main(stones: int = 6600, devices: Sequence[str] = ("cpu", ), repeat: int = 3) -> None:
    """Time the force evaluation of a bed of stones on each device.

    Args:
        stones: number of stones.
        devices: devices to time, "cpu" and "gpu".
        repeat: number of timed evaluations on each device, after one untimed.
    """
    particles = lattice_bed(stones)
    spacing = LATTICE_SPACING * (RADIUS_RANGE[0] + RADIUS_RANGE[1])
    side = spacing * (math.ceil(stones**(1.0 / 3.0)) + 1.0)
    geometry = Geometry3D.box(side, side, side)
    reference: Optional[npt.NDArray[np.float64]] = None
    for device in devices:
        sim = DEMSimulation(geometry, POLYHEDRAL_MATERIAL, particles=particles, device=device)
        sim.compute_forces()
        start = perf_counter()
        for _ in range(repeat):
            sim.compute_forces()
        seconds = (perf_counter() - start) / repeat
        pairs, walls = sim.contacts()
        steps_per_cycle = 1.0 / (FREQUENCY * sim.dt)
        agreement = ""
        if reference is None:
            reference = sim.force.copy()
        else:
            difference = float(np.abs(sim.force - reference).max() / np.abs(reference).max())
            agreement = f"; forces within {difference:.0e} of those on {devices[0]}"
        print(
            f"{device}: {sim.n} stones, {pairs.first.size} contacts between them and {walls.first.size} with the "
            f"walls; {seconds * 1e3:.0f} ms per force evaluation, {steps_per_cycle:.0f} steps per load cycle at "
            f"{FREQUENCY:g} Hz, so {seconds * steps_per_cycle / 3600:.1f} h per cycle{agreement}",
            flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--stones", type=int, default=6600, help="number of stones (default 6600, a full packing)")
    parser.add_argument("--devices",
                        nargs="+",
                        choices=DEVICES,
                        default=["cpu"],
                        help="devices to time, in order (default cpu)")
    parser.add_argument("--repeat", type=int, default=3, help="number of timed evaluations per device (default 3)")
    arguments = parser.parse_args()
    main(arguments.stones, arguments.devices, arguments.repeat)
