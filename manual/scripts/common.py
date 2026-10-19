"""Paths and analysis helpers shared by the scripts that produce the results of the manual."""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Tuple

import numpy as np
import numpy.typing as npt

from DEMcookies import DEMSimulation, MaterialProperties
from DEMcookies.neighbours import find_close_pairs

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "manual" / "figures"


def load_script(path: Path) -> ModuleType:
    """Import a script that is not part of a package, such as ``main.py`` or an example.

    Args:
        path: path of the script.

    Returns:
        The imported module.

    Raises:
        ImportError: if the script cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    # registered so that worker processes can unpickle the functions the script hands them
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def particle_contacts(sim: DEMSimulation) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the overlap of every particle-particle contact and the mean radius of its two particles.

    Args:
        sim: simulation.

    Returns:
        The overlaps [m] and the mean radii of the contacting pairs [m].
    """
    first, second = find_close_pairs(sim.pos, sim.radius, 0.0)
    distance = np.hypot(*(sim.pos[first] - sim.pos[second]).T)
    overlap = sim.radius[first] + sim.radius[second] - distance
    touching = overlap > 0.0
    mean_radius = 0.5 * (sim.radius[first] + sim.radius[second])
    return overlap[touching], mean_radius[touching]


def hertz_force(material: MaterialProperties, effective_radius: npt.NDArray[np.float64],
                overlap: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the elastic Hertz force of contacts at rest, without the damping part.

    Args:
        material: particle material.
        effective_radius: effective radius of each contact [m].
        overlap: overlap of each contact [m].

    Returns:
        The normal force of each contact [N].
    """
    e_star = material.youngs_modulus / (2.0 * (1.0 - material.poisson_ratio**2))
    return np.asarray(4.0 / 3.0 * e_star * np.sqrt(effective_radius) * np.maximum(overlap, 0.0)**1.5)


def floor_force(sim: DEMSimulation, floor: float = 0.0) -> Tuple[float, int]:
    """Return the vertical force a horizontal floor carries and its number of contacts, for a packing at rest.

    Args:
        sim: simulation.
        floor: height of the floor [m].

    Returns:
        The force on the floor [N] and the number of floor contacts.
    """
    overlap = sim.radius - (sim.pos[:, 1] - floor)
    touching = overlap > 0.0
    force = hertz_force(sim.material, sim.radius[touching], overlap[touching])
    return float(force.sum()), int(touching.sum())


def total_weight(sim: DEMSimulation) -> float:
    """Return the weight of all particles [N]."""
    return float(np.sum(sim.mass) * sim.settings.gravity)


def static_overlap(material: MaterialProperties, effective_radius: float, force: float) -> float:
    """Return the Hertz overlap of a contact carrying a given force [m].

    Args:
        material: particle material.
        effective_radius: effective radius of the contact [m].
        force: normal force [N].

    Returns:
        The overlap [m].
    """
    e_star = material.youngs_modulus / (2.0 * (1.0 - material.poisson_ratio**2))
    return float((3.0 * force / (4.0 * e_star * math.sqrt(effective_radius)))**(2.0 / 3.0))
