"""DEMcookies: a small 2D Discrete Element Model.

Features:
    Circular particles with uniformly distributed random radii.
    Gravity ("raining") deposition to build the initial packing.
    Hertz-Mindlin (no-slip) contact law with tangential history and rotation.
    A polygonal domain whose edges are each a wall, open, or a loaded platen.
    Loaded platens driven by a pulse, Heaviside, sinusoidal or user-defined load history.
    Pore water below a water table, with buoyancy, porosity-dependent drag, optional seepage and an optional
    coupled excess pore pressure for undrained loading and consolidation.
"""

from .boundary import BoundaryCondition, LoadedPlaten, Open, Platen, Wall
from .contact import HertzMindlinModel, MaterialProperties, rayleigh_time_step
from .geometry import Geometry
from .globals import GlobalSettings
from .loads import Heaviside, Pulse, Sinusoidal, TimeHistory
from .particle_generation import Particle, generate_lattice_particles
from .solver import DEMSimulation, Frame, LoadCurve, RainSettings, VelocityVerlet
from .water import Water

__all__ = [
    "BoundaryCondition",
    "DEMSimulation",
    "Frame",
    "Geometry",
    "GlobalSettings",
    "Heaviside",
    "HertzMindlinModel",
    "LoadCurve",
    "LoadedPlaten",
    "MaterialProperties",
    "Open",
    "Particle",
    "Platen",
    "Pulse",
    "RainSettings",
    "Sinusoidal",
    "TimeHistory",
    "VelocityVerlet",
    "Wall",
    "Water",
    "generate_lattice_particles",
    "rayleigh_time_step",
]
