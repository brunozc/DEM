"""DEMcookies: a small 2D and 3D Discrete Element Model.

Features:
    Spherical particles with uniformly distributed random radii, in a plane (2D) or in space (3D).
    Gravity ("raining") deposition to build the initial packing.
    Hertz-Mindlin (no-slip) contact law with tangential history and rotation.
    A polygonal (2D) or polyhedral (3D) domain whose facets are each a wall, open, or a loaded platen.
    Loaded platens driven by a pulse, Heaviside, sinusoidal or user-defined load history.
    Pore water below a water table, with buoyancy, porosity-dependent drag, optional seepage and an optional
    coupled excess pore pressure for undrained loading and consolidation.
    Export of the particles, contacts and boundary to VTK files, to be viewed in ParaView.

The geometry sets the number of dimensions: ``Geometry`` is a polygon and ``Geometry3D`` a polyhedron, with the
y axis vertical in both.
"""

from .boundary import BoundaryCondition, LoadedPlaten, Open, Platen, Wall
from .contact import HertzMindlinModel, MaterialProperties, rayleigh_time_step
from .geometry import Geometry
from .geometry3d import Domain, Geometry3D
from .globals import GlobalSettings
from .loads import Heaviside, Pulse, Sinusoidal, TimeHistory
from .particle_generation import Particle, Particle3D, generate_lattice_particles
from .solver import ContactList, DEMSimulation, Frame, LoadCurve, RainSettings, VelocityVerlet
from .water import Water

__all__ = [
    "BoundaryCondition",
    "ContactList",
    "DEMSimulation",
    "Domain",
    "Frame",
    "Geometry",
    "Geometry3D",
    "GlobalSettings",
    "Heaviside",
    "HertzMindlinModel",
    "LoadCurve",
    "LoadedPlaten",
    "MaterialProperties",
    "Open",
    "Particle",
    "Particle3D",
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
