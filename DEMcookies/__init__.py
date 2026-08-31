"""DEMcookies: a small 2D Discrete Element Model.

Features
--------
* Circular particles with uniformly distributed random radii.
* Gravity ("raining") deposition to build the initial packing.
* Hertz-Mindlin (no-slip) contact law with tangential history and rotation.
* A rigid top platen that applies a time-dependent load.
* A rectangular box ``B x L`` with fixed left, right and bottom walls.
"""

from .boundary import Box, LoadingPlaten, linear_ramp
from .contact import HertzMindlinModel, MaterialProperties, rayleigh_time_step
from .particle_generation import Particle, generate_rain_particles
from .solver import DEMSimulation, VelocityVerlet

__all__ = [
    "Box",
    "LoadingPlaten",
    "linear_ramp",
    "HertzMindlinModel",
    "MaterialProperties",
    "rayleigh_time_step",
    "Particle",
    "generate_rain_particles",
    "DEMSimulation",
    "VelocityVerlet",
]
