"""Boundaries for the DEMcookies 2D DEM: the box, walls and the loading platen.

The box has three fixed walls (left ``x = 0``, right ``x = B`` and bottom
``y = 0``).  The top is open during deposition and is later closed by a rigid
*loading platen* that applies a time-dependent vertical load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Box:
    """Rectangular container of width ``B`` and height ``L``."""

    width: float    # B
    height: float   # L


class LoadingPlaten:
    """Rigid horizontal plate that loads the packing from the top.

    The platen can only move vertically.  Its motion is driven by the contact
    reaction from the particles below and by an externally applied,
    time-dependent load ``F(t)`` (positive = pushing down).
    """

    def __init__(self, width: float, mass: float, gravity: float = 0.0):
        self.width = width
        self.mass = mass
        self.gravity = gravity      # self-weight per unit mass (0 -> weightless)
        self.y = 0.0                # vertical position of the plate underside
        self.vy = 0.0               # vertical velocity
        self.fy = 0.0               # net contact reaction from particles (+y)
        self.y0 = 0.0               # position when first placed (for displacement)
        self.active = False

    def place(self, top_of_bed: float, gap: float = 0.0) -> None:
        """Position the platen just above the settled bed and switch it on."""
        self.y = top_of_bed + gap
        self.y0 = self.y
        self.vy = 0.0
        self.fy = 0.0
        self.active = True

    @property
    def displacement(self) -> float:
        """Downward displacement since the platen was placed."""
        return self.y0 - self.y


def linear_ramp(f_max: float, t_ramp: float, hold: bool = True) -> Callable[[float], float]:
    """Return ``F(t)`` that ramps linearly from 0 to ``f_max`` over ``t_ramp``.

    After ``t_ramp`` the load is held at ``f_max`` when ``hold`` is true,
    otherwise it drops back to zero.
    """

    def load(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t >= t_ramp:
            return f_max if hold else 0.0
        return f_max * t / t_ramp

    return load
