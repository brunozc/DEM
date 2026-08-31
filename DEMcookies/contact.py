"""Hertz-Mindlin (no-slip) contact model for the DEMcookies 2D DEM.

The model follows the classic non-linear formulation used in most DEM codes
(e.g. LIGGGHTS / EDEM):

* Normal force    : F_n = 4/3 E* sqrt(R*) delta_n^(3/2)   (Hertz)
* Tangential force : F_t = -S_t * xi_t                     (Mindlin spring)
* Viscous damping  : proportional to the contact stiffness and the
  coefficient of restitution.
* Coulomb limit    : |F_t| <= mu * F_n  (sliding friction).

All particles (and the walls/platen) are assumed to share the same material,
so the effective contact constants ``E*``, ``G*``, ``beta`` and ``mu`` are
constant and pre-computed once.  Only the effective radius ``R*`` and the
effective mass ``m*`` change from contact to contact and are supplied by the
solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 2 * sqrt(5/6) -- the constant that appears in the Tsuji/Hertz damping term.
_DAMP_CONST = 2.0 * math.sqrt(5.0 / 6.0)


@dataclass(frozen=True)
class MaterialProperties:
    """Bulk material properties shared by every particle and boundary."""

    density: float          # kg / m^3
    youngs_modulus: float   # Pa
    poisson_ratio: float    # -
    restitution: float      # - coefficient of restitution in [0, 1]
    friction: float         # - Coulomb friction coefficient

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2 (1 + nu))."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))


def rayleigh_time_step(material: MaterialProperties, min_radius: float,
                       safety: float = 0.2) -> float:
    """Return a stable integration time step.

    Uses a fraction (``safety``) of the Rayleigh critical time step, which is
    the time a shear wave needs to travel across the smallest particle.
    """
    g = material.shear_modulus
    rho = material.density
    nu = material.poisson_ratio
    t_rayleigh = math.pi * min_radius * math.sqrt(rho / g) / (0.1631 * nu + 0.8766)
    return safety * t_rayleigh


class HertzMindlinModel:
    """Stateless evaluator of the Hertz-Mindlin contact force.

    The tangential displacement history (``xi``) lives in the solver; this
    class only turns the current kinematic state of a single contact into a
    force and an updated tangential spring.
    """

    def __init__(self, material: MaterialProperties):
        e = material.youngs_modulus
        nu = material.poisson_ratio
        g = material.shear_modulus

        # Effective moduli for two identical bodies in contact.
        self.e_star = e / (2.0 * (1.0 - nu * nu))
        self.g_star = g / (2.0 * (2.0 - nu))

        # Damping ratio from the coefficient of restitution.
        restitution = min(max(material.restitution, 1e-4), 1.0)
        ln_e = math.log(restitution)
        self.beta = abs(ln_e / math.sqrt(ln_e * ln_e + math.pi * math.pi))

        self.mu = material.friction

    def contact_force(self, nx: float, ny: float, delta: float,
                      vrx: float, vry: float, xi_x: float, xi_y: float,
                      dt: float, r_eff: float, m_eff: float):
        """Evaluate one contact.

        Parameters
        ----------
        nx, ny:
            Unit normal pointing from body *j* towards body *i* (the direction
            in which body *i* is pushed away).
        delta:
            Normal overlap (positive in contact).
        vrx, vry:
            Relative surface velocity of *i* with respect to *j*.
        xi_x, xi_y:
            Stored tangential displacement of the contact.
        dt:
            Time step.
        r_eff, m_eff:
            Effective radius and mass of the contact.

        Returns
        -------
        (fx, fy, ftx, fty, xi_x, xi_y)
            Total force on *i*, its tangential part (for the torque) and the
            updated tangential displacement.
        """
        if delta <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        sqrt_rd = math.sqrt(r_eff * delta)
        s_n = 2.0 * self.e_star * sqrt_rd            # normal stiffness
        s_t = 8.0 * self.g_star * sqrt_rd            # tangential stiffness
        k_n = (4.0 / 3.0) * self.e_star * math.sqrt(r_eff)

        # --- normal force ---------------------------------------------------
        v_n = vrx * nx + vry * ny                    # < 0 while approaching
        c_n = _DAMP_CONST * self.beta * math.sqrt(s_n * m_eff)
        f_n = k_n * delta ** 1.5 - c_n * v_n
        if f_n < 0.0:                                # no cohesion
            f_n = 0.0

        # --- tangential force (Mindlin spring + damping) --------------------
        v_tx = vrx - v_n * nx
        v_ty = vry - v_n * ny

        # keep the stored spring in the (rotated) tangent plane
        xi_dot_n = xi_x * nx + xi_y * ny
        xi_x -= xi_dot_n * nx
        xi_y -= xi_dot_n * ny
        # accumulate the incremental tangential displacement
        xi_x += v_tx * dt
        xi_y += v_ty * dt

        c_t = _DAMP_CONST * self.beta * math.sqrt(s_t * m_eff)
        ftx = -s_t * xi_x - c_t * v_tx
        fty = -s_t * xi_y - c_t * v_ty

        # Coulomb sliding limit
        ft_mag = math.hypot(ftx, fty)
        ft_max = self.mu * f_n
        if ft_mag > ft_max:
            if ft_mag > 0.0:
                scale = ft_max / ft_mag
                ftx *= scale
                fty *= scale
            if s_t > 0.0:
                # rescale the spring so it is consistent with the capped force
                xi_x = -ftx / s_t
                xi_y = -fty / s_t

        fx = f_n * nx + ftx
        fy = f_n * ny + fty
        return fx, fy, ftx, fty, xi_x, xi_y
