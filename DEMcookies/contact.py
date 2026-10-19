"""Hertz-Mindlin (no-slip) contact model for the DEMcookies 2D DEM.

The model follows the classic non-linear formulation used in most DEM codes (e.g. LIGGGHTS / EDEM):

* normal force: F_n = 4/3 E* sqrt(R*) delta_n^(3/2) (Hertz);
* tangential force: F_t = -S_t xi_t (Mindlin spring);
* viscous damping proportional to the contact stiffness and set by the coefficient of restitution;
* Coulomb limit: |F_t| <= mu F_n (sliding friction);
* optional rolling resistance: an elastic-plastic spring with a dashpot between the rotations of the two bodies,
  limited to |M_r| <= mu_r R* F_n (the EPSD, or type C, model of Ai et al., 2011).

Rolling resistance stands in for the shape of the grains. Round particles roll past each other far too easily, so a
packing of discs or spheres is much weaker than one of angular grains such as railway ballast; resisting the relative
rotation of the particles in contact restores much of that strength.

All particles and boundaries share one material, so the effective moduli ``E*`` and ``G*`` and the damping factor
``beta`` are computed once. The effective radius ``R*``, the effective mass ``m*`` and the friction coefficient may
differ from contact to contact and are supplied by the solver. Contacts are evaluated in batches.
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .globals import DAMPING_CONSTANT, MIN_RESTITUTION, RAYLEIGH_OFFSET, RAYLEIGH_POISSON_COEFFICIENT


@dataclass(frozen=True)
class MaterialProperties:
    """Bulk material properties shared by every particle and boundary.

    Attributes:
        density: mass density [kg/m^3].
        youngs_modulus: Young's modulus [Pa].
        poisson_ratio: Poisson ratio.
        restitution: coefficient of restitution, in [0, 1].
        friction: Coulomb friction coefficient.
        rolling_friction: rolling resistance coefficient mu_r, the largest rolling moment over R* F_n. Zero, the
            default, lets particles roll freely; values of 0.05 to 0.5 emulate increasingly angular grains.
    """

    density: float
    youngs_modulus: float
    poisson_ratio: float
    restitution: float
    friction: float
    rolling_friction: float = 0.0

    def __post_init__(self) -> None:
        """Validate the material.

        Raises:
            ValueError: if the density is not positive.
            ValueError: if Young's modulus is not positive.
            ValueError: if the Poisson ratio is not in (-1, 0.5).
            ValueError: if the restitution is not in [0, 1].
            ValueError: if the friction coefficient is negative.
            ValueError: if the rolling friction coefficient is negative.
        """
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}.")
        if self.youngs_modulus <= 0.0:
            raise ValueError(f"youngs_modulus must be positive, got {self.youngs_modulus}.")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError(f"poisson_ratio must be in (-1, 0.5), got {self.poisson_ratio}.")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(f"restitution must be in [0, 1], got {self.restitution}.")
        if self.friction < 0.0:
            raise ValueError(f"friction must be non-negative, got {self.friction}.")
        if self.rolling_friction < 0.0:
            raise ValueError(f"rolling_friction must be non-negative, got {self.rolling_friction}.")

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2 (1 + nu)) [Pa]."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))


def rayleigh_time_step(material: MaterialProperties, min_radius: float, safety: float = 0.2) -> float:
    """Return a stable integration time step.

    Uses a fraction of the Rayleigh critical time step, the time a surface wave needs to cross the smallest particle.

    Args:
        material: particle material.
        min_radius: radius of the smallest particle [m].
        safety: fraction of the Rayleigh time step to use.

    Returns:
        The time step [s].

    Raises:
        ValueError: if the radius is not positive.
        ValueError: if the safety factor is not positive.
    """
    if min_radius <= 0.0:
        raise ValueError(f"min_radius must be positive, got {min_radius}.")
    if safety <= 0.0:
        raise ValueError(f"safety must be positive, got {safety}.")
    wave_speed_ratio = RAYLEIGH_POISSON_COEFFICIENT * material.poisson_ratio + RAYLEIGH_OFFSET
    t_rayleigh = math.pi * min_radius * math.sqrt(material.density / material.shear_modulus) / wave_speed_ratio
    return safety * t_rayleigh


@dataclass(frozen=True, eq=False)
class ContactForces:
    """Result of evaluating a batch of contacts.

    Attributes:
        force: total contact force on body i, shape (k, 2) [N].
        tangential_force: tangential part of the force, used for the torque, shape (k, 2) [N].
        tangential_spring: updated tangential displacement of each contact, shape (k, 2) [m].
        normal_force: magnitude of the normal force, shape (k,) [N].
        normal_stiffness: tangent normal stiffness S_n = 2 E* a of each contact, zero when it is open, shape (k,)
            [N/m].
    """

    force: npt.NDArray[np.float64]
    tangential_force: npt.NDArray[np.float64]
    tangential_spring: npt.NDArray[np.float64]
    normal_force: npt.NDArray[np.float64]
    normal_stiffness: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class RollingMoments:
    """Result of evaluating the rolling resistance of a batch of contacts.

    Attributes:
        moment: rolling moment on body i; body j receives the opposite, shape (k,) [N m].
        rolling_spring: updated relative rotation stored in each contact, shape (k,) [rad].
    """

    moment: npt.NDArray[np.float64]
    rolling_spring: npt.NDArray[np.float64]


class HertzMindlinModel:
    """Stateless evaluator of the Hertz-Mindlin contact force.

    The tangential displacement history lives in the solver; this class turns the kinematic state of a batch of
    contacts into forces and updated tangential springs.
    """

    def __init__(self, material: MaterialProperties) -> None:
        """Pre-compute the material constants of the contact law.

        Args:
            material: material shared by the particles and boundaries.
        """
        e = material.youngs_modulus
        nu = material.poisson_ratio
        g = material.shear_modulus

        # effective moduli of two identical bodies in contact
        self.e_star = e / (2.0 * (1.0 - nu * nu))
        self.g_star = g / (2.0 * (2.0 - nu))

        restitution = min(max(material.restitution, MIN_RESTITUTION), 1.0)
        ln_e = math.log(restitution)
        self.beta = abs(ln_e / math.sqrt(ln_e * ln_e + math.pi * math.pi))
        self.friction = material.friction
        self.rolling_friction = material.rolling_friction

    def evaluate(self,
                 normal: npt.NDArray[np.float64],
                 overlap: npt.NDArray[np.float64],
                 relative_velocity: npt.NDArray[np.float64],
                 tangential_spring: npt.NDArray[np.float64],
                 time_step: float,
                 effective_radius: npt.NDArray[np.float64],
                 effective_mass: npt.NDArray[np.float64],
                 friction: Optional[Union[float, npt.NDArray[np.float64]]] = None) -> ContactForces:
        """Evaluate a batch of k contacts between bodies i and j.

        Args:
            normal: unit normals pointing from body j towards body i, shape (k, 2).
            overlap: normal overlaps, positive in contact, shape (k,) [m].
            relative_velocity: velocity of the contact point of i relative to j, shape (k, 2) [m/s].
            tangential_spring: stored tangential displacements, shape (k, 2) [m].
            time_step: increment over which the relative velocity is accumulated into the spring; zero re-evaluates
                the forces of the current state without advancing the history [s].
            effective_radius: effective radii R*, shape (k,) [m].
            effective_mass: effective masses m*, shape (k,) [kg].
            friction: Coulomb friction coefficient per contact or for all contacts; None uses the material's.

        Returns:
            The contact forces on body i and the updated tangential springs. Contacts with a non-positive overlap
            get zero force and a reset spring.
        """
        mu = self.friction if friction is None else friction
        in_contact = overlap > 0.0
        overlap = np.where(in_contact, overlap, 0.0)

        contact_radius = np.sqrt(effective_radius * overlap)
        s_n = 2.0 * self.e_star * contact_radius
        s_t = 8.0 * self.g_star * contact_radius
        damping = DAMPING_CONSTANT * self.beta

        # normal force: Hertz spring (2/3 S_n delta = 4/3 E* sqrt(R*) delta^1.5) plus dashpot, never cohesive
        v_n = np.einsum("ij,ij->i", relative_velocity, normal)
        f_n = np.maximum((2.0 / 3.0) * s_n * overlap - damping * np.sqrt(s_n * effective_mass) * v_n, 0.0)

        # tangential spring: rotate it into the current tangent plane, then add this step's sliding
        v_t = relative_velocity - v_n[:, None] * normal
        spring = tangential_spring - np.einsum("ij,ij->i", tangential_spring, normal)[:, None] * normal
        spring = spring + v_t * time_step

        f_t = -s_t[:, None] * spring - (damping * np.sqrt(s_t * effective_mass))[:, None] * v_t
        f_t_magnitude = np.hypot(f_t[:, 0], f_t[:, 1])
        f_t_limit = mu * f_n
        sliding = f_t_magnitude > f_t_limit
        if np.any(sliding):
            # project onto the friction cone and reset the spring so it stays consistent with the capped force
            limit = np.broadcast_to(f_t_limit, f_t_magnitude.shape)[sliding]
            f_t[sliding] *= (limit / f_t_magnitude[sliding])[:, None]
            spring[sliding] = -f_t[sliding] / s_t[sliding][:, None]

        spring[~in_contact] = 0.0
        force = f_n[:, None] * normal + f_t
        return ContactForces(force=force,
                             tangential_force=f_t,
                             tangential_spring=spring,
                             normal_force=f_n,
                             normal_stiffness=np.where(in_contact, s_n, 0.0))

    def rolling(self, contact: ContactForces, relative_spin: npt.NDArray[np.float64],
                rolling_spring: npt.NDArray[np.float64], time_step: float, effective_radius: npt.NDArray[np.float64],
                rolling_inertia: npt.NDArray[np.float64]) -> RollingMoments:
        """Evaluate the rolling resistance of a batch of k contacts between bodies i and j.

        The relative rotation of the two bodies is stored in a spring of stiffness k_r = 2.25 mu_r^2 R*^2 S_n, damped
        like the other springs of the contact law, and its moment is capped at mu_r R* F_n. A capped spring is reset
        to the moment it carries, so the contact rolls plastically and, when the rotation reverses, unloads
        elastically (Ai et al., 2011).

        Args:
            contact: forces of the same contacts, from ``evaluate``.
            relative_spin: angular velocity of body i minus that of body j, shape (k,) [rad/s].
            rolling_spring: stored relative rotations, shape (k,) [rad].
            time_step: increment over which the relative spin is accumulated into the spring; zero re-evaluates the
                moments of the current state without advancing the history [s].
            effective_radius: effective radii R*, shape (k,) [m].
            rolling_inertia: effective moment of inertia of the two bodies about the contact point, shape (k,)
                [kg m^2].

        Returns:
            The rolling moments on body i and the updated springs. Open contacts, and every contact when the
            rolling friction is zero, get no moment and a reset spring.
        """
        mu_r = self.rolling_friction
        stiffness = 2.25 * mu_r * mu_r * effective_radius * effective_radius * contact.normal_stiffness
        spring = rolling_spring + relative_spin * time_step
        damping = DAMPING_CONSTANT * self.beta * np.sqrt(stiffness * rolling_inertia)
        moment = -stiffness * spring - damping * relative_spin

        limit = mu_r * effective_radius * contact.normal_force
        rolling = np.abs(moment) > limit
        # an open contact, or a free one, has a zero stiffness and must not divide by it
        loaded = rolling & (stiffness > 0.0)
        moment[rolling] = np.sign(moment[rolling]) * limit[rolling]
        spring[loaded] = -moment[loaded] / stiffness[loaded]
        spring[stiffness <= 0.0] = 0.0
        return RollingMoments(moment=moment, rolling_spring=spring)
