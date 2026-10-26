"""Hertz-Mindlin (no-slip) contact model for the DEMcookies DEM, in two and in three dimensions.

The model follows the classic non-linear formulation used in most DEM codes (e.g. LIGGGHTS / EDEM):

* normal force: F_n = 4/3 E* sqrt(R*) delta_n^(3/2) (Hertz);
* tangential force: F_t = -S_t xi_t (Mindlin spring);
* viscous damping proportional to the contact stiffness and set by the coefficient of restitution;
* Coulomb limit: |F_t| <= mu F_n (sliding friction);
* optional rolling resistance: an elastic-plastic spring with a dashpot between the rotations of the two bodies,
  limited to |M_r| <= mu_r R* F_n (the EPSD, or type C, model of Ai et al., 2011).

Vectors have two components in 2D and three in 3D. Rotations are scalars in 2D, about the out-of-plane axis, and
vectors in 3D, where only the rolling part of the relative rotation, about an axis in the contact plane, is resisted;
twisting about the contact normal is free.

Rolling resistance stands in for the shape of the grains. Round particles roll past each other far too easily, so a
packing of discs or spheres is much weaker than one of angular grains such as railway ballast; resisting the relative
rotation of the particles in contact restores much of that strength.

Polygonal and polyhedral particles follow the same law, with the contact radius ``a`` measured on the region where
the two bodies overlap rather than computed from their curvatures (``evaluate_overlap``). Their shape resists rolling
by itself, so they get no rolling resistance; the flat contact between two faces resists twisting instead, which a
spherical point contact does not (``twisting``).

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
from .vectors import norm


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


def overlap_contact_radius(size: npt.NDArray[np.float64], dimension: int) -> npt.NDArray[np.float64]:
    """Contact radius of contacts between polytopes, from the size of their contact surface.

    The overlap of two spheres is a lens whose rim has the radius sqrt(2 R* delta), and whose section through the
    centres has a chord of twice that; the Hertz contact radius is sqrt(R* delta). The radius of polytope contacts
    keeps that ratio: a = sqrt(A / (2 pi)) for a contact surface of area A in 3D, and a = L / (2 sqrt 2) for a chord of
    length L in 2D.

    Args:
        size: area (3D) or chord length (2D) of each contact surface, shape (k,) [m^2 or m].
        dimension: number of space dimensions, 2 or 3.

    Returns:
        The contact radii, shape (k,) [m].
    """
    if dimension == 2:
        return np.asarray(size / (2.0 * math.sqrt(2.0)))
    return np.asarray(np.sqrt(np.maximum(size, 0.0) / (2.0 * math.pi)))


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
        force: total contact force on body i, shape (k, d) [N].
        tangential_force: tangential part of the force, used for the torque, shape (k, d) [N].
        tangential_spring: updated tangential displacement of each contact, shape (k, d) [m].
        normal_force: magnitude of the normal force, shape (k,) [N].
        normal_stiffness: tangent normal stiffness S_n = 2 E* a of each contact, zero when it is open, shape (k,)
            [N/m].
        normal: unit normal of each contact, pointing from body j towards body i, shape (k, d).
    """

    force: npt.NDArray[np.float64]
    tangential_force: npt.NDArray[np.float64]
    tangential_spring: npt.NDArray[np.float64]
    normal_force: npt.NDArray[np.float64]
    normal_stiffness: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class RollingMoments:
    """Result of evaluating the rolling resistance, or the twisting resistance, of a batch of contacts.

    Attributes:
        moment: rolling (or twisting) moment on body i; body j receives the opposite, shape (k,) in 2D or (k, 3) in
            3D [N m].
        rolling_spring: updated relative rotation stored in each contact, of the shape of the moment [rad].
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
            normal: unit normals pointing from body j towards body i, shape (k, d).
            overlap: normal overlaps, positive in contact, shape (k,) [m].
            relative_velocity: velocity of the contact point of i relative to j, shape (k, d) [m/s].
            tangential_spring: stored tangential displacements, shape (k, d) [m].
            time_step: increment over which the relative velocity is accumulated into the spring; zero re-evaluates
                the forces of the current state without advancing the history [s].
            effective_radius: effective radii R*, shape (k,) [m].
            effective_mass: effective masses m*, shape (k,) [kg].
            friction: Coulomb friction coefficient per contact or for all contacts; None uses the material's.

        Returns:
            The contact forces on body i and the updated tangential springs. Contacts with a non-positive overlap
            get zero force and a reset spring.
        """
        in_contact = overlap > 0.0
        overlap = np.where(in_contact, overlap, 0.0)

        contact_radius = np.sqrt(effective_radius * overlap)
        s_n = 2.0 * self.e_star * contact_radius
        s_t = 8.0 * self.g_star * contact_radius
        # Hertz spring: 2/3 S_n delta = 4/3 E* sqrt(R*) delta^1.5
        return self.__respond(normal, (2.0 / 3.0) * s_n * overlap, s_n, s_t, in_contact, relative_velocity,
                              tangential_spring, time_step, effective_mass, friction)

    def evaluate_overlap(self,
                         normal: npt.NDArray[np.float64],
                         depth: npt.NDArray[np.float64],
                         contact_radius: npt.NDArray[np.float64],
                         relative_velocity: npt.NDArray[np.float64],
                         tangential_spring: npt.NDArray[np.float64],
                         time_step: float,
                         effective_mass: npt.NDArray[np.float64],
                         share: Optional[npt.NDArray[np.float64]] = None,
                         friction: Optional[Union[float, npt.NDArray[np.float64]]] = None) -> ContactForces:
        """Evaluate a batch of k contacts of polygonal or polyhedral particles from the region where they overlap.

        Polytopes have no curvature to compute the contact radius from, so it is measured on the overlap instead
        (``overlap_contact_radius``), and the law is otherwise that of Hertz and Mindlin: S_n = 2 E* a and
        S_t = 8 G* a, with the same damping and friction. The elastic normal force is kappa S_n delta, with delta
        the mean depth of the overlap, its volume over its contact area (its area over its chord in 2D), and
        kappa = 4/3 in 3D and 1 in 2D. For two spheres this is the Hertz force, and it is within a few percent of
        the exact force of a cone pressed into a plane and of a flat punch.

        A contact surface split between several coplanar facets is one contact: every part has the contact radius of
        the whole surface, and the share of the stiffness and of the mass that its area makes up, so that the parts
        add up to the force, the stiffness and the damping of the whole.

        Args:
            normal: unit normals pointing from body j towards body i, shape (k, d).
            depth: mean depth of each overlap, positive in contact, shape (k,) [m].
            contact_radius: contact radius of each contact, shape (k,) [m].
            relative_velocity: velocity of the contact point of i relative to j, shape (k, d) [m/s].
            tangential_spring: stored tangential displacements, shape (k, d) [m].
            time_step: increment over which the relative velocity is accumulated into the spring; zero re-evaluates
                the forces of the current state without advancing the history [s].
            effective_mass: effective masses m*, shape (k,) [kg].
            share: fraction of its contact surface that each contact makes up, shape (k,); None for whole contacts.
            friction: Coulomb friction coefficient per contact or for all contacts; None uses the material's.

        Returns:
            The contact forces on body i and the updated tangential springs. Contacts with a non-positive depth or
            contact radius get zero force and a reset spring.
        """
        in_contact = (depth > 0.0) & (contact_radius > 0.0)
        depth = np.where(in_contact, depth, 0.0)
        weight = np.where(in_contact, contact_radius if share is None else contact_radius * share, 0.0)
        s_n = 2.0 * self.e_star * weight
        s_t = 8.0 * self.g_star * weight
        kappa = 1.0 if normal.shape[1] == 2 else 4.0 / 3.0
        mass = effective_mass if share is None else effective_mass * share
        return self.__respond(normal, kappa * s_n * depth, s_n, s_t, in_contact, relative_velocity, tangential_spring,
                              time_step, mass, friction)

    def __respond(self, normal: npt.NDArray[np.float64], elastic: npt.NDArray[np.float64], s_n: npt.NDArray[np.float64],
                  s_t: npt.NDArray[np.float64], in_contact: npt.NDArray[np.bool_],
                  relative_velocity: npt.NDArray[np.float64], tangential_spring: npt.NDArray[np.float64],
                  time_step: float, effective_mass: npt.NDArray[np.float64],
                  friction: Optional[Union[float, npt.NDArray[np.float64]]]) -> ContactForces:
        """Add the damping and the tangential force to the elastic normal force of a batch of contacts.

        Args:
            normal: unit normals pointing from body j towards body i, shape (k, d).
            elastic: elastic normal force, zero for open contacts, shape (k,) [N].
            s_n: tangent normal stiffness, zero for open contacts, shape (k,) [N/m].
            s_t: tangential stiffness, zero for open contacts, shape (k,) [N/m].
            in_contact: whether each contact is closed, shape (k,).
            relative_velocity: velocity of the contact point of i relative to j, shape (k, d) [m/s].
            tangential_spring: stored tangential displacements, shape (k, d) [m].
            time_step: increment over which the relative velocity is accumulated into the spring [s].
            effective_mass: effective masses m*, shape (k,) [kg].
            friction: Coulomb friction coefficient per contact or for all contacts; None uses the material's.

        Returns:
            The contact forces on body i and the updated tangential springs.
        """
        mu = self.friction if friction is None else friction
        damping = DAMPING_CONSTANT * self.beta

        # normal force: elastic spring plus dashpot, never cohesive
        v_n = np.einsum("ij,ij->i", relative_velocity, normal)
        f_n = np.maximum(elastic - damping * np.sqrt(s_n * effective_mass) * v_n, 0.0)

        # tangential spring: rotate it into the current tangent plane, then add this step's sliding
        v_t = relative_velocity - v_n[:, None] * normal
        spring = tangential_spring - np.einsum("ij,ij->i", tangential_spring, normal)[:, None] * normal
        spring = spring + v_t * time_step

        f_t = -s_t[:, None] * spring - (damping * np.sqrt(s_t * effective_mass))[:, None] * v_t
        f_t_magnitude = norm(f_t)
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
                             normal_stiffness=np.where(in_contact, s_n, 0.0),
                             normal=normal)

    def rolling(self, contact: ContactForces, relative_spin: npt.NDArray[np.float64],
                rolling_spring: npt.NDArray[np.float64], time_step: float, effective_radius: npt.NDArray[np.float64],
                rolling_inertia: npt.NDArray[np.float64]) -> RollingMoments:
        """Evaluate the rolling resistance of a batch of k contacts between bodies i and j.

        The relative rotation of the two bodies is stored in a spring of stiffness k_r = 2.25 mu_r^2 R*^2 S_n, damped
        like the other springs of the contact law, and its moment is capped at mu_r R* F_n. A capped spring is reset
        to the moment it carries, so the contact rolls plastically and, when the rotation reverses, unloads
        elastically (Ai et al., 2011). In 3D the spin and the spring are vectors: only their components in the
        contact plane, which roll the bodies over each other, are kept, the spring is projected onto the current
        contact plane as the tangential spring is, and the magnitude of the moment is capped.

        Args:
            contact: forces of the same contacts, from ``evaluate``.
            relative_spin: angular velocity of body i minus that of body j, shape (k,) in 2D or (k, 3) in 3D
                [rad/s].
            rolling_spring: stored relative rotations, of the shape of the spin [rad].
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
        damping = DAMPING_CONSTANT * self.beta * np.sqrt(stiffness * rolling_inertia)
        limit = mu_r * effective_radius * contact.normal_force
        if relative_spin.ndim == 2:
            return self.__rolling_3d(contact.normal, relative_spin, rolling_spring, time_step, stiffness, damping,
                                     limit)
        spring = rolling_spring + relative_spin * time_step
        moment = -stiffness * spring - damping * relative_spin

        rolling = np.abs(moment) > limit
        # an open contact, or a free one, has a zero stiffness and must not divide by it
        loaded = rolling & (stiffness > 0.0)
        moment[rolling] = np.sign(moment[rolling]) * limit[rolling]
        spring[loaded] = -moment[loaded] / stiffness[loaded]
        spring[stiffness <= 0.0] = 0.0
        return RollingMoments(moment=moment, rolling_spring=spring)

    def twisting(self,
                 contact: ContactForces,
                 relative_spin: npt.NDArray[np.float64],
                 twisting_spring: npt.NDArray[np.float64],
                 time_step: float,
                 contact_radius: npt.NDArray[np.float64],
                 twisting_inertia: npt.NDArray[np.float64],
                 friction: Optional[Union[float, npt.NDArray[np.float64]]] = None) -> RollingMoments:
        """Evaluate the resistance to twisting of a batch of k 3D contacts between polytopes.

        The flat contact between two faces resists a relative spin about its normal, as a spherical point contact
        does not. Its tangential stiffness S_t, spread evenly over a disc of the contact radius a, gives the twisting
        spring the stiffness k_t = S_t a^2 / 2; it is damped like the other springs, and its moment is capped at
        2/3 mu F_n a, the friction moment of an even pressure on that disc. A capped spring is reset to the moment it
        carries, as the rolling spring is.

        Args:
            contact: forces of the same contacts, from ``evaluate_overlap``.
            relative_spin: angular velocity of body i minus that of body j, shape (k, 3) [rad/s].
            twisting_spring: stored relative twist, a vector along the normal, shape (k, 3) [rad].
            time_step: increment over which the relative spin is accumulated into the spring; zero re-evaluates the
                moments of the current state without advancing the history [s].
            contact_radius: contact radius of each contact, shape (k,) [m].
            twisting_inertia: effective moment of inertia of the two bodies about the normal, shape (k,) [kg m^2].
            friction: Coulomb friction coefficient per contact or for all contacts; None uses the material's.

        Returns:
            The twisting moments on body i, along the normals, and the updated springs. Open contacts get no moment
            and a reset spring.
        """
        mu = self.friction if friction is None else friction
        normal = contact.normal
        # the tangential stiffness is 4 G* / E* times the normal stiffness, as for spheres
        stiffness = 2.0 * (self.g_star / self.e_star) * contact.normal_stiffness * contact_radius**2
        damping = DAMPING_CONSTANT * self.beta * np.sqrt(stiffness * twisting_inertia)
        limit = (2.0 / 3.0) * mu * contact.normal_force * contact_radius
        # only the spin about the normal twists the contact; the rest rolls the bodies over each other
        spin = np.einsum("ij,ij->i", relative_spin, normal)
        spring = np.einsum("ij,ij->i", twisting_spring, normal) + spin * time_step
        moment = -stiffness * spring - damping * spin

        twisting = np.abs(moment) > limit
        loaded = twisting & (stiffness > 0.0)
        moment[twisting] = np.sign(moment[twisting]) * np.broadcast_to(limit, moment.shape)[twisting]
        spring[loaded] = -moment[loaded] / stiffness[loaded]
        spring[stiffness <= 0.0] = 0.0
        return RollingMoments(moment=moment[:, None] * normal, rolling_spring=spring[:, None] * normal)

    @staticmethod
    def __rolling_3d(normal: npt.NDArray[np.float64], relative_spin: npt.NDArray[np.float64],
                     rolling_spring: npt.NDArray[np.float64], time_step: float, stiffness: npt.NDArray[np.float64],
                     damping: npt.NDArray[np.float64], limit: npt.NDArray[np.float64]) -> RollingMoments:
        """Evaluate the rolling resistance of a batch of 3D contacts, with vector spins and springs.

        Args:
            normal: unit normals of the contacts, shape (k, 3).
            relative_spin: angular velocity of body i minus that of body j, shape (k, 3) [rad/s].
            rolling_spring: stored relative rotations, shape (k, 3) [rad].
            time_step: increment over which the rolling spin is accumulated into the spring [s].
            stiffness: rolling stiffness k_r of each contact [N m/rad].
            damping: rolling damping coefficient of each contact [N m s/rad].
            limit: largest rolling moment of each contact [N m].

        Returns:
            The rolling moments on body i and the updated springs.
        """
        # twisting about the normal is not resisted, so only the rolling part of the spin and of the spring is kept
        spin = relative_spin - np.einsum("ij,ij->i", relative_spin, normal)[:, None] * normal
        spring = rolling_spring - np.einsum("ij,ij->i", rolling_spring, normal)[:, None] * normal
        spring = spring + spin * time_step
        moment = -stiffness[:, None] * spring - damping[:, None] * spin

        magnitude = norm(moment)
        rolling = magnitude > limit
        loaded = rolling & (stiffness > 0.0)
        moment[rolling] *= (limit[rolling] / magnitude[rolling])[:, None]
        spring[loaded] = -moment[loaded] / stiffness[loaded][:, None]
        spring[stiffness <= 0.0] = 0.0
        return RollingMoments(moment=moment, rolling_spring=spring)
