"""Tests of the material, the time step and the Hertz-Mindlin contact law."""

import math
import re
from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

from DEMcookies.contact import (ContactForces, HertzMindlinModel, MaterialProperties, RollingMoments,
                                overlap_contact_radius, rayleigh_time_step)


def _material(restitution: float = 1.0, friction: float = 0.5, rolling_friction: float = 0.0) -> MaterialProperties:
    """Return the reference material with the given restitution, friction and rolling friction."""
    return MaterialProperties(density=2500.0,
                              youngs_modulus=5.0e6,
                              poisson_ratio=0.25,
                              restitution=restitution,
                              friction=friction,
                              rolling_friction=rolling_friction)


def _single(model: HertzMindlinModel,
            normal: Tuple[float, float],
            overlap: float,
            velocity: Tuple[float, float],
            dt: float = 1.0e-5) -> ContactForces:
    """Evaluate one contact between two 10 mm particles with unit effective mass."""
    return model.evaluate(np.array([normal], dtype=float), np.array([overlap]), np.array([velocity], dtype=float),
                          np.zeros((1, 2)), dt, np.array([0.005]), np.array([1.0]))


def _rolling(model: HertzMindlinModel,
             overlap: float,
             spin: float,
             stored: float,
             dt: float = 1.0e-5) -> Tuple[ContactForces, RollingMoments]:
    """Evaluate the forces and the rolling moment of one contact between two 10 mm particles at rest."""
    contact = _single(model, (0.0, 1.0), overlap, (0.0, 0.0), dt)
    moments = model.rolling(contact, np.array([spin]), np.array([stored]), dt, np.array([0.005]), np.array([1.0e-6]))
    return contact, moments


class TestMaterialProperties:
    """Tests of MaterialProperties."""

    def test_shear_modulus(self) -> None:
        """The shear modulus must be E / (2 (1 + nu))."""
        assert_allclose(_material().shear_modulus, 5.0e6 / 2.5)

    @pytest.mark.parametrize("field, value, message", [
        ("density", 0.0, "density must be positive, got 0.0."),
        ("youngs_modulus", -1.0, "youngs_modulus must be positive, got -1.0."),
        ("poisson_ratio", 0.5, "poisson_ratio must be in (-1, 0.5), got 0.5."),
        ("restitution", 1.5, "restitution must be in [0, 1], got 1.5."),
        ("friction", -0.1, "friction must be non-negative, got -0.1."),
        ("rolling_friction", -0.2, "rolling_friction must be non-negative, got -0.2."),
    ])
    def test_invalid_values_are_rejected(self, field: str, value: float, message: str) -> None:
        """Unphysical material parameters must be rejected with a specific message."""
        values = dict(density=2500.0, youngs_modulus=5.0e6, poisson_ratio=0.25, restitution=0.4, friction=0.5)
        values[field] = value
        with pytest.raises(ValueError, match=re.escape(message)):
            MaterialProperties(**values)


class TestRayleighTimeStep:
    """Tests of rayleigh_time_step."""

    def test_closed_form(self) -> None:
        """The time step must be the safety fraction of pi R sqrt(rho / G) / (0.1631 nu + 0.8766)."""
        material = _material()
        expected = 0.2 * math.pi * 0.004 * math.sqrt(2500.0 / 2.0e6) / (0.1631 * 0.25 + 0.8766)
        assert_allclose(rayleigh_time_step(material, 0.004), expected, rtol=1e-12)

    def test_non_positive_radius_is_rejected(self) -> None:
        """A zero radius has no time scale and must be rejected."""
        with pytest.raises(ValueError, match=re.escape("min_radius must be positive, got 0.0.")):
            rayleigh_time_step(_material(), 0.0)

    def test_non_positive_safety_is_rejected(self) -> None:
        """A non-positive safety factor must be rejected."""
        with pytest.raises(ValueError, match=re.escape("safety must be positive, got -0.1.")):
            rayleigh_time_step(_material(), 0.004, safety=-0.1)


class TestHertzMindlinModel:
    """Tests of the Hertz-Mindlin contact law."""

    def test_effective_moduli(self) -> None:
        """The effective moduli must match the closed-form expressions."""
        material = _material()
        model = HertzMindlinModel(material)
        nu = material.poisson_ratio
        assert_allclose(model.e_star, material.youngs_modulus / (2.0 * (1.0 - nu**2)))
        assert_allclose(model.g_star, material.shear_modulus / (2.0 * (2.0 - nu)))

    def test_hertz_normal_force_magnitude(self) -> None:
        """Without damping or sliding the force must be F = 4/3 E* sqrt(R*) delta^1.5 along the normal."""
        model = HertzMindlinModel(_material(restitution=1.0))
        result = _single(model, (1.0, 0.0), 1.0e-4, (0.0, 0.0))
        expected = (4.0 / 3.0) * model.e_star * math.sqrt(0.005) * 1.0e-4**1.5
        assert_allclose(result.force, [[expected, 0.0]], rtol=1e-12, atol=1e-15)
        assert_allclose(result.tangential_force, [[0.0, 0.0]], atol=1e-15)

    def test_coulomb_friction_limit(self) -> None:
        """Fast sliding must give a tangential force of exactly mu F_n."""
        mu = 0.5
        model = HertzMindlinModel(_material(restitution=1.0, friction=mu))
        result = _single(model, (1.0, 0.0), 1.0e-4, (0.0, 5.0))
        f_n = (4.0 / 3.0) * model.e_star * math.sqrt(0.005) * 1.0e-4**1.5
        assert_allclose(np.hypot(*result.tangential_force[0]), mu * f_n, rtol=1e-9)
        # the spring is reset to carry exactly the capped force
        assert_allclose(result.tangential_spring[0],
                        -result.tangential_force[0] / (8.0 * model.g_star * math.sqrt(0.005 * 1.0e-4)))

    def test_per_contact_friction(self) -> None:
        """A friction array must override the material friction contact by contact."""
        model = HertzMindlinModel(_material(restitution=1.0, friction=0.5))
        normal = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = model.evaluate(normal, np.full(2, 1.0e-4), np.array([[0.0, 5.0], [0.0, 5.0]]), np.zeros((2, 2)),
                                1.0e-5, np.full(2, 0.005), np.ones(2), np.array([0.0, 0.25]))
        f_n = result.force[1, 0]
        assert_allclose(result.tangential_force[0], [0.0, 0.0], atol=1e-15)
        assert_allclose(np.hypot(*result.tangential_force[1]), 0.25 * f_n, rtol=1e-9)

    def test_separated_contact_has_no_force_and_no_history(self) -> None:
        """A non-positive overlap must give zero force and reset the tangential spring."""
        model = HertzMindlinModel(_material())
        result = model.evaluate(np.array([[0.0, 1.0]]), np.array([-1.0e-5]), np.array([[1.0, -1.0]]),
                                np.array([[1.0e-4, 0.0]]), 1.0e-5, np.array([0.005]), np.array([1.0]))
        assert_allclose(result.force, [[0.0, 0.0]], atol=1e-15)
        assert_allclose(result.tangential_spring, [[0.0, 0.0]], atol=1e-15)

    def test_damping_resists_approach(self) -> None:
        """With e < 1, an approaching contact must be pushed apart harder than a static one."""
        model = HertzMindlinModel(_material(restitution=0.4))
        static = _single(model, (1.0, 0.0), 1.0e-4, (0.0, 0.0)).force[0, 0]
        approaching = _single(model, (1.0, 0.0), 1.0e-4, (-0.5, 0.0)).force[0, 0]
        assert approaching > static * (1.0 + 1e-3)

    def test_no_cohesion(self) -> None:
        """A fast separating contact must not attract: the normal force is clipped at zero."""
        model = HertzMindlinModel(_material(restitution=0.1))
        result = _single(model, (1.0, 0.0), 1.0e-8, (50.0, 0.0))
        assert_allclose(result.force, [[0.0, 0.0]], atol=1e-15)

    def test_zero_time_step_does_not_advance_history(self) -> None:
        """A zero time step re-evaluates the state: the spring is only projected onto the tangent plane."""
        model = HertzMindlinModel(_material(restitution=1.0))
        spring = np.array([[1.0e-7, 2.0e-7]])
        result = model.evaluate(np.array([[1.0, 0.0]]), np.array([1.0e-4]), np.array([[0.0, 3.0]]), spring, 0.0,
                                np.array([0.005]), np.array([1.0]))
        assert_allclose(result.tangential_spring, [[0.0, 2.0e-7]], atol=1e-20)

    def test_batch_matches_single_evaluations(self) -> None:
        """Evaluating contacts in one batch must give the same result as evaluating them one by one."""
        model = HertzMindlinModel(_material(restitution=0.4))
        rng = np.random.default_rng(0)
        angles = rng.uniform(0.0, 2.0 * math.pi, 6)
        normal = np.column_stack((np.cos(angles), np.sin(angles)))
        overlap = rng.uniform(-1.0e-5, 2.0e-4, 6)
        velocity = rng.normal(0.0, 0.3, (6, 2))
        spring = rng.normal(0.0, 1.0e-6, (6, 2))
        radius = rng.uniform(0.002, 0.01, 6)
        mass = rng.uniform(0.01, 0.1, 6)
        batch = model.evaluate(normal, overlap, velocity, spring, 1.0e-5, radius, mass)
        for k in range(6):
            single = model.evaluate(normal[k:k + 1], overlap[k:k + 1], velocity[k:k + 1], spring[k:k + 1], 1.0e-5,
                                    radius[k:k + 1], mass[k:k + 1])
            assert_allclose(batch.force[k], single.force[0], rtol=1e-12, atol=1e-18)
            assert_allclose(batch.tangential_spring[k], single.tangential_spring[0], rtol=1e-12, atol=1e-22)


class TestRollingResistance:
    """Tests of the rolling resistance of HertzMindlinModel."""

    def test_elastic_moment_follows_rolling_spring(self) -> None:
        """Below the limit, an undamped contact must resist a stored rotation with k_r = 2.25 mu_r^2 R*^2 S_n."""
        model = HertzMindlinModel(_material(restitution=1.0, rolling_friction=0.3))
        contact, moments = _rolling(model, 1.0e-4, 0.0, 1.0e-4)
        stiffness = 2.25 * 0.3**2 * 0.005**2 * contact.normal_stiffness[0]
        assert_allclose(moments.moment, [-stiffness * 1.0e-4], rtol=1e-12)
        assert abs(moments.moment[0]) < 0.3 * 0.005 * contact.normal_force[0]
        assert_allclose(moments.rolling_spring, [1.0e-4], rtol=1e-12)

    def test_moment_is_capped(self) -> None:
        """A fast relative rotation must mobilise the full rolling moment mu_r R* F_n against it."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        contact, moments = _rolling(model, 1.0e-4, 50.0, 0.0)
        limit = 0.3 * 0.005 * contact.normal_force[0]
        assert_allclose(moments.moment, [-limit], rtol=1e-12)
        # the spring is reset to the capped moment, so a reversal unloads elastically
        stiffness = 2.25 * 0.3**2 * 0.005**2 * contact.normal_stiffness[0]
        assert_allclose(moments.rolling_spring, [limit / stiffness], rtol=1e-12)

    def test_zero_rolling_friction_gives_no_moment(self) -> None:
        """Without rolling friction, a spinning contact must feel no moment and keep no history."""
        model = HertzMindlinModel(_material(restitution=0.4))
        _, moments = _rolling(model, 1.0e-4, 50.0, 1.0e-3)
        assert_allclose(moments.moment, [0.0], atol=1e-30)
        assert_allclose(moments.rolling_spring, [0.0], atol=1e-30)

    def test_open_contact_has_no_moment_and_no_history(self) -> None:
        """A contact that opened must exert no moment and reset its rolling spring."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        _, moments = _rolling(model, -1.0e-5, 50.0, 1.0e-3)
        assert_allclose(moments.moment, [0.0], atol=1e-30)
        assert_allclose(moments.rolling_spring, [0.0], atol=1e-30)

    def test_zero_time_step_does_not_advance_history(self) -> None:
        """A zero time step must leave the stored rotation unchanged however fast the contact spins."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        _, moments = _rolling(model, 1.0e-4, 1.0e-3, 2.0e-5, dt=0.0)
        assert_allclose(moments.rolling_spring, [2.0e-5], rtol=1e-12)


def _rolling_3d(model: HertzMindlinModel,
                normal: Tuple[float, float, float],
                spin: Tuple[float, float, float],
                stored: Tuple[float, float, float],
                overlap: float = 1.0e-4,
                dt: float = 1.0e-5) -> Tuple[ContactForces, RollingMoments]:
    """Evaluate the forces and the rolling moment of one 3D contact between two 10 mm particles at rest."""
    contact = model.evaluate(np.array([normal], dtype=float), np.array([overlap]), np.zeros((1, 3)), np.zeros((1, 3)),
                             dt, np.array([0.005]), np.array([1.0]))
    moments = model.rolling(contact, np.array([spin], dtype=float), np.array([stored], dtype=float), dt,
                            np.array([0.005]), np.array([1.0e-6]))
    return contact, moments


class TestHertzMindlinModel3D:
    """Tests of the Hertz-Mindlin contact law with 3D vectors."""

    def test_force_and_normal_of_a_3d_contact(self) -> None:
        """The normal force must follow Hertz along an oblique normal, which the result carries along."""
        model = HertzMindlinModel(_material(restitution=1.0))
        normal = np.array([[2.0, -1.0, 2.0]]) / 3.0
        result = model.evaluate(normal, np.array([1.0e-4]), np.zeros((1, 3)), np.zeros((1, 3)), 1.0e-5,
                                np.array([0.005]), np.array([1.0]))
        expected = (4.0 / 3.0) * model.e_star * math.sqrt(0.005) * 1.0e-4**1.5
        assert_allclose(result.force, expected * normal, rtol=1e-12)
        assert_allclose(result.normal, normal)

    def test_coulomb_friction_limit_in_3d(self) -> None:
        """Fast oblique sliding must give a tangential force of exactly mu F_n against the sliding direction."""
        mu = 0.5
        model = HertzMindlinModel(_material(restitution=1.0, friction=mu))
        result = model.evaluate(np.array([[0.0, 1.0, 0.0]]), np.array([1.0e-4]), np.array([[3.0, 0.0, -4.0]]),
                                np.zeros((1, 3)), 1.0e-5, np.array([0.005]), np.array([1.0]))
        f_n = (4.0 / 3.0) * model.e_star * math.sqrt(0.005) * 1.0e-4**1.5
        assert_allclose(result.tangential_force, [[-0.6 * mu * f_n, 0.0, 0.8 * mu * f_n]], rtol=1e-9, atol=1e-18)


class TestRollingResistance3D:
    """Tests of the rolling resistance with vector spins and springs."""

    @pytest.mark.parametrize("spin, stored", [(0.0, 1.0e-4), (50.0, 0.0), (-50.0, 2.0e-5)])
    def test_reduces_to_planar_model(self, spin: float, stored: float) -> None:
        """A rotation about z at a contact whose normal lies in the x-y plane must give the moment of the 2D model."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        _, planar = _rolling(model, 1.0e-4, spin, stored)
        _, spatial = _rolling_3d(model, (0.0, 1.0, 0.0), (0.0, 0.0, spin), (0.0, 0.0, stored))
        assert_allclose(spatial.moment, [[0.0, 0.0, planar.moment[0]]], rtol=1e-12, atol=1e-30)
        assert_allclose(spatial.rolling_spring, [[0.0, 0.0, planar.rolling_spring[0]]], rtol=1e-12, atol=1e-30)

    def test_twisting_is_free(self) -> None:
        """A spin about the contact normal must neither load the rolling spring nor meet a moment."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        _, moments = _rolling_3d(model, (0.0, 1.0, 0.0), (0.0, 50.0, 0.0), (0.0, 0.0, 0.0))
        assert_allclose(moments.moment, 0.0, atol=1e-30)
        assert_allclose(moments.rolling_spring, 0.0, atol=1e-30)

    def test_spring_is_projected_onto_contact_plane(self) -> None:
        """A stored rotation must lose its component along the normal, as the tangential spring does."""
        model = HertzMindlinModel(_material(restitution=1.0, rolling_friction=0.3))
        contact, moments = _rolling_3d(model, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0e-4, 1.0e-4, 0.0), dt=0.0)
        assert_allclose(moments.rolling_spring, [[1.0e-4, 0.0, 0.0]], atol=1e-20)
        stiffness = 2.25 * 0.3**2 * 0.005**2 * contact.normal_stiffness[0]
        assert_allclose(moments.moment, [[-stiffness * 1.0e-4, 0.0, 0.0]], rtol=1e-12)

    def test_moment_is_capped_in_magnitude(self) -> None:
        """A fast oblique rolling spin must mobilise the full rolling moment mu_r R* F_n against it."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        contact, moments = _rolling_3d(model, (0.0, 1.0, 0.0), (30.0, 0.0, 40.0), (0.0, 0.0, 0.0))
        limit = 0.3 * 0.005 * contact.normal_force[0]
        assert_allclose(moments.moment, [[-0.6 * limit, 0.0, -0.8 * limit]], rtol=1e-9)
        stiffness = 2.25 * 0.3**2 * 0.005**2 * contact.normal_stiffness[0]
        assert_allclose(moments.rolling_spring, -moments.moment / stiffness, rtol=1e-12)

    def test_open_contact_has_no_moment_and_no_history(self) -> None:
        """A contact that opened must exert no moment and reset its rolling spring."""
        model = HertzMindlinModel(_material(restitution=0.4, rolling_friction=0.3))
        _, moments = _rolling_3d(model, (0.0, 1.0, 0.0), (5.0, 0.0, 5.0), (1.0e-3, 0.0, 0.0), overlap=-1.0e-5)
        assert_allclose(moments.moment, 0.0, atol=1e-30)
        assert_allclose(moments.rolling_spring, 0.0, atol=1e-30)


class TestOverlapContacts:
    """Tests of the contact law of polygonal and polyhedral particles."""

    def test_contact_radius_keeps_the_ratio_of_spheres(self) -> None:
        """The contact radius from the rim of the overlap of two spheres must be the Hertz contact radius."""
        radius, overlap = 0.005, 1.0e-5
        rim_sq = 2.0 * radius * overlap
        assert_allclose(overlap_contact_radius(np.array([math.pi * rim_sq]), 3), math.sqrt(radius * overlap))
        assert_allclose(overlap_contact_radius(np.array([2.0 * math.sqrt(rim_sq)]), 2), math.sqrt(radius * overlap))

    @pytest.mark.parametrize("dimension", [2, 3])
    def test_overlap_of_two_spheres_gives_the_hertz_force(self, dimension: int) -> None:
        """Fed with the overlap of two spheres, the law must give the Hertz force of their overlap."""
        model = HertzMindlinModel(_material())
        radius, overlap = 0.005, 1.0e-5
        rim = math.sqrt(2.0 * radius * overlap)
        if dimension == 3:
            # a lens of volume pi R* delta^2 with a rim of area pi a^2
            size, measure = math.pi * rim * rim, math.pi * radius * overlap**2
        else:
            # a lens of area 4/3 a delta across a chord of 2 a
            size, measure = 2.0 * rim, 4.0 / 3.0 * rim * overlap
        normal = np.zeros((1, dimension))
        normal[0, 1] = 1.0
        forces = model.evaluate_overlap(normal, np.array([measure / size]),
                                        overlap_contact_radius(np.array([size]), dimension), np.zeros((1, dimension)),
                                        np.zeros((1, dimension)), 1.0e-5, np.array([1.0]))
        hertz = model.evaluate(normal, np.array([overlap]), np.zeros((1, dimension)), np.zeros((1, dimension)), 1.0e-5,
                               np.array([radius]), np.array([1.0]))
        assert_allclose(forces.normal_force, hertz.normal_force, rtol=1e-12)
        assert_allclose(forces.normal_stiffness, hertz.normal_stiffness, rtol=1e-12)

    def test_flat_punch_is_close_to_boussinesq(self) -> None:
        """A flat face pressed in by delta must carry within 6 % of the force 2 E* a delta of a flat punch."""
        model = HertzMindlinModel(_material())
        area, depth = 1.0e-4, 1.0e-6
        forces = model.evaluate_overlap(np.array([[0.0, 1.0, 0.0]]), np.array([depth]),
                                        overlap_contact_radius(np.array([area]), 3), np.zeros((1, 3)), np.zeros((1, 3)),
                                        1.0e-5, np.array([1.0]))
        punch = 2.0 * model.e_star * math.sqrt(area / math.pi) * depth
        assert_allclose(forces.normal_force, punch, rtol=0.06)

    def test_shares_of_a_split_contact_add_up(self) -> None:
        """A contact split in two parts must carry the force, stiffness and damping of the whole."""
        model = HertzMindlinModel(_material(restitution=0.3))
        radius = np.array([0.004, 0.004, 0.004])
        depth = np.array([2.0e-6, 2.0e-6, 2.0e-6])
        share = np.array([1.0, 0.3, 0.7])
        normal = np.tile([0.0, 1.0, 0.0], (3, 1))
        velocity = np.tile([0.01, -0.02, 0.0], (3, 1))
        forces = model.evaluate_overlap(normal, depth, radius, velocity, np.zeros((3, 3)), 1.0e-5, np.ones(3), share)
        assert_allclose(forces.force[1] + forces.force[2], forces.force[0], rtol=1e-12)
        assert_allclose(forces.normal_stiffness[1] + forces.normal_stiffness[2], forces.normal_stiffness[0])

    def test_open_overlap_has_no_force(self) -> None:
        """A contact without depth or contact radius must carry nothing and lose its spring."""
        model = HertzMindlinModel(_material())
        forces = model.evaluate_overlap(np.tile([0.0, 1.0], (2, 1)), np.array([0.0, 1.0e-6]), np.array([0.01, 0.0]),
                                        np.zeros((2, 2)), np.ones((2, 2)), 1.0e-5, np.ones(2))
        assert_allclose(forces.force, 0.0)
        assert_allclose(forces.tangential_spring, 0.0)


class TestTwisting:
    """Tests of the twisting resistance of flat contacts."""

    def _contact(self, model: HertzMindlinModel) -> ContactForces:
        """Return a face contact of radius 4 mm pressed in by 2 microns, at rest."""
        return model.evaluate_overlap(np.array([[0.0, 1.0, 0.0]]), np.array([2.0e-6]), np.array([0.004]),
                                      np.zeros((1, 3)), np.zeros((1, 3)), 1.0e-5, np.array([1.0]))

    def test_small_twist_is_elastic(self) -> None:
        """A small stored twist must give the moment of the tangential stiffness spread over the contact disc."""
        model = HertzMindlinModel(_material(friction=0.5))
        contact = self._contact(model)
        twist = 1.0e-6
        moments = model.twisting(contact, np.zeros((1, 3)), np.array([[0.0, twist, 0.0]]), 0.0, np.array([0.004]),
                                 np.array([1.0e-6]))
        tangential = 4.0 * model.g_star / model.e_star * contact.normal_stiffness[0]
        assert_allclose(moments.moment, [[0.0, -0.5 * tangential * 0.004**2 * twist, 0.0]])
        assert_allclose(moments.rolling_spring, [[0.0, twist, 0.0]])

    def test_large_twist_slips_at_the_friction_moment(self) -> None:
        """A large twist must be capped at 2/3 mu F_n a, and the spring reset to that moment."""
        model = HertzMindlinModel(_material(friction=0.5))
        contact = self._contact(model)
        moments = model.twisting(contact, np.zeros((1, 3)), np.array([[0.0, 1.0, 0.0]]), 0.0, np.array([0.004]),
                                 np.array([1.0e-6]))
        limit = 2.0 / 3.0 * 0.5 * contact.normal_force[0] * 0.004
        assert_allclose(moments.moment, [[0.0, -limit, 0.0]])
        tangential = 4.0 * model.g_star / model.e_star * contact.normal_stiffness[0]
        assert_allclose(moments.rolling_spring, [[0.0, limit / (0.5 * tangential * 0.004**2), 0.0]])

    def test_rolling_spin_does_not_twist(self) -> None:
        """A spin about an axis in the contact plane must not twist the contact."""
        model = HertzMindlinModel(_material())
        moments = model.twisting(self._contact(model), np.array([[3.0, 0.0, -2.0]]), np.zeros((1, 3)), 1.0e-5,
                                 np.array([0.004]), np.array([1.0e-6]))
        assert_allclose(moments.moment, 0.0)
