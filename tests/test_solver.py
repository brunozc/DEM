import math

import numpy as np

from DEMcookies.contact import HertzMindlinModel, MaterialProperties
from DEMcookies.solver import VelocityVerlet


def _material(restitution=1.0, friction=0.5):
    return MaterialProperties(
        density=2500.0,
        youngs_modulus=5.0e6,
        poisson_ratio=0.25,
        restitution=restitution,
        friction=friction,
    )


def test_velocity_verlet_free_fall():
    """A single step of free fall must match the analytic result exactly."""
    dt = 1.0e-3
    g = 9.81
    integrator = VelocityVerlet(dt)
    pos = np.zeros((1, 2))
    vel = np.zeros((1, 2))

    def acceleration(p, v):
        a = np.zeros_like(p)
        a[:, 1] = -g
        return a

    pos, vel, _ = integrator.step(pos, vel, acceleration)

    np.testing.assert_allclose(pos[0, 1], -0.5 * g * dt ** 2)
    np.testing.assert_allclose(vel[0, 1], -g * dt)


def test_hertz_normal_force_magnitude():
    """The elastic normal force must follow F = 4/3 E* sqrt(R*) delta^1.5."""
    model = HertzMindlinModel(_material(restitution=1.0))  # no damping
    radius = 0.01
    delta = 1.0e-4
    r_eff = radius * radius / (radius + radius)
    m_eff = 1.0

    # normal along +x, no relative velocity -> purely elastic, no shear
    fx, fy, ftx, fty, _, _ = model.contact_force(
        1.0, 0.0, delta, 0.0, 0.0, 0.0, 0.0, 1.0e-5, r_eff, m_eff)

    expected = (4.0 / 3.0) * model.e_star * math.sqrt(r_eff) * delta ** 1.5
    np.testing.assert_allclose(fx, expected, rtol=1e-12)
    assert fy == 0.0 and ftx == 0.0 and fty == 0.0


def test_coulomb_friction_limit():
    """Tangential force may never exceed mu * F_n."""
    mu = 0.5
    model = HertzMindlinModel(_material(restitution=1.0, friction=mu))
    radius = 0.01
    delta = 1.0e-4
    r_eff = radius * radius / (radius + radius)

    # strong tangential sliding, no normal approach velocity
    fx, fy, ftx, fty, _, _ = model.contact_force(
        1.0, 0.0, delta, 0.0, 5.0, 0.0, 0.0, 1.0e-5, r_eff, 1.0)

    f_n = (4.0 / 3.0) * model.e_star * math.sqrt(r_eff) * delta ** 1.5
    ft = math.hypot(ftx, fty)
    assert ft <= mu * f_n * (1.0 + 1e-9)
    np.testing.assert_allclose(ft, mu * f_n, rtol=1e-9)


def test_effective_moduli():
    """Effective moduli must match the closed-form expressions."""
    mat = _material()
    model = HertzMindlinModel(mat)
    e, nu = mat.youngs_modulus, mat.poisson_ratio
    g = mat.shear_modulus
    np.testing.assert_allclose(model.e_star, e / (2 * (1 - nu ** 2)))
    np.testing.assert_allclose(model.g_star, g / (2 * (2 - nu)))
    assert model.e_star > 0 and model.g_star > 0

