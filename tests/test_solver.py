import numpy as np
from DEMcookies.solver import Verlet
from DEMcookies.particle_generation import Particle


def test_solver():
    # create test to test solver

        particles = [Particle(0, 0, 1)]
        verlet = Verlet(particles, 0.1)

        # add gravity acceleration to the particle
        particles[0].ay = -9.81

        # Update the system
        # verlet.update_acceleration()
        verlet.update()
        verlet.update_velocity()

        # Check that the particle's position has been updated correctly
        np.testing.assert_almost_equal(particles[0].x, 0.01)
        np.testing.assert_almost_equal(particles[0].y, 0.0099)

        # Check that the particle's velocity has been updated correctly
        np.testing.assert_almost_equal(particles[0].vx, 1)
        np.testing.assert_almost_equal(particles[0].vy, 0.99)
