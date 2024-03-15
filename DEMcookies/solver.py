import numpy as np


class Verlet:
    """
    Verlet solver
    """

    def __init__(self, particles, dt):
        self.particles = particles
        self.dt = dt

    def update(self):
        """
        Updates the position of the particles
        """
        for particle in self.particles:
            particle.x += particle.vx * self.dt + 0.5 * particle.ax * self.dt ** 2
            particle.y += particle.vy * self.dt + 0.5 * particle.ay * self.dt ** 2

    # def update_acceleration(self):
    #     """
    #     Updates the acceleration of the particles
    #     """
    #     for particle in self.particles:
    #         particle.ax = 0.0
    #         particle.ay = self.gravity

    def update_velocity(self):
        """
        Updates the velocity of the particles
        """
        for particle in self.particles:
            particle.vx += 0.5 * particle.ax * self.dt

