import os
import numpy as np
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0.0
        self.vy = 0.0

        self.ax = 0.0
        self.ay = 0.0


class InitialiseParticles:
    def __init__(self, box_size, radius_range=(0.05, 0.1), gravity=-9.81, seed=14):
        self.box_size = box_size
        self.radius_range = radius_range
        self.gravity = gravity
        np.random.seed(seed)
        self.max_nb_iter = 50000
        self.num_particles = []
        self.particles = []


    def initialize_particles(self):

        nb_generated_particles = 0
        nb_iterations = 0

        # generate the particles position randomly
        while nb_iterations <= self.max_nb_iter:
            x = np.random.uniform(0, self.box_size[0])
            y = np.random.uniform(0, self.box_size[1])
            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])

            # Ensure the particle is within the box considering its radius
            x = min(max(x, radius), self.box_size[0] - radius)
            y = min(max(y, radius), self.box_size[1] - radius)

            # Check for overlap with existing particles
            overlap = False
            for particle in self.particles:
                distance = np.sqrt((x - particle.x) ** 2 + (y - particle.y) ** 2)
                if distance < radius + particle.radius:
                    overlap = True
                    break

            if overlap is False:
                new_particle = Particle(x, y, radius)
                self.particles.append(new_particle)
                nb_generated_particles += 1
            nb_iterations += 1
        self.num_particles = nb_generated_particles
        print(f"Number of particles generated: {self.num_particles}")

        # initialise gravity
        self.initialise_gravity()


    def initialise_gravity(self):
        for particle in self.particles:
            particle.ay = self.gravity



    def update_particles(self, dt):
        for particle in self.particles:
            particle.vx += particle.ax * dt
            particle.vy += particle.ay * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt


    def plot_particles(self, output_folder, time_step):

        if os.path.isdir(output_folder) is False:
            os.mkdir(output_folder)

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.box_size[0])
        ax.set_ylim(0, self.box_size[1])
        for particle in self.particles:
            circle = plt.Circle((particle.x, particle.y), particle.radius, color="b")
            ax.add_artist(circle)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(os.path.join(output_folder, f"{time_step}.png"))
        plt.close()
