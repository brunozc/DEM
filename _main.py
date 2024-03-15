import numpy as np
import matplotlib.pyplot as plt
import imageio


class Particle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0


def initialize_particles(num_particles, box_size):
    particles = []
    radius_range = (0.05, 0.1)

    # Calculate the number of particles along each dimension of the grid
    num_particles_x = int(np.sqrt(num_particles))
    num_particles_y = int(num_particles / num_particles_x)

    # Calculate the spacing between particles
    dx = box_size / num_particles_x
    dy = box_size / num_particles_y

    for i in range(num_particles_x):
        for j in range(num_particles_y):
            x = i * dx + radius_range[1]  # Add radius to center particles in the grid cells
            y = j * dy + radius_range[1]
            radius = np.random.uniform(*radius_range)

            # Ensure the particle is within the box considering its radius
            x = min(max(x, radius), box_size - radius)
            y = min(max(y, radius), box_size - radius)

            # Check for overlap with existing particles
            overlap = False
            for particle in particles:
                distance = np.sqrt((x - particle.x)**2 + (y - particle.y)**2)
                if distance < radius + particle.radius:
                    overlap = True
                    break

            # If no overlap, add the particle
            if not overlap:
                particles.append(Particle(x, y, radius))

    return particles



def calculate_gravity(particles, g):
    for particle in particles:
        particle.ay -= g


def apply_boundary_conditions(particles, box_size):
    for particle in particles:
        if particle.x - particle.radius < 0:
            particle.x = particle.radius
            particle.vx = 0
        if particle.x + particle.radius > box_size:
            particle.x = box_size - particle.radius
            particle.vx = 0
        if particle.y - particle.radius < 0:
            particle.y = particle.radius
            particle.vy = 0
        if particle.y + particle.radius > box_size:
            particle.y = box_size - particle.radius
            particle.vy = 0


def hertz_mindlin_contact(p1, p2):
    overlap = p1.radius + p2.radius - np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    if overlap > 0:
        normal_force = k * overlap**1.5
        tangent_force = mu * normal_force
        return normal_force, tangent_force
    else:
        return 0, 0


def mohr_coulomb_contact(p1, p2):
    overlap = p1.radius + p2.radius - np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    if overlap > 0:
        normal_force = k * overlap
        tangent_force = mu * normal_force
        return normal_force, tangent_force
    else:
        return 0, 0


def calculate_contact_forces(particles):
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            normal_force, tangent_force = mohr_coulomb_contact(
                particles[i], particles[j]
            )
            # Apply forces to particles
            particles[i].ax += normal_force / particles[i].radius
            particles[i].ay += tangent_force / particles[i].radius
            particles[j].ax -= normal_force / particles[j].radius
            particles[j].ay -= tangent_force / particles[j].radius


def verlet_integration(particles, dt):
    for particle in particles:
        particle.x += particle.vx * dt + 0.5 * particle.ax * dt**2
        particle.y += particle.vy * dt + 0.5 * particle.ay * dt**2
        particle.vx += particle.ax * dt
        particle.vy += particle.ay * dt

        # Adjust velocity at the bottom boundary
        if particle.y - particle.radius < 0:
            particle.y = particle.radius
            particle.vy = 0


# Simulation parameters
num_particles = 10
box_size = 1.0
g = 9.81
dt = 0.01
k = 1e4  # Hertz-Mindlin stiffness
mu = 0.5  # Hertz-Mindlin coefficient of friction
nb_steps = 100

# Initialize particles
particles = initialize_particles(num_particles, box_size)

# Main simulation loop
for step in range(nb_steps):
    calculate_gravity(particles, g)
    calculate_contact_forces(particles)
    apply_boundary_conditions(particles, box_size)
    verlet_integration(particles, dt)

    # Plotting the particles
    fig, ax = plt.subplots()
    for particle in particles:
        circle = plt.Circle((particle.x, particle.y), particle.radius, color="b")
        ax.add_patch(circle)

    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(f"results/{step:05d}.png")
    plt.close()

# Create a video using imageio
images = []
for step in range(nb_steps):
    filename = f"results/{step:05d}.png"
    images.append(imageio.v3.imread(filename))

imageio.mimsave("output.gif", images, fps=5)