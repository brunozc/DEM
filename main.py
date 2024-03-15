from DEMcookies.particle_generation import InitialiseParticles


particles = InitialiseParticles((2, 2), radius_range=(0.05, 0.1))
particles.initialize_particles()

particles.plot_particles("res", "asd")


# # Simulate for a certain time (e.g., 1 second) with a small time step
# dt = 0.1
# num_steps = int(1.0 / dt)
# for t in range(num_steps):
#     particles.update_particles(dt)
#     particles.plot_particles("res", t)
