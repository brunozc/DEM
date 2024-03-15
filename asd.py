import numpy as np
import matplotlib.pyplot as plt

# Set the dimensions of the 2D space
x_max = 2
y_max = 2

# Set the number of particles
num_particles = 50

# Set the range for the distance between particles
min_distance = 0.5
max_distance = 1.0

# Generate random x and y coordinates for the particles
x_coordinates = np.random.uniform(0, x_max, num_particles)
y_coordinates = np.random.uniform(0, y_max, num_particles)

# Generate random angles for the particles
angles = np.random.uniform(0, 2 * np.pi, num_particles)

# Generate random distances between particles
distances = np.random.uniform(min_distance, max_distance, num_particles)

# Calculate the new x and y coordinates based on the distances and angles
new_x_coordinates = x_coordinates + distances * np.cos(angles)
new_y_coordinates = y_coordinates + distances * np.sin(angles)

# Plot the original and new particle positions
# plt.scatter(x_coordinates, y_coordinates, label="Original Positions")
plt.scatter(new_x_coordinates, new_y_coordinates, label="New Positions")
plt.title("Particle Distribution in 2D Space")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
