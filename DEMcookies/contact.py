import numpy as np


radius_1 = 1
radius_2 = 2
shear_modulus = 100
poisson_ratio = 0.2


radius_bar = 2 * radius_1 * radius_2 / (radius_1 + radius_2)
M = 2 * np.sqrt(2 * radius_bar) * shear_modulus / (3 * (1 - poisson_ratio))
n = radius_1 + radius_2 - delattr
contact_force = M * (n)  ** 1.5


shear_force = shear_stiffness * shear_displacement
