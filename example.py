import numpy as np
import OcularAmbientIrradiance as OAI

# define an instance of 
# the OcularAmbientIrradiance class

ocuIrr = OAI.OcularAmbientIrradiance()

# generate n=1000 random number
# to simulate n sun positions

n = 1000
theta = np.arccos( np.random.rand(n) )
phi = 2 * np.pi * np.random.rand(n)

# taking a random irradiance

DNI = 1.2

intens = DNI * ocuIrr.F_dir(theta, phi, 0, 0)
print(intens)