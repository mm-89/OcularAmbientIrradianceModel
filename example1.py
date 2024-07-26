import numpy as np
import OcularAmbientIrradiance as OAI

#----------------------------------------
# Example 1:

# simulate n sun positions for F_dir
# for a constant orientation of the head
#----------------------------------------

# define an instance of 
# the OcularAmbientIrradiance class

ocuIrr = OAI.OcularAmbientIrradiance()

n = 1000
theta = np.arccos( np.random.rand(n) )
phi = 2 * np.pi * np.random.rand(n)

# taking a random irradiance

DNI = 1.2

I_dir = DNI * ocuIrr.F_dir(theta, phi, 0, 0)
print(I_dir)