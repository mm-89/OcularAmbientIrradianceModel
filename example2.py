import numpy as np
import OcularAmbientIrradiance as OAI

#----------------------------------------
# Example 2:

# Implementation of the whole model
#----------------------------------------

# define an instance of 
# the OcularAmbientIrradiance class

ocuIrr = OAI.OcularAmbientIrradiance()

theta = np.array([np.pi/4])
phi = np.array([np.pi])
beta = np.array([np.pi/2])
alpha = np.array([np.pi/2])

# taking a random irradiances

DNI = 1.2   # direct irradiance
DHI = 0.8   # diffuse irradiance
RHO = 0.05  # albedo

I_dir = DNI * ocuIrr.F_dir(theta, phi, beta, alpha)
I_dif = DHI * ocuIrr.F_dif(beta)
I_ref = RHO * DHI * ocuIrr.F_ref(beta)

I_glo = I_dir + I_dif + I_ref

print(I_glo)