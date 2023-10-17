## Model of Ocular Ambient Irradiance

This model allows for the calculation of ocular irradiance (irradiance received by the ocular area) based on local ambient irradiance (DNI, DHI and albedo), the sun's position (zenith and azimuth angles), and head orientation (pitch and yaw).

## Install requirements

To use this code, it is necessary to install the two libraries it relies on: numpy and scipy. If you do not have these libraries, you can install them by running the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can install them one at a time using the following commands:

```bash
pip install numpy
pip install scipy
```

## Quick Start

To use the model, you need to import the corresponding class and provide the necessary inputs.

Let's assume you have a DNI = 1.4 W/m2, a Solar Zenith Angle (SZA) of 45°, a Solar Azimuth Angle (SAA) of 180°, and you want to calculate the direct ocular irradiance with the head oriented vertically toward East.

```python
import numpy as np
import OcularAmbientIrradiance as OAI

DNI = 1.4 # W / m2

theta = np.array([np.pi/4])
phi = np.array([np.pi])
beta = np.array([np.pi/2])
alpha = np.array([np.pi/2])

ocuIrr = OAI.OcularAmbientIrradiance()
intens = DNI * ocuIrr.F_dir(theta=theta, 
                            phi=phi,
                            beta=beta,
                            alpha=alpha)

print(intens) # result is: 0.25141949 W/m2
```

## Extended example
