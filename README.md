## Model of Ocular Ambient Irradiance

Questo coice è una prima semplice implementazione del modello descritt in Marro et al. 2024 "A model of ocular ambient irradiance at any head orientation".

This model allows for the calculation of ocular irradiance (irradiance received by the ocular area) based on local ambient irradiance (DNI, DHI and albedo), the sun's position (zenith and azimuth angles), and  orientation of the head (pitch and yaw).

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

## Example with rotating head

Cnsideriamo che la testa ruoti di 180 gradi (da 90 a 270) e che sia orientata verso il basso di 15 gradi (tipico orientamento di persona che cammina). Assumiamo che la rotazione sia abbastanza veloce da non apprezzare un cambiamento nella direzione del sole.

```python
import numpy as np
import OcularAmbientIrradiance as OAI

ocuIrr = OAI.OcularAmbientIrradiance()

theta = np.ones(100) * np.pi/4
phi = np.ones(100) * np.pi
beta = np.ones(100) * np.radians(105)
alpha = np.array([np.pi/2 + i*np.pi/100 for i in range(100)])

DNI = 1.4 # W / m2

I_dir = DNI * ocuIrr.F_dir(theta, phi, beta, alpha)

print(I_dir)
```

## Cite this work

```@article{Marro_2024,
title = {A model of ocular ambient irradiance at any head orientation},
journal = {Computers in Biology and Medicine},
volume = {179},
pages = {108903},
year = {2024},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2024.108903},
url = {https://www.sciencedirect.com/science/article/pii/S0010482524009880},
author = {Michele Marro and Laurent Moccozet and David Vernez},
}
```
