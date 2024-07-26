import scipy.special as sc
import numpy as np

# This is a first version of the ocular ambient irradiance model 
# aimed at making the model described in Marro et al. 2024 usable.
# Author: Michele Marro

class OcularAmbientIrradiance:
    """
    Implementation of the Model of ocular ambient irradiance described in:
    
    Michele Marro, Laurent Moccozet, David Vernez,
    A model of ocular ambient irradiance at any head orientation,
    Computers in Biology and Medicine,
    Volume 179, 2024, 108903, ISSN 0010-4825,
    https://doi.org/10.1016/j.compbiomed.2024.108903.
    (https://www.sciencedirect.com/science/article/pii/S0010482524009880).

    This model enables the rapid calculation of ocular irradiance 
    based on ambient irradiance data, accounting for the complex interaction 
    with head anatomy and varying environmental parameters and 
    head orientation during the exposure period.
    """
    def __init__(self, coefficients_filename="coefficients.txt"):

        # load coefficients to compute Fdir
        self.coeffs = np.loadtxt(coefficients_filename)

        # total number of coefficients (100)
        self.tot_coeffs = np.shape(self.coeffs)[0]
        self.l_max = int(np.sqrt(self.tot_coeffs - 1))

        # order and degree
        self.m_and_l = self.indices(self.l_max)

        # regression coefficients to compute F_dif and F_ref 
        # f(x) = +- a * (x  - pi/2) + b/(1 + c*exp(+- d * (x - pi/2))) + f

        self.a = 0.384 
        self.b = 2.713
        self.c = 1.201
        self.d = -0.899

    def get_zen_p_azi_p(self, theta, phi, beta, alpha):
        """
        Compute solar zenith and azimuth angles in 
        the reference system of the head O' (Equation 6 of Marro et al. 2024).

        Parameters:
        -----------
        theta : float
            solar zenith angle (in radians)
        phi : float
            solar azimuth angle (in radians)
        beta : float
            zenith angle of the head (in radians)
        alpha: float
            azimuth angle of the head (in radians)

        Return:
        -------
        zen_p : float
            solar zenith angle in O'
        azi_p : float
            solar azimuth angle in O'
        """
        zen_p = np.arccos(np.sin(theta) * np.sin(beta) * np.cos(phi - alpha) +
                          np.cos(theta) * np.cos(beta))
        azi_p = np.arctan2(np.sin(theta) * np.sin(phi - alpha),
                           np.sin(theta) * np.cos(beta) * np.cos(phi - alpha) -
                           np.cos(theta) * np.sin(beta))
        return zen_p, azi_p

    def shiftedP(self, m, l, z):
        """
        Shifted Associated 
        Legendre polynomials (ALP)

        Parameters:
        -----------
        m : int
            order of the ALP
        l : int
            degree of the ALP
        z : float
            z = cos(theta) where theta
            is the zenith angle (0 <= z <= 1)
        """
        return sc.lpmv(m, l, 2 * z - 1)

    def K(self, m, l):
        """
        Compute the normalization
        factor for the shifted ALP
        of order m and degree l.

        Parameters:
        -----------
        m : int
            order of the ALP
        l : int
            degree of the ALP
        -----------
        """
        return np.sqrt(((2 * l + 1) * sc.factorial(l - np.abs(m))) / (2 * np.pi * sc.factorial(l + np.abs(m))))

    def H(self, m, l, th, ph):
        """
        Hemispherical basis function of degree l and order m
        (Equation 8 of Gautron et al. 2004).

        Parameters:
        -----------
        m : int
            order of the ALP
        l : int
            degree of the ALP
        th : float
            th = cos(theta) cosine of the zenith angle in radians
            (0 <= th <= 1)
        ph : float
            azimuth angle
        """
        if m == 0:
            mid_term = self.K(m, l)
        elif m > 0:
            mid_term = self.K(m, l) * np.sqrt(2) * np.cos(m * ph)
        elif m < 0:
            mid_term = self.K(m, l) * np.sqrt(2) * np.sin(-m * ph)
            m = -m
        return mid_term * self.shiftedP(m, l, th)

    def indices(self, L_max):
        """
        Compute the sequence of indices
        given the maximun degree L_max.

        Parameters:
        -----------
        L_max : int
            maximum degree to compute 
            the sequence of l and m.
        """
        return np.array([[m, l] for l in range(L_max + 1) for m in range(-l, l + 1)])


    def F_dir(self, theta, phi, beta, alpha):
        """
        Compute the fraction of direct irradiance
        of degree L=9 (Equation 5 of Marro et al. 2024).

        For the input refers to Fig 3 of Marro  et al. 2024.

        Parameters:
        -----------
        theta : float
            solar zenith angle (in radians)
        phi : float
            solar azimuth angle (in radians)
        beta : float
            zenith angle of the head (in radians)
        alpha : float
            azimuth angle of the head (in radians)
        """
        theta_p, phi_p = self.get_zen_p_azi_p(theta, phi, beta, alpha)
        intens = np.zeros((np.shape(theta)[0], self.tot_coeffs))
        for k, ind in enumerate(self.m_and_l):
            intens[:, k] = self.coeffs[k] * self.H(ind[0], ind[1], np.cos(theta_p), phi_p)
        intens = np.sum(intens, axis=1)
        return intens

    def F_dif(self, beta):
        """
        Compute the fraction of diffuse irradiance
        (Equation 8 of Marro et al. 2024).

        Parameters:
        -----------
        beta : float
            zenith angle of the head (in radians)
        """
        return self.a * (beta - np.pi/2) + self.b / (1 + np.exp(self.c * (beta - np.pi / 2))) + self.d

    def F_ref(self, beta):
        """
        Compute the fraction of ground-reflected irradiance
        (Equation 9 of Marro et al. 2024).

        Parameters:
        -----------
        beta : float
            zenith angle of the head (in radians)
        """
        return - self.a * (beta - np.pi/2) + self.b / (1 + np.exp(- self.c * (beta - np.pi / 2))) + self.d