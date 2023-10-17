import scipy.special as sc
import numpy as np

class OcularAmbientIrradiance:
    """
    Model of ocular ambient irradiance described in X.

    Description
    """
    def __init__(self, coefficients_filename="coefficients.txt"):

        # load coefficients to compute Fdir
        self.coeffs = np.loadtxt(coefficients_filename)

        # total number of coefficients (100)
        self.l_max = int(np.sqrt(np.shape(self.coeffs)[0]) - 1)

        # regression coefficients to compute F_dif and F_ref
        # given from the equation
        # 
        # f(x) = +- a * (x  - pi/2) + b/(1 + c*exp(+- d * (x - pi/2))) + f

        self.a = 0.4099
        self.b = 2.859
        self.c = 0.9918
        self.d = 1.175
        self.f = -0.9762

    def get_zen_p_azi_p(self, zenith, azimuth, alpha, beta):
        """
        Description
        """
        zen_p = np.arccos(np.sin(zenith) * np.sin(beta) * np.cos(azimuth - alpha) +
                          np.cos(zenith) * np.cos(beta))
        azi_p = np.arctan2(np.sin(zenith) * np.sin(azimuth - alpha),
                           np.sin(zenith) * np.cos(beta) * np.cos(azimuth - alpha) -
                           np.cos(zenith) * np.sin(beta))
        return zen_p, azi_p

    def shiftedP(self, m, l, z):
        """
        Description (reference)
        """
        return sc.lpmv(m, l, 2 * z - 1)

    def K(self, m, l):
        """
        Normalizazion factor

        Description (reference)
        """
        return np.sqrt(((2 * l + 1) * sc.factorial(l - np.abs(m))) / (2 * np.pi * sc.factorial(l + np.abs(m))))

    def H(self, m, l, th, ph):
        """
        Hemispherical Harmonics

        Description (reference)
        """
        if m == 0:
            mid_term = self.K(m, l)
        elif m > 0:
            mid_term = self.K(m, l) * np.sqrt(2) * np.cos(m * ph)
        elif m < 0:
            mid_term = self.K(m, l) * np.sqrt(2) * np.sin(-m * ph)
            m = -m
        return mid_term * self.shiftedP(m, l, th)

    def indices(self, index_n):
        index = []
        for n in range(index_n + 1):
            for l in range(-n, n + 1):
                index.append([l, n])
        return index

    def F_dir(self, theta, phi, beta, alpha):
        """
        Description 
        """
        theta_p, phi_p = self.get_zen_p_azi_p(theta, phi, alpha, beta)
        for i in range(self.l_max):
            ind_max = len(self.indices(self.l_max))
            intens = np.zeros((np.shape(theta)[0], ind_max))
            for k, ind in enumerate(self.indices(self.l_max)):
                intens[:, k] = self.coeffs[k] * self.H(ind[0], ind[1], np.cos(theta_p), phi_p)
            intens = np.sum(intens, axis=1)
        return intens

    def F_dif(self, beta):
        """
        Description
        """
        return self.a * (beta - np.pi/2) + self.b / (1 + self.c * np.exp(self.d * (beta - np.pi / 2))) + self.f

    def F_ref(self, beta):
        "Description"
        return - self.a * (beta - np.pi/2) + self.b / (1 + self.c * np.exp(- self.d * (beta - np.pi / 2))) + self.f
    
    def I_dir(self, DNI):
        """
        Description
        """
        pass

    def I_dif(self, DHI):
        """
        Description (reference)
        """
        pass

    def I_ref(self, DHI, rho):
        """
        Description (reference)
        """
        pass