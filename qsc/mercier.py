"""
This module contains the routine for computing the terms in
Mercier's criterion.
"""

import numpy as np
from .util import mu0

def mercier(self):
    """
    Compute the terms in Mercier's criterion.
    """

    # See Overleaf note "Mercier criterion near the magnetic axis- detailed notes".
    # See also "20200604-02 Checking sign in Mercier DGeod near axis.docx"

    # Shorthand:
    d_l_d_phi = self.d_l_d_phi
    B0 = self.B0
    Bbar = self.Bbar
    G0 = self.G0
    p2 = self.p2
    # etabar = self.etabar
    # curvature = self.curvature
    # sigma = self.sigma
    # iotaN = self.iotaN
    # iota = self.iota
    pi = np.pi
    d_phi = self.d_phi
    nfp = self.nfp
    B1s = self.B1s
    B1c = self.B1c
    B20 = self.B20
    X1s = self.X1s
    X1c = self.X1c
    Y1s = self.Y1s
    Y1c = self.Y1c
    beta_1s = self.beta_1s
    beta_1c = self.beta_1c

    #integrand = d_l_d_phi * (Y1c * Y1c + X1c * (X1c + Y1s)) / (Y1c * Y1c + (X1c + Y1s) * (X1c + Y1s))
    # integrand = d_l_d_phi * (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*sigma*sigma + etabar*etabar*curvature*curvature) \
    #     / (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*(1+sigma*sigma) + 2*etabar*etabar*curvature*curvature)

    # integral = np.sum(integrand) * self.d_phi * self.nfp * 2 * pi / self.axis_length

    #DGeod_times_r2 = -(2 * sG * spsi * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar &
    # self.DGeod_times_r2 = -(2 * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar \
    #                    / (pi * pi * pi * Bbar * Bbar * Bbar * Bbar * Bbar * Bbar * Bbar * Bbar * Bbar * Bbar * iotaN * iotaN)) \
    #                    * integral
    
    jacobian = self.sG * d_l_d_phi * B0 / G0

    V1 = X1s * X1s + X1c * X1c + Y1s * Y1s + Y1c * Y1c
    a = (X1s * X1s - X1c * X1c + Y1s * Y1s - Y1c * Y1c) / V1
    b = -2 * (X1s * X1c + Y1s * Y1c) / V1
    integrad_I_beta = (1/(B0 * B0 * V1)) * ((a * a + b * b)*(beta_1s * beta_1s + beta_1c * beta_1c) + \
                                             (np.sqrt(1 - a * a - b * b) - 1) * (a * (beta_1s * beta_1s - beta_1c * beta_1c) - 2 * b * beta_1s * beta_1c)) \
                                            / (np.sqrt(1 - a * a - b * b) * (a * a + b * b))
    I_beta  = np.sum(integrad_I_beta * jacobian) * d_phi * nfp
    self.DGeod_times_r2 = - abs(G0) * self.axis_length * I_beta / (16 * pi * pi * pi * pi * abs(Bbar))

    # self.d2_volume_d_psi2 = 4*pi*pi*abs(G0)/(Bbar*Bbar*Bbar)*(3*etabar*etabar - 4*self.B20_mean/Bbar + 2 * (self.G2 + iota * self.I2)/G0)
    integrand1 = 1 / (B0 * B0)
    integral1  = np.sum(integrand1 * jacobian) * d_phi * nfp
    integrand2 = (1 / (B0 * B0 * B0 * B0)) * (3 * (B1s * B1s + B1c * B1c) - 4 * B0 * B20 - mu0 * p2 * B0 * B0 / np.pi * integral1)
    integral2  = np.sum(integrand2 * jacobian) * d_phi * nfp
    self.d2_volume_d_psi2 = 2 * pi * abs(G0 / Bbar) * integral2

    # self.DWell_times_r2   = (mu0 * p2 * abs(G0) / (8 * pi * pi * pi * pi * Bbar * Bbar * Bbar)) * \
    #     (self.d2_volume_d_psi2 - 8 * pi * pi * mu0 * p2 * abs(G0) / (Bbar * Bbar * Bbar * Bbar * Bbar))
    self.DWell_times_r2 = mu0 * p2 * self.axis_length / (16 * pi * pi * pi * pi * pi * Bbar * Bbar) * \
        (self.d2_volume_d_psi2 - 4 * pi * mu0 * p2 * abs(G0 / Bbar) * np.sum((1/(B0 * B0 * B0 * B0)) * jacobian) * d_phi * nfp)

    self.DMerc_times_r2 = self.DWell_times_r2 + self.DGeod_times_r2
