"""
This module contains the calculation for the O(r^2) solution
"""

import logging
import numpy as np
from .util import mu0

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_r2(self):
    """
    Compute the O(r^2) quantities.
    """
    logger.debug('Calculating O(r^2) terms')
    # First, some shorthand:
    nphi = self.nphi
    Bbar = self.Bbar
    B0_over_abs_G0 = self.B0 / np.abs(self.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    lprime = abs_G0_over_B0
    G0_over_Bbar = self.G0 / self.Bbar
    X1c = self.X1c
    X1s = self.X1s
    Y1s = self.Y1s
    Y1c = self.Y1c
    d_d_varphi = self.d_d_varphi
    iota_N = self.iotaN
    iota = self.iota
    curvature = self.curvature
    torsion = self.torsion
    B0 = self.B0
    G0 = self.G0
    I2 = self.I2
    B1c = self.B1c
    B1s = self.B1s
    # B2s = self.B2s
    # B2c = self.B2c
    p2 = self.p2
    sG = self.sG
    I2_over_Bbar = self.I2 / self.Bbar

    if self.omn:
        # If in QI, the magnetic field is specified in terms of varphi instead of phi
        integral_one_over_B0_squared_over_varphi = np.sum(1 / (B0 * B0)) / nphi
        # B2 in QI has an alpha and theta instead of vartheta
        B2cPlunk = np.array(sum([self.B2c_cvals[i]*np.cos(self.nfp*i*self.varphi) for i in range(len(self.B2c_cvals))]))
        B2cPlunk = B2cPlunk + np.array(sum([self.B2c_svals[i]*np.sin(self.nfp*i*self.varphi) for i in range(len(self.B2c_svals))]))
        B2sPlunk = np.array(sum([self.B2s_cvals[i]*np.cos(self.nfp*i*self.varphi) for i in range(len(self.B2s_cvals))]))
        B2sPlunk = B2sPlunk + np.array(sum([self.B2s_svals[i]*np.sin(self.nfp*i*self.varphi) for i in range(len(self.B2s_svals))]))
        # angle = self.alpha - (-self.helicity * self.nfp * self.varphi)
        # self.B2c_array = B2cPlunk * np.cos(2 * angle) - B2sPlunk * np.sin(2 * angle)
        # self.B2s_array = B2sPlunk * np.cos(2 * angle) + B2cPlunk * np.sin(2 * angle)
    else:
        # If not in QI, the magnetic field is specified in terms of phi
        integral_one_over_B0_squared_over_varphi = np.sum(self.d_l_d_phi / (abs(G0) * B0)) / nphi
        B2c = np.array(sum([self.B2c_cvals[i]*np.cos(self.nfp*i*self.phi) for i in range(len(self.B2c_cvals))]))
        B2c = B2c + np.array(sum([self.B2c_svals[i]*np.sin(self.nfp*i*self.phi) for i in range(len(self.B2c_svals))]))
        B2s = np.array(sum([self.B2s_cvals[i]*np.cos(self.nfp*i*self.phi) for i in range(len(self.B2s_cvals))]))
        B2s = B2s + np.array(sum([self.B2s_svals[i]*np.sin(self.nfp*i*self.phi) for i in range(len(self.B2s_svals))]))
        self.B2c_array=B2c
        self.B2s_array=B2s

    G2 = -mu0 * p2 * G0 * integral_one_over_B0_squared_over_varphi - iota * I2

    ## Calculate beta_0
    rhs_beta_0_equation = 2 * mu0 * p2 * G0 / Bbar * (1/(B0 * B0)-integral_one_over_B0_squared_over_varphi)
    if np.all(rhs_beta_0_equation == np.zeros((1,nphi))[0]):
        beta_0 = np.zeros((1,nphi))[0]
    else:
        beta_0 = np.linalg.solve(d_d_varphi, rhs_beta_0_equation)
        beta_0 = beta_0 - beta_0[0] # Fix to be zero at origin

    ## Calculate beta_1
    # beta_1c = 0
    # beta_1s = -4 * spsi * sG * mu0 * p2 * etabar * abs_G0_over_B0 / (iota_N * B0 * B0)
    matrix_beta_1 = np.zeros((2 * nphi, 2 * nphi))
    rhs_beta_1 = np.zeros(2 * nphi)
    for j in range(nphi):
        # Handle the terms involving d beta_1c / d zeta and d beta_1s / d zeta:
        # ----------------------------------------------------------------
        # Equation 1, terms involving beta_1c:
        matrix_beta_1[j, 0:nphi] = d_d_varphi[j, :]
        # Equation 2, terms involving beta_1s:
        matrix_beta_1[j+nphi, nphi:(2*nphi)] = d_d_varphi[j, :]

        # Now handle the terms involving beta_1c and beta_1s without d/dzeta derivatives:
        # ----------------------------------------------------------------
        matrix_beta_1[j, j + nphi] = matrix_beta_1[j, j + nphi] + iota_N
        matrix_beta_1[j + nphi, j       ] = matrix_beta_1[j + nphi, j       ] - iota_N

    rhs_beta_1[0:nphi] = -4 * mu0 * p2 * G0 * B1c / (Bbar * B0 * B0 * B0)
    rhs_beta_1[nphi:2 * nphi] = -4 * mu0 * p2 * G0 * B1s / (Bbar * B0 * B0 * B0)

    solution_beta_1 = np.linalg.solve(matrix_beta_1, rhs_beta_1)
    beta_1c = solution_beta_1[0:nphi]
    beta_1s = solution_beta_1[nphi:2 * nphi]

    if np.abs(iota_N) < 1e-8:
        print('Warning: |iota_N| is very small so O(r^2) solve will be poorly conditioned. iota_N=', iota_N)

    V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s + X1s * X1s
    V2 = 2 * (Y1s * Y1c + X1s * X1c)
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s - X1s * X1s

    factor = - B0_over_abs_G0 / 8
    Z20 = beta_0 * Bbar / (2 * G0 * B0_over_abs_G0) + factor * np.matmul(d_d_varphi,V1)
    Z2s = factor*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = factor*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * torsion * abs_G0_over_B0
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * torsion * abs_G0_over_B0
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    # X2s = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c + B0_over_abs_G0 * ( abs_G0_over_B0*abs_G0_over_B0*self.B2s_array/B0 + (qc * qs + rc * rs)/2)) / curvature
    if self.omn:
        # X2s = self.B2s_array
        X2s = np.array(sum([self.B2s_svals[i]*np.sin(self.nfp*i*self.varphi) for i in range(len(self.B2s_svals))])) \
            + np.array(sum([self.B2s_cvals[i]*np.cos(self.nfp*i*self.varphi) for i in range(len(self.B2s_cvals))]))
        # self.B2s_array = X2s * curvature * B0 - B0 * B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c - B0_over_abs_G0 * ( \
        #     + 3 * G0 * G0 * B1c * B1s / (2 * B0**4) - X1c * X1s / 2 * (curvature * abs_G0_over_B0)**2 - (qc * qs + rc * rs)/2))
        self.B2s_array = 3 * B1c * B1s / (2 * B0) \
                       - B0/(2*G0*G0)*( B0 * B0 * (qc * qs + rc * rs) \
                                      + curvature * G0 * G0 * (curvature * X1c * X1s - 2 * X2s) \
                                      + 2 * B0 * G0 * sG * (np.matmul(d_d_varphi, Z2s) - 2 * Z2c * iota_N) )
    else:
        X2s = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c - B0_over_abs_G0 * ( -abs_G0_over_B0*abs_G0_over_B0*self.B2s_array/B0 \
            + 3 * G0 * G0 * B1c * B1s / (2 * B0**4) - X1c * X1s / 2 * (curvature * abs_G0_over_B0)**2 - (qc * qs + rc * rs)/2)) / curvature

    # X2c = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*self.B2c_array/B0 \
    #        + abs_G0_over_B0*abs_G0_over_B0*etabar*etabar/2 - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature
    if self.omn:
        # X2c = self.B2c_array
        X2c = np.array(sum([self.B2c_cvals[i]*np.cos(self.nfp*i*self.varphi) for i in range(len(self.B2c_cvals))])) \
            + np.array(sum([self.B2c_svals[i]*np.sin(self.nfp*i*self.varphi) for i in range(len(self.B2c_svals))]))
        # self.B2c_array = X2c * curvature * B0 - B0 * B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * ( \
        #     + 3 * G0 * G0 * (B1c*B1c-B1s*B1s)/(4*B0**4) - (X1c*X1c - X1s*X1s)/4*(curvature*abs_G0_over_B0)**2 \
        #     - (qc * qc - qs * qs + rc * rc - rs * rs)/4))
        self.B2c_array = (1/(4*B0)) * (3 * B1c * B1c - 3 * B1s * B1s + B0_over_abs_G0 * B0_over_abs_G0 \
                                      * ( B0 * B0 * (-qc * qc + qs * qs - rc * rc + rs * rs) \
                                        + G0 * G0 * curvature * (curvature * (-X1c * X1c + X1s * X1s) + 4 * X2c)
                                        - 4 * B0 * G0 * sG * (np.matmul(d_d_varphi, Z2c) + 2 * Z2s * iota_N)))
    else:
        X2c = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*self.B2c_array/B0 \
               + 3 * G0 * G0 * (B1c*B1c-B1s*B1s)/(4*B0**4) - (X1c*X1c - X1s*X1s)/4*(curvature*abs_G0_over_B0)**2 \
               - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature

    # Y2s_from_X20 = -sG * spsi * curvature * curvature / (etabar * etabar)
    Y2s_from_X20 = -(X1s * Y1c + X1c * Y1s) / (X1c * X1c + X1s * X1s + 1e-30)
    # Y2s_inhomogeneous = sG * spsi * (-curvature/2 + curvature*curvature/(etabar*etabar)*(-X2c + X2s * sigma))
    Y2s_inhomogeneous = 1/(2 * B0 * (X1c * X1c + X1s * X1s+ 1e-30))*(Bbar * curvature * sG * (-X1c * X1c + X1s * X1s)\
        + 2 * B0 * (X1s * X2c * Y1c + X1c * X2s * Y1c - X1c * X2c * Y1s + X1s * X2s * Y1s))
    # Y2s_from_Y20 = 0 * X1s
    Y2s_from_Y20 = 2 * X1c * X1s / (X1c * X1c + X1s * X1s+ 1e-30)

    # Y2c_from_X20 = -sG * spsi * curvature * curvature * sigma / (etabar * etabar)
    Y2c_from_X20 = (-X1c * Y1c + X1s * Y1s) / (X1c * X1c + X1s * X1s+ 1e-30)
    # Y2c_from_Y20 = np.full(nphi, 1)
    Y2c_from_Y20 = (X1c * X1c - X1s * X1s) / (X1c * X1c + X1s * X1s+ 1e-30)
    # Y2c_inhomogeneous = sG * spsi * curvature * curvature / (etabar * etabar) * (X2s + X2c * sigma)
    Y2c_inhomogeneous = (Bbar * curvature * sG * X1c * X1s + B0 * (X1c * X2c * Y1c - X1s * X2s * Y1c\
        + X1s * X2c * Y1s + X1c * X2s * Y1s)) / (B0 * (X1c * X1c + X1s * X1s+ 1e-30))

    # Note: in the fX* and fY* quantities below, I've omitted the
    # contributions from X20 and Y20 to the d/dzeta terms. These
    # contributions are handled later when we assemble the large
    # matrix.

    # fX0_from_X20 = -4 * sG * spsi * abs_G0_over_B0 * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_X20 = -4 * G0_over_Bbar * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    # fX0_from_Y20 = -torsion * abs_G0_over_B0 - 4 * sG * spsi * abs_G0_over_B0 * (Z2s) \
    #     - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fX0_from_Y20 = -torsion * abs_G0_over_B0 -4 * G0_over_Bbar * (Y2c_from_Y20 * Z2s - Y2s_from_Y20 * Z2c) \
        - I2_over_Bbar * (-2) * abs_G0_over_B0
    # fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * sG * spsi * abs_G0_over_B0 * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
    #     - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi) * abs_G0_over_B0 + beta_1s * abs_G0_over_B0 / 2 * Y1c
    fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
        - I2_over_Bbar * 0.5 * curvature * (X1s*Y1s + X1c*Y1c) * abs_G0_over_B0 - 0.5 * beta_0 * curvature * abs_G0_over_B0 * (X1s * Y1c - X1c * Y1s)\
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s - beta_1s * Y1c)

    # fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_from_X20 * Z20) \
    #     - spsi * I2_over_B0 * (- 2 * Y2s_from_X20) * abs_G0_over_B0
    fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * G0_over_Bbar * (Y2c_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_X20)
    # fXs_from_Y20 = - 4 * spsi * sG * abs_G0_over_B0 * (-Z2c + Z20)
    fXs_from_Y20 = -torsion * abs_G0_over_B0 * Y2s_from_Y20 - 4 * G0_over_Bbar * (-Z2c + Y2c_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_Y20)
    # fXs_inhomogeneous = np.matmul(d_d_varphi,X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
    #     - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_inhomogeneous * Z20) \
    #     - spsi * I2_over_B0 * (0.5 * curvature * spsi * sG - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
    #     - (0.5) * abs_G0_over_B0 * beta_1s * Y1s
    fXs_inhomogeneous = np.matmul(d_d_varphi,X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1s * Y1c + X1c * Y1s) - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * Y2c_inhomogeneous + 0.5 * curvature * (X1c * Y1c - X1s * Y1s)) \
        - 0.5 * abs_G0_over_B0 * (beta_1s * Y1s - beta_1c * Y1c)

    # fXc_from_X20 = - torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_from_X20 * Z20) \
    #     - spsi * I2_over_B0 * (- 2 * Y2c_from_X20) * abs_G0_over_B0
    fXc_from_X20 =  -torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * G0_over_Bbar * (-Y2s_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_X20)
    # fXc_from_Y20 = - torsion * abs_G0_over_B0 - 4 * spsi * sG * abs_G0_over_B0 * (Z2s) \
    #     - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fXc_from_Y20 = -torsion * abs_G0_over_B0 * Y2c_from_Y20 - 4 * G0_over_Bbar * (Z2s - Y2s_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_Y20)
    # fXc_inhomogeneous = np.matmul(d_d_varphi,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
    #     - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_inhomogeneous * Z20) \
    #     - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
    #     - (0.5) * abs_G0_over_B0 * beta_1s * Y1c
    fXc_inhomogeneous = np.matmul(d_d_varphi,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * G0_over_Bbar * (-Y2s_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1c * Y1c - X1s * Y1s) - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * Y2s_inhomogeneous - 0.5 * curvature * (X1c * Y1s + X1s * Y1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s + beta_1s * Y1c)

    # fY0_from_X20 = torsion * abs_G0_over_B0 - spsi * I2_over_B0 * (2) * abs_G0_over_B0
    fY0_from_X20 = torsion * abs_G0_over_B0 - I2_over_Bbar * (2) * abs_G0_over_B0
    # fY0_from_Y20 = np.zeros(nphi)
    fY0_from_Y20 = np.zeros(nphi)
    # fY0_inhomogeneous = -4 * spsi * sG * abs_G0_over_B0 * (X2s * Z2c - X2c * Z2s) \
    #     - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c) * abs_G0_over_B0 - (0.5) * abs_G0_over_B0 * beta_1s * X1c
    fY0_inhomogeneous = -4 * G0_over_Bbar * (X2s * Z2c - X2c * Z2s) \
        - I2_over_Bbar * (-0.5 * curvature * (X1s * X1s + X1c * X1c)) * abs_G0_over_B0 \
        - 0.5 * abs_G0_over_B0 * (beta_1s * X1c - beta_1c * X1s)

    # fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Z2c)
    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * G0_over_Bbar * (Z2c)
    # fYs_from_Y20 = np.full(nphi, -2 * iota_N)
    fYs_from_Y20 = -2 * iota_N * Y2c_from_Y20
    # fYs_inhomogeneous = np.matmul(d_d_varphi,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
    #     - 4 * spsi * sG * abs_G0_over_B0 * (-X2c * Z20) - 2 * spsi * I2_over_B0 * X2s * abs_G0_over_B0
    fYs_inhomogeneous = np.matmul(d_d_varphi,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
        - 4 * G0_over_Bbar * (-X2c * Z20) - I2_over_Bbar * (-curvature * X1s * X1c + 2 * X2s) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * X2c + 0.5 * curvature*  (X1s * X1s - X1c * X1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * X1c - beta_1s * X1s)

    # fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Z2s)
    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * G0_over_Bbar * (-Z2s)
    # fYc_from_Y20 = np.zeros(nphi)
    fYc_from_Y20 = 2 * iota_N * Y2s_from_Y20
    # fYc_inhomogeneous = np.matmul(d_d_varphi,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
    #     - 4 * spsi * sG * abs_G0_over_B0 * (X2s * Z20) \
    #     - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c + 2 * X2c) * abs_G0_over_B0 + 0.5 * abs_G0_over_B0 * beta_1s * X1c
    fYc_inhomogeneous = np.matmul(d_d_varphi,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
        - 4 * G0_over_Bbar * (X2s * Z20) - I2_over_Bbar * (0.5 * curvature * (X1s * X1s - X1c * X1c) + 2 * X2c) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * X2s + curvature * X1s * X1c) \
        + 0.5 * abs_G0_over_B0 * (beta_1c * X1s + beta_1s * X1c)

    matrix = np.zeros((2 * nphi, 2 * nphi))
    right_hand_side = np.zeros(2 * nphi)
    for j in range(nphi):
        # Handle the terms involving d X_0 / d zeta and d Y_0 / d zeta:
        # ----------------------------------------------------------------

        # Equation 1, terms involving X0:
        # Contributions arise from -X1s * fX0 + Y1c * fYs - Y1s * fYc.
        # matrix[j, 0:nphi] = Y1c[j] * d_d_varphi[j, :] * Y2s_from_X20 - Y1s[j] * d_d_varphi[j, :] * Y2c_from_X20
        matrix[j, 0:nphi] = (-X1s[j] + Y1c[j] * Y2s_from_X20 - Y1s[j] * Y2c_from_X20) * d_d_varphi[j, :]
  
        # Equation 1, terms involving Y0:
        # Contributions arise from  -Y1s * fY0 + Y1c * fYs - Y1s * fYc.
        # matrix[j, nphi:(2*nphi)] = -2 * Y1s[j] * d_d_varphi[j, :]
        matrix[j, nphi:(2*nphi)] = (-Y1s[j] - Y1s[j] * Y2c_from_Y20 + Y1c[j] * Y2s_from_Y20) * d_d_varphi[j, :]

        # Equation 2, terms involving X0:
        # Contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
        # matrix[j+nphi, 0:nphi] = -X1c[j] * d_d_varphi[j, :] + Y1s[j] * d_d_varphi[j, :] * Y2s_from_X20 + Y1c[j] * d_d_varphi[j, :] * Y2c_from_X20
        matrix[j+nphi, 0:nphi] = (-X1c[j] + Y1s[j] * Y2s_from_X20 + Y1c[j] * Y2c_from_X20) * d_d_varphi[j, :]

        # Equation 2, terms involving Y0:
        # Contributions arise from -Y1c * fY0 + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, nphi:(2*nphi)] = (-Y1c[j] + Y1s[j] * Y2s_from_Y20 + Y1c[j] * Y2c_from_Y20) * d_d_varphi[j, :]

        # Now handle the terms involving X_0 and Y_0 without d/dzeta derivatives:
        # ----------------------------------------------------------------

        # matrix[j, j       ] = matrix[j, j       ] + X1c[j] * fXs_from_X20[j] - Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
        matrix[j, j       ] = matrix[j, j       ] - X1s[j] * fX0_from_X20[j] + X1c[j] * fXs_from_X20[j] - X1s[j] * fXc_from_X20[j] - Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
        # matrix[j, j + nphi] = matrix[j, j + nphi] + X1c[j] * fXs_from_Y20[j] - Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]
        matrix[j, j + nphi] = matrix[j, j + nphi] - X1s[j] * fX0_from_Y20[j] + X1c[j] * fXs_from_Y20[j] - X1s[j] * fXc_from_Y20[j] - Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]

        # matrix[j + nphi, j       ] = matrix[j + nphi, j       ] - X1c[j] * fX0_from_X20[j] + X1c[j] * fXc_from_X20[j] - Y1c[j] * fY0_from_X20[j] + Y1s[j] * fYs_from_X20[j] + Y1c[j] * fYc_from_X20[j]
        matrix[j + nphi, j       ] = matrix[j + nphi, j       ] - X1c[j] * fX0_from_X20[j] + X1s[j] * fXs_from_X20[j] + X1c[j] * fXc_from_X20[j] - Y1c[j] * fY0_from_X20[j] + Y1s[j] * fYs_from_X20[j] + Y1c[j] * fYc_from_X20[j]
        # matrix[j + nphi, j + nphi] = matrix[j + nphi, j + nphi] - X1c[j] * fX0_from_Y20[j] + X1c[j] * fXc_from_Y20[j] - Y1c[j] * fY0_from_Y20[j] + Y1s[j] * fYs_from_Y20[j] + Y1c[j] * fYc_from_Y20[j]
        matrix[j + nphi, j + nphi] = matrix[j + nphi, j + nphi] - X1c[j] * fX0_from_Y20[j] + X1s[j] * fXs_from_Y20[j] + X1c[j] * fXc_from_Y20[j] - Y1c[j] * fY0_from_Y20[j] + Y1s[j] * fYs_from_Y20[j] + Y1c[j] * fYc_from_Y20[j]


    right_hand_side[0:nphi] = -(-X1s * fX0_inhomogeneous + X1c * fXs_inhomogeneous - X1s * fXc_inhomogeneous - Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
    right_hand_side[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1s * fXs_inhomogeneous + X1c * fXc_inhomogeneous - Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)

    solution = np.linalg.solve(matrix, right_hand_side)
    X20 = solution[0:nphi]
    Y20 = solution[nphi:2 * nphi]

    # Now that we have X20 and Y20 explicitly, we can reconstruct Y2s, Y2c, and B20:
    Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20 + Y2s_from_Y20 * Y20
    Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y2c_from_Y20 * Y20

    # B20 = B0 * (curvature * X20 - B0_over_abs_G0 * np.matmul(d_d_varphi,Z20) + (0.5) * etabar * etabar - mu0 * p2 / (B0 * B0) \
    #             - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs))
    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * np.matmul(d_d_varphi, Z20)) + (3/(4*B0)) * (B1c*B1c + B1s*B1s)\
        + (B0/G0)*(G2 + iota * I2) - 0.25 * B0 * curvature * curvature * (X1c*X1c + X1s*X1s)\
        - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs)

    d_l_d_phi = self.d_l_d_phi
    normalizer = 1 / np.sum(d_l_d_phi)
    self.B20_mean = np.sum(B20 * d_l_d_phi) * normalizer
    self.B20_anomaly = B20 - self.B20_mean
    self.B20_residual = np.sqrt(np.sum((B20 - self.B20_mean) * (B20 - self.B20_mean) * d_l_d_phi) * normalizer) / B0
    self.B20_variation = np.max(B20) - np.min(B20)

    # In QI, B2=B20+B2cQI*Cos(2*theta-2*alpha)+B2sQI*Sin(2*theta-2*alpha)
    angle = self.alpha - (-self.helicity * self.nfp * self.varphi)
    self.B2cQI = self.B2c_array * np.cos(2*angle) + self.B2s_array * np.sin(2*angle)
    self.B2sQI = self.B2s_array * np.cos(2*angle) - self.B2c_array * np.sin(2*angle)


    self.G2 = G2

    self.d_X20_d_varphi = np.matmul(d_d_varphi, X20)
    self.d_X2s_d_varphi = np.matmul(d_d_varphi, X2s)
    self.d_X2c_d_varphi = np.matmul(d_d_varphi, X2c)
    self.d_Y20_d_varphi = np.matmul(d_d_varphi, Y20)
    self.d_Y2s_d_varphi = np.matmul(d_d_varphi, Y2s)
    self.d_Y2c_d_varphi = np.matmul(d_d_varphi, Y2c)
    self.d_Z20_d_varphi = np.matmul(d_d_varphi, Z20)
    self.d_Z2s_d_varphi = np.matmul(d_d_varphi, Z2s)
    self.d_Z2c_d_varphi = np.matmul(d_d_varphi, Z2c)
    self.d2_X1s_d_varphi2 = np.matmul(d_d_varphi, self.d_X1s_d_varphi)
    self.d2_X1c_d_varphi2 = np.matmul(d_d_varphi, self.d_X1c_d_varphi)
    self.d2_Y1c_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1c_d_varphi)
    self.d2_Y1s_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1s_d_varphi)

    # Store all important results in self:
    self.V1 = V1
    self.V2 = V2
    self.V3 = V3

    self.X20 = X20
    self.X2s = X2s
    self.X2c = X2c
    self.Y20 = Y20
    self.Y2s = Y2s
    self.Y2c = Y2c
    self.Z20 = Z20
    self.Z2s = Z2s
    self.Z2c = Z2c
    self.beta_0  = beta_0
    self.beta_1s = beta_1s
    self.beta_1c = beta_1c
    self.B20 = B20
    self.B20_spline = self.convert_to_spline(self.B20)
    self.B2c_spline = self.convert_to_spline(self.B2c_array)
    self.B2s_spline = self.convert_to_spline(self.B2s_array)
    # self.B2cQI_spline = self.convert_to_spline(self.B2cQI)
    # self.B2sQI_spline = self.convert_to_spline(self.B2sQI)
    # self.B20_of_varphi = self.B20_spline(self.varphi-self.nu_spline(self.phi))
    if self.omn:
        d_B0_d_varphi = np.matmul(self.d_d_varphi, self.B0)
        d_2_B0_d_varphi2 = np.matmul(self.d_d_varphi, d_B0_d_varphi)
        k_factor = self.d * self.B0 / d_B0_d_varphi
        d_dB0_d_varphi = np.matmul(self.d_d_varphi,k_factor) * d_B0_d_varphi + k_factor * d_2_B0_d_varphi2
        self.B2QI_factor = k_factor * d_dB0_d_varphi * (1 - (3/2) * k_factor * d_2_B0_d_varphi2/d_dB0_d_varphi)
        self.B20QI_deviation = self.B20   + self.B20[::-1]   + self.B2QI_factor
        self.B2cQI_deviation = self.B2cQI + self.B2cQI[::-1] + self.B2QI_factor
        self.B2sQI_deviation = self.B2sQI + self.B2sQI[::-1]
        self.B20QI_deviation_max = max(abs(self.B20QI_deviation))
        self.B2cQI_deviation_max = max(abs(self.B2cQI_deviation))
        self.B2sQI_deviation_max = max(abs(self.B2sQI_deviation))


    # O(r^2) diagnostics:
    self.mercier()
    self.calculate_grad_grad_B_tensor()
    #self.grad_grad_B_inverse_scale_length_vs_varphi = t.grad_grad_B_inverse_scale_length_vs_varphi
    #self.grad_grad_B_inverse_scale_length = t.grad_grad_B_inverse_scale_length
    self.calculate_r_singularity()

    if self.helicity == 0:
        self.X20_untwisted = self.X20
        self.X2s_untwisted = self.X2s
        self.X2c_untwisted = self.X2c
        self.Y20_untwisted = self.Y20
        self.Y2s_untwisted = self.Y2s
        self.Y2c_untwisted = self.Y2c
        self.Z20_untwisted = self.Z20
        self.Z2s_untwisted = self.Z2s
        self.Z2c_untwisted = self.Z2c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X20_untwisted = self.X20
        self.Y20_untwisted = self.Y20
        self.Z20_untwisted = self.Z20
        sinangle = np.sin(2*angle)
        cosangle = np.cos(2*angle)
        self.X2s_untwisted = self.X2s *   cosangle  + self.X2c * sinangle
        self.X2c_untwisted = self.X2s * (-sinangle) + self.X2c * cosangle
        self.Y2s_untwisted = self.Y2s *   cosangle  + self.Y2c * sinangle
        self.Y2c_untwisted = self.Y2s * (-sinangle) + self.Y2c * cosangle
        self.Z2s_untwisted = self.Z2s *   cosangle  + self.Z2c * sinangle
        self.Z2c_untwisted = self.Z2s * (-sinangle) + self.Z2c * cosangle
