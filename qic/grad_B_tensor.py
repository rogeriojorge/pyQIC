#!/usr/bin/env python3

"""
Functions for computing the grad B tensor and grad grad B tensor.
"""

import logging
import numpy as np
from .util import Struct, fourier_minimum

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_grad_B_tensor(self):
    """
    Compute the components of the grad B tensor, and the scale
    length L grad B associated with the Frobenius norm of this
    tensor.
    The formula for the grad B tensor is eq (3.12) of
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP

    self should be an instance of qic with X1c, Y1s etc populated.
    """

    s = self # Shorthand
    tensor = Struct()
    
    factor = s.B0 * s.B0 / s.d_l_d_varphi / s.Bbar
    tensor.tn = s.sG * s.B0 * s.curvature
    tensor.nt = tensor.tn
    tensor.bb = factor * (s.X1c * s.d_Y1s_d_varphi - s.X1s * s.d_Y1c_d_varphi \
                          - s.iotaN * (s.X1s * s.Y1s + s.X1c * s.Y1c))
    tensor.nn = factor * (s.d_X1c_d_varphi * s.Y1s - s.d_X1s_d_varphi * s.Y1c \
                          + s.iotaN * (s.X1s * s.Y1s + s.X1c * s.Y1c))
    tensor.bn = factor * (-s.sG * s.Bbar * s.d_l_d_varphi * s.torsion / s.B0 \
                          + s.X1c * s.d_X1s_d_varphi - s.X1s * s.d_X1c_d_varphi \
                          - s.iotaN * (s.X1c * s.X1c + s.X1s * s.X1s))
    tensor.nb = factor * (s.d_Y1c_d_varphi * s.Y1s - s.d_Y1s_d_varphi * s.Y1c \
                          + s.sG * s.Bbar * s.d_l_d_varphi * s.torsion / s.B0 \
                          + s.iotaN * (s.Y1s * s.Y1s + s.Y1c * s.Y1c))
    if hasattr(s.B0, "__len__"): # check if B0 is an array (in quasisymmetry B0 is a scalar)
        tensor.tt = s.sG * np.matmul(s.d_d_varphi, s.B0) / s.d_l_d_varphi
    else:
        tensor.tt = 0

    self.grad_B_tensor = tensor
    
    t = s.tangent_cylindrical.transpose()
    n = s.normal_cylindrical.transpose()
    b = s.binormal_cylindrical.transpose()
    self.grad_B_tensor_cylindrical = np.array([[
                              tensor.nn * n[i] * n[j] \
                            + tensor.bn * b[i] * n[j] + tensor.nb * n[i] * b[j] \
                            + tensor.bb * b[i] * b[j] \
                            + tensor.tn * t[i] * n[j] + tensor.nt * n[i] * t[j] \
                            + tensor.tt * t[i] * t[j]
                        for i in range(3)] for j in range(3)])
    self.grad_B_tensor_cylindrical_array = np.reshape(self.grad_B_tensor_cylindrical, 9 * self.nphi)

    self.grad_B_colon_grad_B = tensor.tn * tensor.tn + tensor.nt * tensor.nt \
        + tensor.bb * tensor.bb + tensor.nn * tensor.nn \
        + tensor.nb * tensor.nb + tensor.bn * tensor.bn \
        + tensor.tt * tensor.tt

    self.L_grad_B = s.B0 * np.sqrt(2 / self.grad_B_colon_grad_B)
    self.inv_L_grad_B = 1.0 / self.L_grad_B
    self.min_L_grad_B = fourier_minimum(self.L_grad_B)
    
def calculate_grad_grad_B_tensor(self, two_ways=False):
    """
    Compute the components of the grad grad B tensor, and the scale
    length L grad grad B associated with the Frobenius norm of this
    tensor.
    self should be an instance of Qic with X1c, Y1s etc populated.
    The grad grad B tensor in discussed around eq (3.13)
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP
    although an explicit formula is not given there.

    If ``two_ways`` is ``True``, an independent calculation of
    the tensor is also computed, to confirm the answer is the same.
    """

    # Shortcuts
    s = self
    
    X1s = s.X1s
    X1c = s.X1c
    Y1s = s.Y1s
    Y1c = s.Y1c

    X20 = s.X20
    X2s = s.X2s
    X2c = s.X2c

    Y20 = s.Y20
    Y2s = s.Y2s
    Y2c = s.Y2c

    Z20 = s.Z20
    Z2s = s.Z2s
    Z2c = s.Z2c

    iota_N0 = s.iotaN
    iota = s.iota
    iota0 = iota
    lp = np.abs(s.G0) / s.B0

    curvature = s.curvature
    torsion = s.torsion
    d_l_d_varphi = self.d_l_d_varphi
    lprime = d_l_d_varphi

    sG = s.sG
    sign_G = s.sG
    sign_psi = s.spsi
    B0 = s.B0
    Bbar = s.Bbar
    G0 = s.G0
    I2 = s.I2
    G2 = s.G2
    p2 = s.p2

    B20 = s.B20
    B2s = s.B2s
    B2c = s.B2c

    d_B0_d_varphi = np.matmul(s.d_d_varphi, B0)

    d_X1s_d_varphi = s.d_X1s_d_varphi
    d_X1c_d_varphi = s.d_X1c_d_varphi
    d_Y1s_d_varphi = s.d_Y1s_d_varphi
    d_Y1c_d_varphi = s.d_Y1c_d_varphi

    d_X20_d_varphi = s.d_X20_d_varphi
    d_X2s_d_varphi = s.d_X2s_d_varphi
    d_X2c_d_varphi = s.d_X2c_d_varphi

    d_Y20_d_varphi = s.d_Y20_d_varphi
    d_Y2s_d_varphi = s.d_Y2s_d_varphi
    d_Y2c_d_varphi = s.d_Y2c_d_varphi

    d_Z20_d_varphi = s.d_Z20_d_varphi
    d_Z2s_d_varphi = s.d_Z2s_d_varphi
    d_Z2c_d_varphi = s.d_Z2c_d_varphi

    d2_B0_d_varphi2  = np.matmul(s.d_d_varphi, d_B0_d_varphi)
    d2_X1s_d_varphi2 = s.d2_X1s_d_varphi2
    d2_X1c_d_varphi2 = s.d2_X1c_d_varphi2
    d2_Y1s_d_varphi2 = s.d2_Y1s_d_varphi2
    d2_Y1c_d_varphi2 = s.d2_Y1c_d_varphi2
    d_curvature_d_varphi = s.d_curvature_d_varphi
    d_torsion_d_varphi = s.d_torsion_d_varphi

    grad_grad_B = np.zeros((s.nphi, 3, 3, 3))
    grad_grad_B_alt = np.zeros((s.nphi, 3, 3, 3))

    # The elements that follow are computed in the Mathematica notebook "20200407-01 Grad grad B tensor near axis"
    # and then formatted for fortran by the python script process_grad_grad_B_tensor_code

    # The order is (normal, binormal, tangent). So element 123 means nbt.

    # Element111
    grad_grad_B[:,0,0,0]=(B0**6*lprime*lprime*(8*iota_N0*X2c*Y1c*Y1s + 4*iota_N0*X2s*(-Y1c*Y1c + Y1s*Y1s) - \
        2*iota_N0*X1s*Y1c*Y20 + 2*iota_N0*X1c*Y1s*Y20 + 2*iota_N0*X1s*Y1c*Y2c + 2*iota_N0*X1c*Y1s*Y2c - \
        2*iota_N0*X1c*Y1c*Y2s + 2*iota_N0*X1s*Y1s*Y2s - 5*iota_N0*X1c*X1s*Y1c*Y1c*curvature + 5*iota_N0*X1c*X1c*Y1c*Y1s*curvature - \
        5*iota_N0*X1s*X1s*Y1c*Y1s*curvature + 5*iota_N0*X1c*X1s*Y1s*Y1s*curvature - 2*Y1c*Y20*d_X1c_d_varphi + \
        2*Y1c*Y2c*d_X1c_d_varphi + 2*Y1s*Y2s*d_X1c_d_varphi - 5*X1s*Y1c*Y1s*curvature*d_X1c_d_varphi + \
        5*X1c*Y1s*Y1s*curvature*d_X1c_d_varphi - 2*Y1s*Y20*d_X1s_d_varphi - 2*Y1s*Y2c*d_X1s_d_varphi + \
        2*Y1c*Y2s*d_X1s_d_varphi + 5*X1s*Y1c*Y1c*curvature*d_X1s_d_varphi - 5*X1c*Y1c*Y1s*curvature*d_X1s_d_varphi + \
        2*Y1c*Y1c*d_X20_d_varphi + 2*Y1s*Y1s*d_X20_d_varphi - 2*Y1c*Y1c*d_X2c_d_varphi + 2*Y1s*Y1s*d_X2c_d_varphi - \
        4*Y1c*Y1s*d_X2s_d_varphi))/(Bbar**2*G0*G0*G0)

    # Element112
    grad_grad_B[:,0,0,1]=(B0**6*lprime*lprime*(-5*iota_N0*X1s*Y1c*Y1c*Y1c*curvature + Y1c*Y1c * (-6*iota_N0*Y2s + \
        5*iota_N0*X1c*Y1s*curvature + 2*lprime*X20*torsion - 2*lprime*X2c*torsion + 5*lprime*X1s*X1s*curvature*torsion + \
        5*X1s*curvature*d_Y1s_d_varphi + 2*d_Y20_d_varphi - 2*d_Y2c_d_varphi) + Y1s * (5*iota_N0*X1c*Y1s*Y1s*curvature - \
        2 * (lprime * (X1s*(Y20 + Y2c) - X1c*Y2s) * torsion - Y2s*d_Y1c_d_varphi + (Y20 + Y2c) * d_Y1s_d_varphi) + \
        Y1s * (6*iota_N0*Y2s + lprime * (2*X20 + 2*X2c + 5*X1c*X1c*curvature) * torsion + 5*X1c*curvature*d_Y1c_d_varphi + \
        2*d_Y20_d_varphi + 2*d_Y2c_d_varphi)) - Y1c * (5*iota_N0*X1s*Y1s*Y1s*curvature + 2*lprime * (X1c*(Y20 - Y2c) - \
        X1s*Y2s) * torsion + 2*Y20*d_Y1c_d_varphi - 2*Y2c*d_Y1c_d_varphi - 2*Y2s*d_Y1s_d_varphi + Y1s * (-12*iota_N0*Y2c + \
        2*lprime * (2*X2s + 5*X1c*X1s*curvature) * torsion + 5*X1s*curvature*d_Y1c_d_varphi + 5*X1c*curvature*d_Y1s_d_varphi + \
        4*d_Y2s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element113
    grad_grad_B[:,0,0,2]= - ((B0**5*lprime*lprime*(2*G0 * (4*B2s*lprime*Y1c*Y1s + 2*B2c*lprime*(Y1c*Y1c - Y1s*Y1s) - \
        2*B20*lprime*(Y1c*Y1c + Y1s*Y1s) + Y1c*Y1c*Z20*d_B0_d_varphi + Y1s*Y1s*Z20*d_B0_d_varphi - Y1c*Y1c*Z2c*d_B0_d_varphi + \
        Y1s*Y1s*Z2c*d_B0_d_varphi - 2*Y1c*Y1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(Y1c*Y1c*(2*G2 + 2*I2*iota + 2*G0*X20*curvature - \
        2*G0*X2c*curvature + G0*X1s*X1s*curvature*curvature) - 2*G0*Y1c*curvature*(2*X2s*Y1s + X1s*Y2s + X1c*(-Y20 + Y2c + \
        X1s*Y1s*curvature)) + Y1s * (2*G0 * (X1s*(Y20 + Y2c) - X1c*Y2s) * curvature + Y1s*(2*G2 + 2*I2*iota + 2*G0*X20*curvature + \
        2*G0*X2c*curvature + G0*X1c*X1c*curvature*curvature))) + 2*G0 * (Y1c*Y1c*(2*iota_N0*Z2s - d_Z20_d_varphi + \
        d_Z2c_d_varphi) - Y1s*Y1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) + 2*Y1c*Y1s*(-2*iota_N0*Z2c + \
        d_Z2s_d_varphi)))))/(Bbar**2*G0*G0*G0*G0))

    # Element121
    grad_grad_B[:,0,1,0]= - ((B0**6*lprime*lprime*(-3*iota_N0*X1s*X1s*X1s*Y1c*curvature + 3*iota_N0*X1c*X1c*X1c*Y1s*curvature + \
        3*X1s*X1s*curvature*(iota_N0*X1c*Y1s + Y1c * (lprime*Y1c*torsion - d_X1c_d_varphi)) + 3*X1c*X1c*Y1s*curvature*(lprime*Y1s*torsion - \
        d_X1s_d_varphi) + 2 * (lprime * (-2*X2s*Y1c*Y1s + X2c*(-Y1c*Y1c + Y1s*Y1s) + X20 * (Y1c*Y1c + Y1s*Y1s)) * torsion + \
        X2c*Y1c*d_X1c_d_varphi + X2s*Y1s*d_X1c_d_varphi + X2s*Y1c*d_X1s_d_varphi - X2c*Y1s*d_X1s_d_varphi - X20 * (Y1c*d_X1c_d_varphi + \
        Y1s*d_X1s_d_varphi)) + X1s * (-2*iota_N0*X20*Y1c + 6*iota_N0*X2c*Y1c + 6*iota_N0*X2s*Y1s - 3*iota_N0*X1c*X1c*Y1c*curvature - \
        2*lprime*Y1s*Y20*torsion - 2*lprime*Y1s*Y2c*torsion + 2*lprime*Y1c*Y2s*torsion - 6*lprime*X1c*Y1c*Y1s*curvature*torsion + \
        3*X1c*Y1s*curvature*d_X1c_d_varphi + 3*X1c*Y1c*curvature*d_X1s_d_varphi + 2*Y1s*d_X20_d_varphi + 2*Y1s*d_X2c_d_varphi -\
        2*Y1c*d_X2s_d_varphi) - 2*X1c * (3*iota_N0*X2s*Y1c - iota_N0*X20*Y1s - 3*iota_N0*X2c*Y1s + lprime*Y1c*Y20*torsion - \
        lprime*Y1c*Y2c*torsion - lprime*Y1s*Y2s*torsion - Y1c*d_X20_d_varphi + Y1c*d_X2c_d_varphi + \
        Y1s*d_X2s_d_varphi)))/(Bbar**2*G0*G0*G0))

    # Element122
    grad_grad_B[:,0,1,1]=(B0**6*lprime*lprime*(-4*iota_N0*X1s*Y1c*Y2c - 4*iota_N0*X1c*Y1s*Y2c + 4*iota_N0*X1c*Y1c*Y2s -\
        4*iota_N0*X1s*Y1s*Y2s + 3*iota_N0*X1c*X1s*Y1c*Y1c*curvature - 3*iota_N0*X1c*X1c*Y1c*Y1s*curvature + \
        3*iota_N0*X1s*X1s*Y1c*Y1s*curvature - 3*iota_N0*X1c*X1s*Y1s*Y1s*curvature + 2*X20*Y1c*d_Y1c_d_varphi + \
        3*X1s*X1s*Y1c*curvature*d_Y1c_d_varphi - 3*X1c*X1s*Y1s*curvature*d_Y1c_d_varphi - 2*X2c*Y1c*(2*iota_N0*Y1s + \
        d_Y1c_d_varphi) + 2*X20*Y1s*d_Y1s_d_varphi + 2*X2c*Y1s*d_Y1s_d_varphi - 3*X1c*X1s*Y1c*curvature*d_Y1s_d_varphi + \
        3*X1c*X1c*Y1s*curvature*d_Y1s_d_varphi + 2*X2s * (iota_N0*Y1c*Y1c - Y1s * (iota_N0*Y1s + d_Y1c_d_varphi) - \
        Y1c*d_Y1s_d_varphi) - 2*X1c*Y1c*d_Y20_d_varphi - 2*X1s*Y1s*d_Y20_d_varphi + 2*X1c*Y1c*d_Y2c_d_varphi - \
        2*X1s*Y1s*d_Y2c_d_varphi + 2*X1s*Y1c*d_Y2s_d_varphi + 2*X1c*Y1s*d_Y2s_d_varphi))/(Bbar**2*G0*G0*G0)

    # Element123
    grad_grad_B[:,0,1,2]=(2*B0**5*lprime*lprime*(G0 * (2*B2s*lprime*X1s*Y1c + 2*B2s*lprime*X1c*Y1s + 2*B2c*lprime*(X1c*Y1c - \
        X1s*Y1s) - 2*B20*lprime*(X1c*Y1c + X1s*Y1s) + X1c*Y1c*Z20*d_B0_d_varphi + X1s*Y1s*Z20*d_B0_d_varphi -\
        X1c*Y1c*Z2c*d_B0_d_varphi + X1s*Y1s*Z2c*d_B0_d_varphi - X1s*Y1c*Z2s*d_B0_d_varphi - X1c*Y1s*Z2s*d_B0_d_varphi) +\
        B0 * (lprime*(X1c * (-2*G0*X2s*Y1s*curvature + Y1c*(G2 + I2*iota + 2*G0*X20*curvature - 2*G0*X2c*curvature)) + \
        X1s * (-2*G0*X2s*Y1c*curvature + Y1s*(G2 + I2*iota + 2*G0*X20*curvature + 2*G0*X2c*curvature))) - G0 * (X1s * (Y1s*(2*iota_N0*Z2s + \
        d_Z20_d_varphi + d_Z2c_d_varphi) + Y1c*(2*iota_N0*Z2c - d_Z2s_d_varphi)) + X1c * (Y1c*(-2*iota_N0*Z2s + d_Z20_d_varphi - \
        d_Z2c_d_varphi) + Y1s*(2*iota_N0*Z2c - d_Z2s_d_varphi))))))/(Bbar**2*G0*G0*G0*G0)

    # Element131
    grad_grad_B[:,0,2,0]=(B0**5*lprime*((2*Bbar*sG*d_B0_d_varphi*(iota_N0*X1c*Y1c + iota_N0*X1s*Y1s + Y1s*d_X1c_d_varphi - \
        Y1c*d_X1s_d_varphi))/B0 + B0*(lprime*lprime*curvature*(-2*X2c*Y1c*Y1c - 4*X2s*Y1c*Y1s + 2*X2c*Y1s*Y1s + 2*X20*(Y1c*Y1c +\
        Y1s*Y1s) - 2*X1c*Y1c*Y20 - 2*X1s*Y1s*Y20 + 2*X1c*Y1c*Y2c - 2*X1s*Y1s*Y2c + 2*X1s*Y1c*Y2s + 2*X1c*Y1s*Y2s + \
        3*X1s*X1s*Y1c*Y1c*curvature - 6*X1c*X1s*Y1c*Y1s*curvature + 3*X1c*X1c*Y1s*Y1s*curvature) - Y1s*Y1s*d_X1c_d_varphi*d_X1c_d_varphi + \
        iota_N0*X1c*Y1c*Y1c*d_X1s_d_varphi + iota_N0*X1c*Y1s*Y1s*d_X1s_d_varphi + 2*Y1c*Y1s*d_X1c_d_varphi*d_X1s_d_varphi -\
        Y1c*Y1c*d_X1s_d_varphi**2 + iota_N0*X1c*X1c*Y1s*d_Y1c_d_varphi - X1c*Y1s*d_X1s_d_varphi*d_Y1c_d_varphi - \
        iota_N0*X1c*X1c*Y1c*d_Y1s_d_varphi + X1c*Y1c*d_X1s_d_varphi*d_Y1s_d_varphi + iota_N0*X1s*X1s*(Y1s*d_Y1c_d_varphi -\
        Y1c*d_Y1s_d_varphi) + (Bbar*sG*lprime*torsion*(iota_N0*X1c*X1c + iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s +\
        X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + Y1c*d_Y1s_d_varphi))/B0 + X1c*Y1s*Y1s*d2_X1c_d_varphi2 -\
        X1s * (Y1s*d_X1c_d_varphi*(iota_N0*Y1s - d_Y1c_d_varphi) + Y1c * (d_X1c_d_varphi*d_Y1s_d_varphi + Y1s*d2_X1c_d_varphi2) + \
        Y1c*Y1c*(iota_N0*d_X1c_d_varphi - d2_X1s_d_varphi2)) - X1c*Y1c*Y1s*d2_X1s_d_varphi2)))/(Bbar**2*G0*G0*G0)

    # Element132
    grad_grad_B[:,0,2,1]=(B0**5*lprime*(-((Bbar*sG*d_B0_d_varphi*(-2*iota_N0*Y1c*Y1c - Y1s*(2*iota_N0*Y1s + lprime*X1c*torsion + \
        2*d_Y1c_d_varphi) + Y1c * (lprime*X1s*torsion + 2*d_Y1s_d_varphi)))/B0) + B0*(iota_N0*Y1c*Y1c*Y1c*d_X1s_d_varphi + \
        (Bbar*sG*lprime*(torsion*(Y1s*d_X1c_d_varphi - Y1c*d_X1s_d_varphi) + X1s*(2*iota_N0*Y1s*torsion + torsion*d_Y1c_d_varphi -\
        Y1c*d_torsion_d_varphi) + X1c*(2*iota_N0*Y1c*torsion - torsion*d_Y1s_d_varphi + Y1s*d_torsion_d_varphi)))/B0 + \
        Y1s*(-(iota_N0*Y1s*Y1s*d_X1c_d_varphi) + d_Y1c_d_varphi*(X1s*d_Y1c_d_varphi - X1c*d_Y1s_d_varphi) + \
        Y1s * (iota_N0*X1s*d_Y1c_d_varphi - d_X1c_d_varphi*d_Y1c_d_varphi + X1c*(iota_N0*d_Y1s_d_varphi + d2_Y1c_d_varphi2))) +\
        Y1c * (iota_N0*Y1s*Y1s*d_X1s_d_varphi + d_Y1s_d_varphi*(-(X1s*d_Y1c_d_varphi) + X1c*d_Y1s_d_varphi) + \
        Y1s * (d_X1s_d_varphi*d_Y1c_d_varphi - 2*iota_N0*X1s*d_Y1s_d_varphi + d_X1c_d_varphi*d_Y1s_d_varphi - X1s*d2_Y1c_d_varphi2 + \
        X1c*(2*iota_N0*d_Y1c_d_varphi - d2_Y1s_d_varphi2))) - Y1c*Y1c * (iota_N0*Y1s*d_X1c_d_varphi + (iota_N0*X1c +\
        d_X1s_d_varphi) * d_Y1s_d_varphi + X1s * (iota_N0*d_Y1c_d_varphi - d2_Y1s_d_varphi2)))))/(Bbar**2*G0*G0*G0)

    # Element133
    grad_grad_B[:,0,2,2]=(B0**5*lprime*lprime*((-2*X2c*Y1c*Y1c - 4*X2s*Y1c*Y1s + 2*X2c*Y1s*Y1s + 2*X20 * (Y1c*Y1c + Y1s*Y1s) -\
        2*X1c*Y1c*Y20 - 2*X1s*Y1s*Y20 + 2*X1c*Y1c*Y2c - 2*X1s*Y1s*Y2c + 2*X1s*Y1c*Y2s + 2*X1c*Y1s*Y2s + 3*X1s*X1s*Y1c*Y1c*curvature -\
        6*X1c*X1s*Y1c*Y1s*curvature + 3*X1c*X1c*Y1s*Y1s*curvature) * d_B0_d_varphi - Bbar*sG * (iota_N0*X1s*Y1s*curvature +\
        Y1s*curvature*d_X1c_d_varphi - Y1c*curvature*d_X1s_d_varphi + X1s*Y1c*d_curvature_d_varphi + X1c * (iota_N0*Y1c*curvature -\
            Y1s*d_curvature_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element211
    grad_grad_B[:,1,0,0]=(-2*B0**6*lprime*lprime*(-(iota_N0*X1s*X1s*X1s*Y1c*curvature) + X1s*X1s * (iota_N0*Y2s + \
        curvature*(iota_N0*X1c*Y1s + lprime*Y1c*Y1c*torsion - Y1c*d_X1c_d_varphi)) + X1s * (2*iota_N0*X2c*Y1c + 2*iota_N0*X2s*Y1s +\
        2*iota_N0*X1c*Y2c - iota_N0*X1c*X1c*Y1c*curvature - 2*lprime*X1c*Y1c*Y1s*curvature*torsion + Y2s*d_X1c_d_varphi + \
        X1c*Y1s*curvature*d_X1c_d_varphi - Y20*d_X1s_d_varphi - Y2c*d_X1s_d_varphi + X1c*Y1c*curvature*d_X1s_d_varphi + \
        Y1s*d_X20_d_varphi + Y1s*d_X2c_d_varphi - Y1c*d_X2s_d_varphi) + X1c * (-2*iota_N0*X2s*Y1c + 2*iota_N0*X2c*Y1s - \
        iota_N0*X1c*Y2s + iota_N0*X1c*X1c*Y1s*curvature + lprime*X1c*Y1s*Y1s*curvature*torsion - Y20*d_X1c_d_varphi + \
        Y2c*d_X1c_d_varphi + Y2s*d_X1s_d_varphi - X1c*Y1s*curvature*d_X1s_d_varphi + Y1c*d_X20_d_varphi - Y1c*d_X2c_d_varphi - \
        Y1s*d_X2s_d_varphi)))/(Bbar**2*G0*G0*G0)

    # Element212
    grad_grad_B[:,1,0,1]=(2*B0**6*lprime*lprime*(X1s*X1s * (lprime*(Y20 + Y2c) * torsion + Y1c*curvature*(iota_N0*Y1s + d_Y1c_d_varphi)) - \
        X1s * (-(iota_N0*X1c*Y1c*Y1c*curvature) + iota_N0*X1c*Y1s*Y1s*curvature + 2*lprime*X1c*Y2s*torsion + Y2s*d_Y1c_d_varphi - \
        Y20*d_Y1s_d_varphi - Y2c*d_Y1s_d_varphi + Y1s * (3*iota_N0*Y2s + lprime * (X20 + X2c) * torsion + X1c*curvature*d_Y1c_d_varphi +\
        d_Y20_d_varphi + d_Y2c_d_varphi) + Y1c * (iota_N0*Y20 + 3*iota_N0*Y2c - lprime*X2s*torsion + X1c*curvature*d_Y1s_d_varphi - \
        d_Y2s_d_varphi)) + X1c * (lprime*X1c*Y20*torsion - lprime*X1c*Y2c*torsion + Y20*d_Y1c_d_varphi - Y2c*d_Y1c_d_varphi - \
        Y2s*d_Y1s_d_varphi + Y1c * (3*iota_N0*Y2s + lprime * (-X20 + X2c) * torsion - d_Y20_d_varphi + d_Y2c_d_varphi) + \
        Y1s * (iota_N0*Y20 - 3*iota_N0*Y2c - iota_N0*X1c*Y1c*curvature + lprime*X2s*torsion + X1c*curvature*d_Y1s_d_varphi + \
        d_Y2s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element213
    grad_grad_B[:,1,0,2]=(2*B0**5*lprime*lprime*(G0 * (2*B2s*lprime*X1s*Y1c + 2*B2s*lprime*X1c*Y1s + 2*B2c*lprime*(X1c*Y1c - \
        X1s*Y1s) - 2*B20*lprime*(X1c*Y1c + X1s*Y1s) + X1c*Y1c*Z20*d_B0_d_varphi + X1s*Y1s*Z20*d_B0_d_varphi - X1c*Y1c*Z2c*d_B0_d_varphi + \
        X1s*Y1s*Z2c*d_B0_d_varphi - X1s*Y1c*Z2s*d_B0_d_varphi - X1c*Y1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(G0*X1c*X1c * (Y20 - \
        Y2c) * curvature + X1c * (-(G0*(X2s*Y1s + 2*X1s*Y2s) * curvature) + Y1c*(G2 + I2*iota + G0*X20*curvature - G0*X2c*curvature)) + \
        X1s * (G0*(-(X2s*Y1c) + X1s * (Y20 + Y2c)) * curvature + Y1s*(G2 + I2*iota + G0*X20*curvature + G0*X2c*curvature))) - \
        G0 * (X1s * (Y1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) + Y1c*(2*iota_N0*Z2c - d_Z2s_d_varphi)) + \
        X1c * (Y1c*(-2*iota_N0*Z2s + d_Z20_d_varphi - d_Z2c_d_varphi) + Y1s*(2*iota_N0*Z2c - d_Z2s_d_varphi))))))/(Bbar**2*G0*G0*G0*G0)

    # Element221
    grad_grad_B[:,1,1,0]=(-2*B0**6*lprime*lprime*(X1c*X1c * (3*iota_N0*X2s + lprime * (Y20 - Y2c) * torsion - d_X20_d_varphi + \
        d_X2c_d_varphi) + X1s * (lprime*(X2s*Y1c - (X20 + X2c) * Y1s) * torsion - X2s*d_X1c_d_varphi + X20*d_X1s_d_varphi + \
        X2c*d_X1s_d_varphi - X1s * (3*iota_N0*X2s - lprime * (Y20 + Y2c) * torsion + d_X20_d_varphi + d_X2c_d_varphi)) + \
        X1c * (lprime*(-(X20*Y1c) + X2c*Y1c + X2s*Y1s) * torsion + X20*d_X1c_d_varphi - X2c*d_X1c_d_varphi - X2s*d_X1s_d_varphi - \
        2*X1s * (3*iota_N0*X2c + lprime*Y2s*torsion - d_X2s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element222
    grad_grad_B[:,1,1,1]=(2*B0**6*lprime*lprime*(X1s*X1s * (2*iota_N0*Y2s + d_Y20_d_varphi + d_Y2c_d_varphi) + X1c * (iota_N0*X2c*Y1s - \
        2*iota_N0*X1c*Y2s + X2c*d_Y1c_d_varphi - X20 * (iota_N0*Y1s + d_Y1c_d_varphi) + X2s * (-(iota_N0*Y1c) + d_Y1s_d_varphi) + \
        X1c*d_Y20_d_varphi - X1c*d_Y2c_d_varphi) + X1s * (iota_N0*X2s*Y1s + 4*iota_N0*X1c*Y2c + X2s*d_Y1c_d_varphi + X20 * (iota_N0*Y1c - \
        d_Y1s_d_varphi) + X2c * (iota_N0*Y1c - d_Y1s_d_varphi) - 2*X1c*d_Y2s_d_varphi)))/(Bbar**2*G0*G0*G0)

    # Element223
    grad_grad_B[:,1,1,2]=(-2*B0**5*lprime*lprime*(G0 * (4*B2s*lprime*X1c*X1s + 2*B2c*lprime*(X1c*X1c - X1s*X1s) - \
        2*B20*lprime*(X1c*X1c + X1s*X1s) + X1c*X1c*Z20*d_B0_d_varphi + X1s*X1s*Z20*d_B0_d_varphi - X1c*X1c*Z2c*d_B0_d_varphi + \
        X1s*X1s*Z2c*d_B0_d_varphi - 2*X1c*X1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(-4*G0*X1c*X1s*X2s*curvature + X1c*X1c*(G2 + \
        I2*iota + 2*G0*X20*curvature - 2*G0*X2c*curvature) + X1s*X1s*(G2 + I2*iota + 2*G0*X20*curvature + 2*G0*X2c*curvature)) + \
        G0 * (X1c*X1c*(2*iota_N0*Z2s - d_Z20_d_varphi + d_Z2c_d_varphi) - X1s*X1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) +\
        2*X1c*X1s*(-2*iota_N0*Z2c + d_Z2s_d_varphi)))))/(Bbar**2*G0*G0*G0*G0)

    # Element231
    grad_grad_B[:,1,2,0]=(B0**5*lprime*(-((Bbar*sG*d_B0_d_varphi*(2*iota_N0*X1c*X1c + X1s*(2*iota_N0*X1s - lprime*Y1c*torsion + \
        2*d_X1c_d_varphi) + X1c * (lprime*Y1s*torsion - 2*d_X1s_d_varphi)))/B0) + B0*(2*lprime*lprime * (X1c*X1c*(Y20 - Y2c) +\
        X1s * (X2s*Y1c - X20*Y1s - X2c*Y1s + X1s*Y20 + X1s*Y2c) + X1c * (-(X20*Y1c) + X2c*Y1c + X2s*Y1s - 2*X1s*Y2s)) * curvature +\
        iota_N0*X1c*X1c*X1c*d_Y1s_d_varphi + (Bbar*sG*lprime*(torsion*(Y1s*d_X1c_d_varphi - Y1c*d_X1s_d_varphi) + \
        X1s*(2*iota_N0*Y1s*torsion + torsion*d_Y1c_d_varphi + Y1c*d_torsion_d_varphi) + X1c*(2*iota_N0*Y1c*torsion - \
        torsion*d_Y1s_d_varphi - Y1s*d_torsion_d_varphi)))/B0 + X1s*(d_X1c_d_varphi*(Y1s*d_X1c_d_varphi - Y1c*d_X1s_d_varphi) -\
        iota_N0*X1s*X1s*d_Y1c_d_varphi + X1s * (iota_N0*Y1s*d_X1c_d_varphi - d_X1c_d_varphi*d_Y1c_d_varphi + \
        Y1c*(iota_N0*d_X1s_d_varphi + d2_X1c_d_varphi2))) + X1c * (d_X1s_d_varphi*(-(Y1s*d_X1c_d_varphi) + Y1c*d_X1s_d_varphi) +\
        iota_N0*X1s*X1s*d_Y1s_d_varphi + X1s * (d_X1s_d_varphi*d_Y1c_d_varphi + d_X1c_d_varphi*d_Y1s_d_varphi - \
        Y1s*(2*iota_N0*d_X1s_d_varphi + d2_X1c_d_varphi2) + Y1c*(2*iota_N0*d_X1c_d_varphi - d2_X1s_d_varphi2))) - \
        X1c*X1c * (iota_N0*Y1c*d_X1s_d_varphi + iota_N0*X1s*d_Y1c_d_varphi + d_X1s_d_varphi*d_Y1s_d_varphi + \
        Y1s * (iota_N0*d_X1c_d_varphi - d2_X1s_d_varphi2)))))/(Bbar**2*G0*G0*G0)

    # Element232
    grad_grad_B[:,1,2,1]= - ((B0**5*lprime*((2*Bbar*sG*d_B0_d_varphi*(X1s * (iota_N0*Y1s + d_Y1c_d_varphi) + X1c * (iota_N0*Y1c -\
        d_Y1s_d_varphi)))/B0 + B0 * ((Bbar*sG*lprime*torsion*(iota_N0*X1c*X1c + iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s +\
        X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + Y1c*d_Y1s_d_varphi))/B0 + X1s*X1s*(iota_N0*Y1s*d_Y1c_d_varphi +\
        d_Y1c_d_varphi**2 - Y1c*(iota_N0*d_Y1s_d_varphi + d2_Y1c_d_varphi2)) + X1c * (iota_N0*Y1c*Y1c*d_X1s_d_varphi + \
        iota_N0*Y1s*Y1s*d_X1s_d_varphi - Y1c*(iota_N0*X1c + d_X1s_d_varphi) * d_Y1s_d_varphi + X1c*d_Y1s_d_varphi*d_Y1s_d_varphi +\
        Y1s*(iota_N0*X1c*d_Y1c_d_varphi + d_X1s_d_varphi*d_Y1c_d_varphi - X1c*d2_Y1s_d_varphi2)) + \
        X1s * (-(iota_N0*Y1c*Y1c*d_X1c_d_varphi) - iota_N0*Y1s*Y1s*d_X1c_d_varphi - 2*X1c*d_Y1c_d_varphi*d_Y1s_d_varphi + \
        Y1s*(-(d_X1c_d_varphi*d_Y1c_d_varphi) + X1c*d2_Y1c_d_varphi2) + Y1c*(d_X1c_d_varphi*d_Y1s_d_varphi + \
        X1c*d2_Y1s_d_varphi2)))))/(Bbar**2*G0*G0*G0))

    # Element233
    grad_grad_B[:,1,2,2]=(B0**5*lprime*lprime*(2 * (X1c*X1c * (Y20 - Y2c) + X1s * (X2s*Y1c - X20*Y1s - X2c*Y1s + X1s*Y20 + X1s*Y2c) + \
        X1c * (-(X20*Y1c) + X2c*Y1c + X2s*Y1s - 2*X1s*Y2s)) * d_B0_d_varphi + Bbar*sG*curvature*(iota_N0*X1c*X1c + \
        X1s * (iota_N0*X1s - 2*lprime*Y1c*torsion + d_X1c_d_varphi) + X1c * (2*lprime*Y1s*torsion - \
        d_X1s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element311
    grad_grad_B[:,2,0,0]=(sG*B0*B0*B0*B0*lprime*(3*d_B0_d_varphi*(iota_N0*X1c*Y1c + iota_N0*X1s*Y1s + Y1s*d_X1c_d_varphi - \
        Y1c*d_X1s_d_varphi) + B0 * ((2*Bbar*sG*lprime*lprime*curvature*curvature)/B0 + iota_N0*Y1c*d_X1c_d_varphi + \
        iota_N0*Y1s*d_X1s_d_varphi + iota_N0*X1c*d_Y1c_d_varphi - d_X1s_d_varphi*d_Y1c_d_varphi + iota_N0*X1s*d_Y1s_d_varphi +\
        d_X1c_d_varphi*d_Y1s_d_varphi + lprime*torsion*(iota_N0*X1c*X1c + iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s +\
        X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + Y1c*d_Y1s_d_varphi) + Y1s*d2_X1c_d_varphi2 - \
        Y1c*d2_X1s_d_varphi2)))/(Bbar*G0*G0*G0)

    # Element312
    grad_grad_B[:,2,0,1]=(sG*B0*B0*B0*B0*lprime*(d_B0_d_varphi*(3*iota_N0*Y1c*Y1c + Y1s * (3*iota_N0*Y1s + 2*lprime*X1c*torsion + \
        3*d_Y1c_d_varphi) - Y1c * (2*lprime*X1s*torsion + 3*d_Y1s_d_varphi)) + B0 * (lprime*(2*iota_N0*X1s*Y1s*torsion + \
        2*Y1s*torsion*d_X1c_d_varphi - 2*Y1c*torsion*d_X1s_d_varphi - X1s*Y1c*d_torsion_d_varphi + X1c * (2*iota_N0*Y1c*torsion +\
        Y1s*d_torsion_d_varphi)) + Y1s * (2*iota_N0*d_Y1s_d_varphi + d2_Y1c_d_varphi2) + Y1c * (2*iota_N0*d_Y1c_d_varphi -\
        d2_Y1s_d_varphi2))))/(Bbar*G0*G0*G0)

    # Element313
    grad_grad_B[:,2,0,2]= - ((sG*B0*B0*B0*B0*lprime*lprime*((-3*Bbar*sG*curvature*d_B0_d_varphi)/B0 + B0*(X1s * (iota_N0*Y1s*curvature +\
        curvature*d_Y1c_d_varphi + Y1c*d_curvature_d_varphi) + X1c * (iota_N0*Y1c*curvature - curvature*d_Y1s_d_varphi - \
        Y1s*d_curvature_d_varphi))))/(Bbar*G0*G0*G0))

    # Element321
    grad_grad_B[:,2,1,0]= - ((sG*B0*B0*B0*B0*lprime*(d_B0_d_varphi*(3*iota_N0*X1c*X1c + X1s * (3*iota_N0*X1s - 2*lprime*Y1c*torsion + \
        3*d_X1c_d_varphi) + X1c * (2*lprime*Y1s*torsion - 3*d_X1s_d_varphi)) + B0 * (-(lprime*(X1s*(2*iota_N0*Y1s*torsion + \
        2*torsion*d_Y1c_d_varphi + Y1c*d_torsion_d_varphi) + X1c*(2*iota_N0*Y1c*torsion - 2*torsion*d_Y1s_d_varphi -\
        Y1s*d_torsion_d_varphi))) + X1s * (2*iota_N0*d_X1s_d_varphi + d2_X1c_d_varphi2) + X1c * (2*iota_N0*d_X1c_d_varphi -\
        d2_X1s_d_varphi2))))/(Bbar*G0*G0*G0))

    # Element322
    grad_grad_B[:,2,1,1]= - ((sG*B0*B0*B0*B0*lprime*(3*d_B0_d_varphi*(X1s * (iota_N0*Y1s + d_Y1c_d_varphi) + X1c * (iota_N0*Y1c - \
        d_Y1s_d_varphi)) + B0 * (iota_N0*Y1c*d_X1c_d_varphi + iota_N0*Y1s*d_X1s_d_varphi + iota_N0*X1c*d_Y1c_d_varphi + \
        d_X1s_d_varphi*d_Y1c_d_varphi + iota_N0*X1s*d_Y1s_d_varphi - d_X1c_d_varphi*d_Y1s_d_varphi + lprime*torsion*(iota_N0*X1c*X1c + \
        iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s + X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + \
        Y1c*d_Y1s_d_varphi) + X1s*d2_Y1c_d_varphi2 - X1c*d2_Y1s_d_varphi2)))/(Bbar*G0*G0*G0))

    # Element323
    grad_grad_B[:,2,1,2]=(sG*B0**5*lprime*lprime*curvature*(iota_N0*X1c*X1c + X1s * (iota_N0*X1s - 2*lprime*Y1c*torsion + \
        d_X1c_d_varphi) + X1c * (2*lprime*Y1s*torsion - d_X1s_d_varphi)))/(Bbar*G0*G0*G0)

    # Element331
    grad_grad_B[:,2,2,0]= - ((sG*B0*B0*B0*B0*lprime*lprime*((-3*Bbar*sG*curvature*d_B0_d_varphi)/B0 + B0*(X1s * (iota_N0*Y1s*curvature +\
        curvature*d_Y1c_d_varphi + Y1c*d_curvature_d_varphi) + X1c * (iota_N0*Y1c*curvature - curvature*d_Y1s_d_varphi - \
        Y1s*d_curvature_d_varphi))))/(Bbar*G0*G0*G0))

    # Element332
    grad_grad_B[:,2,2,1]= - ((sG*B0**5*lprime*lprime*curvature*(iota_N0*Y1c*Y1c + Y1s * (iota_N0*Y1s + d_Y1c_d_varphi) - \
        Y1c*d_Y1s_d_varphi))/(Bbar*G0*G0*G0))

    # Element333
    grad_grad_B[:,2,2,2]=(sG*B0*B0*B0*lprime*(-2*Bbar*sG*B0*lprime*lprime*curvature*curvature + (2*Bbar*sG*d_B0_d_varphi**2)/B0 +\
        B0 * (d_B0_d_varphi*(-(X1s*d_Y1c_d_varphi) + X1c*d_Y1s_d_varphi) + Y1s * (d_B0_d_varphi*d_X1c_d_varphi + X1c*d2_B0_d_varphi2) -\
        Y1c * (d_B0_d_varphi*d_X1s_d_varphi + X1s*d2_B0_d_varphi2))))/(Bbar*G0*G0*G0)





    self.grad_grad_B = grad_grad_B

    # Compute the (inverse) scale length
    squared = grad_grad_B * grad_grad_B
    norm_squared = np.sum(squared, axis=(1,2,3))
    self.grad_grad_B_inverse_scale_length_vs_varphi = np.sqrt(np.sqrt(norm_squared) / (4*B0))
    self.L_grad_grad_B = 1 / self.grad_grad_B_inverse_scale_length_vs_varphi
    self.grad_grad_B_inverse_scale_length = np.max(self.grad_grad_B_inverse_scale_length_vs_varphi)

    if not two_ways:
        return

    # Build the whole tensor again using Rogerio's approach,
    # "20200424-01 Rogerio's GradGradB calculation.nb"
    # and verify the two calculations match.

    # Element111
    grad_grad_B_alt[:,0,0,0]=(B0**6*lprime*lprime*(8*iota_N0*X2c*Y1c*Y1s + 4*iota_N0*X2s*(-Y1c*Y1c + Y1s*Y1s) - \
        2*iota_N0*X1s*Y1c*Y20 + 2*iota_N0*X1c*Y1s*Y20 + 2*iota_N0*X1s*Y1c*Y2c + 2*iota_N0*X1c*Y1s*Y2c - 2*iota_N0*X1c*Y1c*Y2s + \
        2*iota_N0*X1s*Y1s*Y2s - 5*iota_N0*X1c*X1s*Y1c*Y1c*curvature + 5*iota_N0*X1c*X1c*Y1c*Y1s*curvature - \
        5*iota_N0*X1s*X1s*Y1c*Y1s*curvature + 5*iota_N0*X1c*X1s*Y1s*Y1s*curvature - 2*Y1c*Y20*d_X1c_d_varphi + \
        2*Y1c*Y2c*d_X1c_d_varphi + 2*Y1s*Y2s*d_X1c_d_varphi - 5*X1s*Y1c*Y1s*curvature*d_X1c_d_varphi + \
        5*X1c*Y1s*Y1s*curvature*d_X1c_d_varphi - 2*Y1s*Y20*d_X1s_d_varphi - 2*Y1s*Y2c*d_X1s_d_varphi + 2*Y1c*Y2s*d_X1s_d_varphi + \
        5*X1s*Y1c*Y1c*curvature*d_X1s_d_varphi - 5*X1c*Y1c*Y1s*curvature*d_X1s_d_varphi + 2*Y1c*Y1c*d_X20_d_varphi + \
        2*Y1s*Y1s*d_X20_d_varphi - 2*Y1c*Y1c*d_X2c_d_varphi + 2*Y1s*Y1s*d_X2c_d_varphi - 4*Y1c*Y1s*d_X2s_d_varphi))/(Bbar**2*G0*G0*G0)

    # Element112
    grad_grad_B_alt[:,0,0,1]=(B0**6*lprime*lprime*(-5*iota_N0*X1s*Y1c*Y1c*Y1c*curvature + Y1c*Y1c * (-6*iota_N0*Y2s + \
        5*iota_N0*X1c*Y1s*curvature + 2*lprime*X20*torsion - 2*lprime*X2c*torsion + 5*lprime*X1s*X1s*curvature*torsion + \
        5*X1s*curvature*d_Y1s_d_varphi + 2*d_Y20_d_varphi - 2*d_Y2c_d_varphi) + Y1s * (5*iota_N0*X1c*Y1s*Y1s*curvature - \
        2*(lprime * (X1s*(Y20 + Y2c) - X1c*Y2s) * torsion - Y2s*d_Y1c_d_varphi + (Y20 + Y2c) * d_Y1s_d_varphi) + \
        Y1s * (6*iota_N0*Y2s + lprime * (2*X20 + 2*X2c + 5*X1c*X1c*curvature) * torsion + 5*X1c*curvature*d_Y1c_d_varphi + \
        2*d_Y20_d_varphi + 2*d_Y2c_d_varphi)) - Y1c * (5*iota_N0*X1s*Y1s*Y1s*curvature + 2*lprime * (X1c*(Y20 - Y2c) - \
        X1s*Y2s) * torsion + 2*Y20*d_Y1c_d_varphi - 2*Y2c*d_Y1c_d_varphi - 2*Y2s*d_Y1s_d_varphi + Y1s * (-12*iota_N0*Y2c + \
        2*lprime * (2*X2s + 5*X1c*X1s*curvature) * torsion + 5*X1s*curvature*d_Y1c_d_varphi + 5*X1c*curvature*d_Y1s_d_varphi + \
        4*d_Y2s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element113
    grad_grad_B_alt[:,0,0,2]= - ((B0**5*lprime*lprime*(2*G0 * (4*B2s*lprime*Y1c*Y1s + 2*B2c*lprime*(Y1c*Y1c - Y1s*Y1s) - \
        2*B20*lprime*(Y1c*Y1c + Y1s*Y1s) + Y1c*Y1c*Z20*d_B0_d_varphi + Y1s*Y1s*Z20*d_B0_d_varphi - Y1c*Y1c*Z2c*d_B0_d_varphi + \
        Y1s*Y1s*Z2c*d_B0_d_varphi - 2*Y1c*Y1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(Y1c*Y1c*(2*G2 + 2*I2*iota + 2*G0*X20*curvature - \
        2*G0*X2c*curvature + G0*X1s*X1s*curvature*curvature) - 2*G0*Y1c*curvature*(2*X2s*Y1s + X1s*Y2s + X1c*(-Y20 + Y2c + \
        X1s*Y1s*curvature)) + Y1s * (2*G0 * (X1s*(Y20 + Y2c) - X1c*Y2s) * curvature + Y1s*(2*G2 + 2*I2*iota + 2*G0*X20*curvature +\
        2*G0*X2c*curvature + G0*X1c*X1c*curvature*curvature))) + 2*G0 * (Y1c*Y1c*(2*iota_N0*Z2s - d_Z20_d_varphi + d_Z2c_d_varphi) - \
        Y1s*Y1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) + 2*Y1c*Y1s*(-2*iota_N0*Z2c + d_Z2s_d_varphi)))))/(Bbar**2*G0*G0*G0*G0))

    # Element121
    grad_grad_B_alt[:,0,1,0]= - ((B0**6*lprime*lprime*(-3*iota_N0*X1s*X1s*X1s*Y1c*curvature + 3*iota_N0*X1c*X1c*X1c*Y1s*curvature + \
        3*X1s*X1s*curvature*(iota_N0*X1c*Y1s + Y1c * (lprime*Y1c*torsion - d_X1c_d_varphi)) + 3*X1c*X1c*Y1s*curvature*(lprime*Y1s*torsion -\
        d_X1s_d_varphi) + 2 * (lprime * (-2*X2s*Y1c*Y1s + X2c*(-Y1c*Y1c + Y1s*Y1s) + X20 * (Y1c*Y1c + Y1s*Y1s)) * torsion + \
        X2c*Y1c*d_X1c_d_varphi + X2s*Y1s*d_X1c_d_varphi + X2s*Y1c*d_X1s_d_varphi - X2c*Y1s*d_X1s_d_varphi - X20 * (Y1c*d_X1c_d_varphi + \
        Y1s*d_X1s_d_varphi)) + X1s * (-2*iota_N0*X20*Y1c + 6*iota_N0*X2c*Y1c + 6*iota_N0*X2s*Y1s - 3*iota_N0*X1c*X1c*Y1c*curvature - \
        2*lprime*Y1s*Y20*torsion - 2*lprime*Y1s*Y2c*torsion + 2*lprime*Y1c*Y2s*torsion - 6*lprime*X1c*Y1c*Y1s*curvature*torsion + \
        3*X1c*Y1s*curvature*d_X1c_d_varphi + 3*X1c*Y1c*curvature*d_X1s_d_varphi + 2*Y1s*d_X20_d_varphi + 2*Y1s*d_X2c_d_varphi - \
        2*Y1c*d_X2s_d_varphi) - 2*X1c * (3*iota_N0*X2s*Y1c - iota_N0*X20*Y1s - 3*iota_N0*X2c*Y1s + lprime*Y1c*Y20*torsion - \
        lprime*Y1c*Y2c*torsion - lprime*Y1s*Y2s*torsion - Y1c*d_X20_d_varphi + Y1c*d_X2c_d_varphi + \
        Y1s*d_X2s_d_varphi)))/(Bbar**2*G0*G0*G0))

    # Element122
    grad_grad_B_alt[:,0,1,1]=(B0**6*lprime*lprime*(-4*iota_N0*X1s*Y1c*Y2c - 4*iota_N0*X1c*Y1s*Y2c + 4*iota_N0*X1c*Y1c*Y2s - \
        4*iota_N0*X1s*Y1s*Y2s + 3*iota_N0*X1c*X1s*Y1c*Y1c*curvature - 3*iota_N0*X1c*X1c*Y1c*Y1s*curvature + \
        3*iota_N0*X1s*X1s*Y1c*Y1s*curvature - 3*iota_N0*X1c*X1s*Y1s*Y1s*curvature + 2*X20*Y1c*d_Y1c_d_varphi + \
        3*X1s*X1s*Y1c*curvature*d_Y1c_d_varphi - 3*X1c*X1s*Y1s*curvature*d_Y1c_d_varphi - 2*X2c*Y1c*(2*iota_N0*Y1s + \
        d_Y1c_d_varphi) + 2*X20*Y1s*d_Y1s_d_varphi + 2*X2c*Y1s*d_Y1s_d_varphi - 3*X1c*X1s*Y1c*curvature*d_Y1s_d_varphi + \
        3*X1c*X1c*Y1s*curvature*d_Y1s_d_varphi + 2*X2s * (iota_N0*Y1c*Y1c - Y1s * (iota_N0*Y1s + d_Y1c_d_varphi) - \
        Y1c*d_Y1s_d_varphi) - 2*X1c*Y1c*d_Y20_d_varphi - 2*X1s*Y1s*d_Y20_d_varphi + 2*X1c*Y1c*d_Y2c_d_varphi - 2*X1s*Y1s*d_Y2c_d_varphi +\
        2*X1s*Y1c*d_Y2s_d_varphi + 2*X1c*Y1s*d_Y2s_d_varphi))/(Bbar**2*G0*G0*G0)

    # Element123
    grad_grad_B_alt[:,0,1,2]=(2*B0**5*lprime*lprime*(G0 * (2*B2s*lprime*X1s*Y1c + 2*B2s*lprime*X1c*Y1s + 2*B2c*lprime*(X1c*Y1c - X1s*Y1s) -\
        2*B20*lprime*(X1c*Y1c + X1s*Y1s) + X1c*Y1c*Z20*d_B0_d_varphi + X1s*Y1s*Z20*d_B0_d_varphi - X1c*Y1c*Z2c*d_B0_d_varphi +\
        X1s*Y1s*Z2c*d_B0_d_varphi - X1s*Y1c*Z2s*d_B0_d_varphi - X1c*Y1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(X1c * (-2*G0*X2s*Y1s*curvature + \
        Y1c*(G2 + I2*iota + 2*G0*X20*curvature - 2*G0*X2c*curvature)) + X1s * (-2*G0*X2s*Y1c*curvature + Y1s*(G2 + I2*iota + \
        2*G0*X20*curvature + 2*G0*X2c*curvature))) - G0 * (X1s * (Y1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) + \
        Y1c*(2*iota_N0*Z2c - d_Z2s_d_varphi)) + X1c * (Y1c*(-2*iota_N0*Z2s + d_Z20_d_varphi - d_Z2c_d_varphi) + Y1s*(2*iota_N0*Z2c - \
        d_Z2s_d_varphi))))))/(Bbar**2*G0*G0*G0*G0)

    # Element131
    grad_grad_B_alt[:,0,2,0]=(B0*B0*B0*B0*lprime*(2*Bbar*sG*d_B0_d_varphi*(iota_N0*X1c*Y1c + iota_N0*X1s*Y1s + Y1s*d_X1c_d_varphi - \
    Y1c*d_X1s_d_varphi) + Bbar*sG*B0*lprime*torsion*(iota_N0*X1c*X1c + iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s + \
    X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + Y1c*d_Y1s_d_varphi) + B0**2 * (lprime*lprime*curvature*(-2*X2c*Y1c*Y1c -\
    4*X2s*Y1c*Y1s + 2*X2c*Y1s*Y1s + 2*X20*(Y1c*Y1c + Y1s*Y1s) - 2*X1c*Y1c*Y20 - 2*X1s*Y1s*Y20 + 2*X1c*Y1c*Y2c - 2*X1s*Y1s*Y2c + \
    2*X1s*Y1c*Y2s + 2*X1c*Y1s*Y2s + 3*X1s*X1s*Y1c*Y1c*curvature - 6*X1c*X1s*Y1c*Y1s*curvature + 3*X1c*X1c*Y1s*Y1s*curvature) - \
    Y1s*Y1s*d_X1c_d_varphi*d_X1c_d_varphi + iota_N0*X1c*Y1c*Y1c*d_X1s_d_varphi + iota_N0*X1c*Y1s*Y1s*d_X1s_d_varphi + \
    2*Y1c*Y1s*d_X1c_d_varphi*d_X1s_d_varphi - Y1c*Y1c*d_X1s_d_varphi**2 + iota_N0*X1c*X1c*Y1s*d_Y1c_d_varphi - \
    X1c*Y1s*d_X1s_d_varphi*d_Y1c_d_varphi - iota_N0*X1c*X1c*Y1c*d_Y1s_d_varphi + X1c*Y1c*d_X1s_d_varphi*d_Y1s_d_varphi + \
    iota_N0*X1s*X1s*(Y1s*d_Y1c_d_varphi - Y1c*d_Y1s_d_varphi) + X1c*Y1s*Y1s*d2_X1c_d_varphi2 - X1s * (Y1s*d_X1c_d_varphi*(iota_N0*Y1s - \
    d_Y1c_d_varphi) + Y1c * (d_X1c_d_varphi*d_Y1s_d_varphi + Y1s*d2_X1c_d_varphi2) + Y1c*Y1c*(iota_N0*d_X1c_d_varphi - \
    d2_X1s_d_varphi2)) - X1c*Y1c*Y1s*d2_X1s_d_varphi2)))/(Bbar**2*G0*G0*G0)

    # Element132
    grad_grad_B_alt[:,0,2,1]=(B0*B0*B0*B0*lprime*(Bbar*sG*d_B0_d_varphi*(2*iota_N0*Y1c*Y1c + Y1s * (2*iota_N0*Y1s + \
        lprime*X1c*torsion + 2*d_Y1c_d_varphi) - Y1c * (lprime*X1s*torsion + 2*d_Y1s_d_varphi)) + \
        Bbar*sG*B0*lprime*(torsion * (Y1s*d_X1c_d_varphi - Y1c*d_X1s_d_varphi) + X1s * (2*iota_N0*Y1s*torsion + \
        torsion*d_Y1c_d_varphi - Y1c*d_torsion_d_varphi) + X1c * (2*iota_N0*Y1c*torsion - torsion*d_Y1s_d_varphi + \
        Y1s*d_torsion_d_varphi)) + B0**2 * (iota_N0*Y1c*Y1c*Y1c*d_X1s_d_varphi + Y1s * (-(iota_N0*Y1s*Y1s*d_X1c_d_varphi) + \
        d_Y1c_d_varphi*(X1s*d_Y1c_d_varphi - X1c*d_Y1s_d_varphi) + Y1s * (iota_N0*X1s*d_Y1c_d_varphi - d_X1c_d_varphi*d_Y1c_d_varphi + \
        X1c*(iota_N0*d_Y1s_d_varphi + d2_Y1c_d_varphi2))) + Y1c * (iota_N0*Y1s*Y1s*d_X1s_d_varphi + d_Y1s_d_varphi*(-(X1s*d_Y1c_d_varphi) +\
        X1c*d_Y1s_d_varphi) + Y1s * (d_X1s_d_varphi*d_Y1c_d_varphi - 2*iota_N0*X1s*d_Y1s_d_varphi + d_X1c_d_varphi*d_Y1s_d_varphi - \
        X1s*d2_Y1c_d_varphi2 + X1c*(2*iota_N0*d_Y1c_d_varphi - d2_Y1s_d_varphi2))) - Y1c*Y1c * (iota_N0*Y1s*d_X1c_d_varphi + (iota_N0*X1c + \
        d_X1s_d_varphi) * d_Y1s_d_varphi + X1s * (iota_N0*d_Y1c_d_varphi - d2_Y1s_d_varphi2)))))/(Bbar**2*G0*G0*G0)

    # Element133
    grad_grad_B_alt[:,0,2,2]=(B0**5*lprime*lprime*((-4*X2s*Y1c*Y1s - 2*X2c * (Y1c*Y1c - Y1s*Y1s) + 2*X20 * (Y1c*Y1c + Y1s*Y1s) - \
        2*X1c*Y1c*Y20 - 2*X1s*Y1s*Y20 + 2*X1c*Y1c*Y2c - 2*X1s*Y1s*Y2c + 2*X1s*Y1c*Y2s + 2*X1c*Y1s*Y2s + 3*X1s*X1s*Y1c*Y1c*curvature - \
        6*X1c*X1s*Y1c*Y1s*curvature + 3*X1c*X1c*Y1s*Y1s*curvature) * d_B0_d_varphi - Bbar*sG * (curvature*(Y1s*d_X1c_d_varphi - \
        Y1c*d_X1s_d_varphi) + X1s * (iota_N0*Y1s*curvature + Y1c*d_curvature_d_varphi) + X1c * (iota_N0*Y1c*curvature - \
        Y1s*d_curvature_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element211
    grad_grad_B_alt[:,1,0,0]=(-2*B0**6*lprime*lprime*(-(iota_N0*X1s*X1s*X1s*Y1c*curvature) + X1s*X1s * (iota_N0*Y2s + curvature*(iota_N0*X1c*Y1s + \
        lprime*Y1c*Y1c*torsion - Y1c*d_X1c_d_varphi)) + X1s * (2*iota_N0*X2c*Y1c + 2*iota_N0*X2s*Y1s + 2*iota_N0*X1c*Y2c - \
        iota_N0*X1c*X1c*Y1c*curvature - 2*lprime*X1c*Y1c*Y1s*curvature*torsion + Y2s*d_X1c_d_varphi + X1c*Y1s*curvature*d_X1c_d_varphi - \
        Y20*d_X1s_d_varphi - Y2c*d_X1s_d_varphi + X1c*Y1c*curvature*d_X1s_d_varphi + Y1s*d_X20_d_varphi + Y1s*d_X2c_d_varphi - \
        Y1c*d_X2s_d_varphi) + X1c * (-2*iota_N0*X2s*Y1c + 2*iota_N0*X2c*Y1s - iota_N0*X1c*Y2s + iota_N0*X1c*X1c*Y1s*curvature + \
        lprime*X1c*Y1s*Y1s*curvature*torsion - Y20*d_X1c_d_varphi + Y2c*d_X1c_d_varphi + Y2s*d_X1s_d_varphi - X1c*Y1s*curvature*d_X1s_d_varphi +\
        Y1c*d_X20_d_varphi - Y1c*d_X2c_d_varphi - Y1s*d_X2s_d_varphi)))/(Bbar**2*G0*G0*G0)

    # Element212
    grad_grad_B_alt[:,1,0,1]=(2*B0**6*lprime*lprime*(X1s*X1s * (lprime*(Y20 + Y2c) * torsion + Y1c*curvature*(iota_N0*Y1s + d_Y1c_d_varphi)) - \
        X1s * (-(iota_N0*X1c*Y1c*Y1c*curvature) + iota_N0*X1c*Y1s*Y1s*curvature + 2*lprime*X1c*Y2s*torsion + Y2s*d_Y1c_d_varphi -\
        Y20*d_Y1s_d_varphi - Y2c*d_Y1s_d_varphi + Y1s * (3*iota_N0*Y2s + lprime * (X20 + X2c) * torsion + X1c*curvature*d_Y1c_d_varphi +\
        d_Y20_d_varphi + d_Y2c_d_varphi) + Y1c * (iota_N0*Y20 + 3*iota_N0*Y2c - lprime*X2s*torsion + X1c*curvature*d_Y1s_d_varphi - \
        d_Y2s_d_varphi)) + X1c * (lprime*X1c*Y20*torsion - lprime*X1c*Y2c*torsion + Y20*d_Y1c_d_varphi - Y2c*d_Y1c_d_varphi - \
        Y2s*d_Y1s_d_varphi + Y1c * (3*iota_N0*Y2s + lprime * (-X20 + X2c) * torsion - d_Y20_d_varphi + d_Y2c_d_varphi) + \
        Y1s * (iota_N0*Y20 - 3*iota_N0*Y2c - iota_N0*X1c*Y1c*curvature + lprime*X2s*torsion + X1c*curvature*d_Y1s_d_varphi +\
        d_Y2s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element213
    grad_grad_B_alt[:,1,0,2]=(2*B0**5*lprime*lprime*(G0 * (2*B2s*lprime*X1s*Y1c + 2*B2s*lprime*X1c*Y1s + 2*B2c*lprime*(X1c*Y1c - X1s*Y1s) - \
        2*B20*lprime*(X1c*Y1c + X1s*Y1s) + X1c*Y1c*Z20*d_B0_d_varphi + X1s*Y1s*Z20*d_B0_d_varphi - X1c*Y1c*Z2c*d_B0_d_varphi +\
        X1s*Y1s*Z2c*d_B0_d_varphi - X1s*Y1c*Z2s*d_B0_d_varphi - X1c*Y1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(G0*X1c*X1c * (Y20 - \
        Y2c) * curvature + X1c * (-(G0*(X2s*Y1s + 2*X1s*Y2s) * curvature) + Y1c*(G2 + I2*iota + G0*X20*curvature - G0*X2c*curvature)) + \
        X1s * (G0*(-(X2s*Y1c) + X1s * (Y20 + Y2c)) * curvature + Y1s*(G2 + I2*iota + G0*X20*curvature + G0*X2c*curvature))) - \
        G0 * (X1s * (Y1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) + Y1c*(2*iota_N0*Z2c - d_Z2s_d_varphi)) + \
        X1c * (Y1c*(-2*iota_N0*Z2s + d_Z20_d_varphi - d_Z2c_d_varphi) + Y1s*(2*iota_N0*Z2c - \
        d_Z2s_d_varphi))))))/(Bbar**2*G0*G0*G0*G0)

    # Element221
    grad_grad_B_alt[:,1,1,0]=(-2*B0**6*lprime*lprime*(X1c*X1c * (3*iota_N0*X2s + lprime * (Y20 - Y2c) * torsion - d_X20_d_varphi + \
        d_X2c_d_varphi) + X1s * (lprime*(X2s*Y1c - (X20 + X2c) * Y1s) * torsion - X2s*d_X1c_d_varphi + X20*d_X1s_d_varphi + \
        X2c*d_X1s_d_varphi - X1s * (3*iota_N0*X2s - lprime * (Y20 + Y2c) * torsion + d_X20_d_varphi + d_X2c_d_varphi)) + \
        X1c * (lprime*(-(X20*Y1c) + X2c*Y1c + X2s*Y1s) * torsion + X20*d_X1c_d_varphi - X2c*d_X1c_d_varphi - X2s*d_X1s_d_varphi - \
        2*X1s * (3*iota_N0*X2c + lprime*Y2s*torsion - d_X2s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element222
    grad_grad_B_alt[:,1,1,1]=(2*B0**6*lprime*lprime*(X1s*X1s * (2*iota_N0*Y2s + d_Y20_d_varphi + d_Y2c_d_varphi) + \
        X1c * (iota_N0*X2c*Y1s - 2*iota_N0*X1c*Y2s + X2c*d_Y1c_d_varphi - X20 * (iota_N0*Y1s + d_Y1c_d_varphi) + \
        X2s * (-(iota_N0*Y1c) + d_Y1s_d_varphi) + X1c*d_Y20_d_varphi - X1c*d_Y2c_d_varphi) + X1s * (iota_N0*X2s*Y1s + \
        4*iota_N0*X1c*Y2c + X2s*d_Y1c_d_varphi + X20 * (iota_N0*Y1c - d_Y1s_d_varphi) + X2c * (iota_N0*Y1c - d_Y1s_d_varphi) - \
        2*X1c*d_Y2s_d_varphi)))/(Bbar**2*G0*G0*G0)

    # Element223
    grad_grad_B_alt[:,1,1,2]=(-2*B0**5*lprime*lprime*(G0 * (4*B2s*lprime*X1c*X1s + 2*B2c*lprime*(X1c*X1c - X1s*X1s) - \
        2*B20*lprime*(X1c*X1c + X1s*X1s) + X1c*X1c*Z20*d_B0_d_varphi + X1s*X1s*Z20*d_B0_d_varphi - X1c*X1c*Z2c*d_B0_d_varphi + \
        X1s*X1s*Z2c*d_B0_d_varphi - 2*X1c*X1s*Z2s*d_B0_d_varphi) + B0 * (lprime*(-4*G0*X1c*X1s*X2s*curvature + X1c*X1c*(G2 + \
        I2*iota + 2*G0*X20*curvature - 2*G0*X2c*curvature) + X1s*X1s*(G2 + I2*iota + 2*G0*X20*curvature + 2*G0*X2c*curvature)) + \
        G0 * (X1c*X1c*(2*iota_N0*Z2s - d_Z20_d_varphi + d_Z2c_d_varphi) - X1s*X1s*(2*iota_N0*Z2s + d_Z20_d_varphi + d_Z2c_d_varphi) + \
        2*X1c*X1s*(-2*iota_N0*Z2c + d_Z2s_d_varphi)))))/(Bbar**2*G0*G0*G0*G0)

    # Element231
    grad_grad_B_alt[:,1,2,0]=(B0*B0*B0*B0*lprime*(-(Bbar*sG*d_B0_d_varphi*(2*iota_N0*X1c*X1c + X1s * (2*iota_N0*X1s - \
        lprime*Y1c*torsion + 2*d_X1c_d_varphi) + X1c * (lprime*Y1s*torsion - 2*d_X1s_d_varphi))) + \
        Bbar*sG*B0*lprime*(torsion * (Y1s*d_X1c_d_varphi - Y1c*d_X1s_d_varphi) + X1s * (2*iota_N0*Y1s*torsion + \
        torsion*d_Y1c_d_varphi + Y1c*d_torsion_d_varphi) + X1c * (2*iota_N0*Y1c*torsion - torsion*d_Y1s_d_varphi - \
        Y1s*d_torsion_d_varphi)) + B0**2 * (2*lprime*lprime*(X1c*X1c * (Y20 - Y2c) + X1s * (X2s*Y1c - X20*Y1s - X2c*Y1s + \
        X1s*Y20 + X1s*Y2c) + X1c * (-(X20*Y1c) + X2c*Y1c + X2s*Y1s - 2*X1s*Y2s)) * curvature + iota_N0*X1c*X1c*X1c*d_Y1s_d_varphi + \
        X1s * (d_X1c_d_varphi*(Y1s*d_X1c_d_varphi - Y1c*d_X1s_d_varphi) - iota_N0*X1s*X1s*d_Y1c_d_varphi + \
        X1s * (iota_N0*Y1s*d_X1c_d_varphi - d_X1c_d_varphi*d_Y1c_d_varphi + Y1c*(iota_N0*d_X1s_d_varphi + d2_X1c_d_varphi2))) + \
        X1c * (d_X1s_d_varphi*(-(Y1s*d_X1c_d_varphi) + Y1c*d_X1s_d_varphi) + iota_N0*X1s*X1s*d_Y1s_d_varphi + \
        X1s * (d_X1s_d_varphi*d_Y1c_d_varphi + d_X1c_d_varphi*d_Y1s_d_varphi - Y1s*(2*iota_N0*d_X1s_d_varphi + d2_X1c_d_varphi2) +\
        Y1c*(2*iota_N0*d_X1c_d_varphi - d2_X1s_d_varphi2))) - X1c*X1c * (iota_N0*Y1c*d_X1s_d_varphi + iota_N0*X1s*d_Y1c_d_varphi + \
        d_X1s_d_varphi*d_Y1s_d_varphi + Y1s * (iota_N0*d_X1c_d_varphi - d2_X1s_d_varphi2)))))/(Bbar**2*G0*G0*G0)

    # Element232
    grad_grad_B_alt[:,1,2,1]= - ((B0*B0*B0*B0*lprime*(2*Bbar*sG*d_B0_d_varphi*(X1s * (iota_N0*Y1s + d_Y1c_d_varphi) + X1c * (iota_N0*Y1c -\
        d_Y1s_d_varphi)) + Bbar*sG*B0*lprime*torsion*(iota_N0*X1c*X1c + iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s + \
        X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + Y1c*d_Y1s_d_varphi) + B0**2 * (X1s*X1s*(iota_N0*Y1s*d_Y1c_d_varphi + \
        d_Y1c_d_varphi**2 - Y1c*(iota_N0*d_Y1s_d_varphi + d2_Y1c_d_varphi2)) + X1c * (iota_N0*Y1c*Y1c*d_X1s_d_varphi + \
        iota_N0*Y1s*Y1s*d_X1s_d_varphi - Y1c*(iota_N0*X1c + d_X1s_d_varphi) * d_Y1s_d_varphi + X1c*d_Y1s_d_varphi*d_Y1s_d_varphi + \
        Y1s*(iota_N0*X1c*d_Y1c_d_varphi + d_X1s_d_varphi*d_Y1c_d_varphi - X1c*d2_Y1s_d_varphi2)) + X1s * (-(iota_N0*Y1c*Y1c*d_X1c_d_varphi) -\
        iota_N0*Y1s*Y1s*d_X1c_d_varphi - 2*X1c*d_Y1c_d_varphi*d_Y1s_d_varphi + Y1s*(-(d_X1c_d_varphi*d_Y1c_d_varphi) + \
        X1c*d2_Y1c_d_varphi2) + Y1c*(d_X1c_d_varphi*d_Y1s_d_varphi + X1c*d2_Y1s_d_varphi2)))))/(Bbar**2*G0*G0*G0))

    # Element233
    grad_grad_B_alt[:,1,2,2]=(B0**5*lprime*lprime*(2 * (X1c*X1c * (Y20 - Y2c) + X1s * (X2s*Y1c - X20*Y1s - X2c*Y1s + X1s*Y20 + X1s*Y2c) + \
        X1c * (-(X20*Y1c) + X2c*Y1c + X2s*Y1s - 2*X1s*Y2s)) * d_B0_d_varphi + Bbar*sG*curvature*(iota_N0*X1c*X1c + X1s * (iota_N0*X1s - \
        2*lprime*Y1c*torsion + d_X1c_d_varphi) + X1c * (2*lprime*Y1s*torsion - d_X1s_d_varphi))))/(Bbar**2*G0*G0*G0)

    # Element311
    grad_grad_B_alt[:,2,0,0]=(sG*B0*B0*B0*B0*lprime*(3*d_B0_d_varphi*(iota_N0*X1c*Y1c + iota_N0*X1s*Y1s + Y1s*d_X1c_d_varphi - \
        Y1c*d_X1s_d_varphi) + B0 * ((2*Bbar*sG*lprime*lprime*curvature*curvature)/B0 + iota_N0*Y1c*d_X1c_d_varphi + iota_N0*Y1s*d_X1s_d_varphi +\
        iota_N0*X1c*d_Y1c_d_varphi - d_X1s_d_varphi*d_Y1c_d_varphi + iota_N0*X1s*d_Y1s_d_varphi + d_X1c_d_varphi*d_Y1s_d_varphi +\
        lprime*torsion*(iota_N0*X1c*X1c + iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s + X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi -\
        Y1s*d_Y1c_d_varphi + Y1c*d_Y1s_d_varphi) + Y1s*d2_X1c_d_varphi2 - Y1c*d2_X1s_d_varphi2)))/(Bbar*G0*G0*G0)

    # Element312
    grad_grad_B_alt[:,2,0,1]=(sG*B0*B0*B0*B0*lprime*(d_B0_d_varphi*(3*iota_N0*Y1c*Y1c + Y1s * (3*iota_N0*Y1s + 2*lprime*X1c*torsion + \
        3*d_Y1c_d_varphi) - Y1c * (2*lprime*X1s*torsion + 3*d_Y1s_d_varphi)) + B0 * (lprime*(2*iota_N0*X1s*Y1s*torsion + \
        2*Y1s*torsion*d_X1c_d_varphi - 2*Y1c*torsion*d_X1s_d_varphi - X1s*Y1c*d_torsion_d_varphi + X1c * (2*iota_N0*Y1c*torsion + \
        Y1s*d_torsion_d_varphi)) + Y1s * (2*iota_N0*d_Y1s_d_varphi + d2_Y1c_d_varphi2) + Y1c * (2*iota_N0*d_Y1c_d_varphi - \
        d2_Y1s_d_varphi2))))/(Bbar*G0*G0*G0)

    # Element313
    grad_grad_B_alt[:,2,0,2]=(sG*B0*B0*B0*lprime*lprime*(3*Bbar*sG*curvature*d_B0_d_varphi - B0**2 * (X1s*(iota_N0*Y1s*curvature + \
        curvature*d_Y1c_d_varphi + Y1c*d_curvature_d_varphi) + X1c * (iota_N0*Y1c*curvature - curvature*d_Y1s_d_varphi - \
        Y1s*d_curvature_d_varphi))))/(Bbar*G0*G0*G0)

    # Element321
    grad_grad_B_alt[:,2,1,0]=(sG*B0*B0*B0*B0*lprime*(-(d_B0_d_varphi*(3*iota_N0*X1c*X1c + X1s * (3*iota_N0*X1s - 2*lprime*Y1c*torsion + \
        3*d_X1c_d_varphi) + X1c * (2*lprime*Y1s*torsion - 3*d_X1s_d_varphi))) + B0 * (lprime*(X1s * (2*iota_N0*Y1s*torsion + \
        2*torsion*d_Y1c_d_varphi + Y1c*d_torsion_d_varphi) + X1c * (2*iota_N0*Y1c*torsion - 2*torsion*d_Y1s_d_varphi - Y1s*d_torsion_d_varphi)) -\
        X1s * (2*iota_N0*d_X1s_d_varphi + d2_X1c_d_varphi2) + X1c * (-2*iota_N0*d_X1c_d_varphi + d2_X1s_d_varphi2))))/(Bbar*G0*G0*G0)

    # Element322
    grad_grad_B_alt[:,2,1,1]= - ((sG*B0*B0*B0*B0*lprime*(3*d_B0_d_varphi*(X1s * (iota_N0*Y1s + d_Y1c_d_varphi) + X1c * (iota_N0*Y1c -\
        d_Y1s_d_varphi)) + B0 * (iota_N0*Y1c*d_X1c_d_varphi + iota_N0*Y1s*d_X1s_d_varphi + iota_N0*X1c*d_Y1c_d_varphi + \
        d_X1s_d_varphi*d_Y1c_d_varphi + iota_N0*X1s*d_Y1s_d_varphi - d_X1c_d_varphi*d_Y1s_d_varphi + lprime*torsion*(iota_N0*X1c*X1c +\
        iota_N0*X1s*X1s - iota_N0*Y1c*Y1c - iota_N0*Y1s*Y1s + X1s*d_X1c_d_varphi - X1c*d_X1s_d_varphi - Y1s*d_Y1c_d_varphi + \
        Y1c*d_Y1s_d_varphi) + X1s*d2_Y1c_d_varphi2 - X1c*d2_Y1s_d_varphi2)))/(Bbar*G0*G0*G0))

    # Element323
    grad_grad_B_alt[:,2,1,2]=(sG*B0**5*lprime*lprime*curvature*(iota_N0*X1c*X1c + X1s * (iota_N0*X1s - 2*lprime*Y1c*torsion + \
        d_X1c_d_varphi) + X1c * (2*lprime*Y1s*torsion - d_X1s_d_varphi)))/(Bbar*G0*G0*G0)

    # Element331
    grad_grad_B_alt[:,2,2,0]=(sG*B0*B0*B0*lprime*lprime*(3*Bbar*sG*curvature*d_B0_d_varphi - B0**2 * (X1s*(iota_N0*Y1s*curvature +\
        curvature*d_Y1c_d_varphi + Y1c*d_curvature_d_varphi) + X1c * (iota_N0*Y1c*curvature - curvature*d_Y1s_d_varphi - \
        Y1s*d_curvature_d_varphi))))/(Bbar*G0*G0*G0)

    # Element332
    grad_grad_B_alt[:,2,2,1]= - ((sG*B0**5*lprime*lprime*curvature*(iota_N0*Y1c*Y1c + Y1s * (iota_N0*Y1s + d_Y1c_d_varphi) - \
        Y1c*d_Y1s_d_varphi))/(Bbar*G0*G0*G0))

    # Element333
    grad_grad_B_alt[:,2,2,2]= - ((sG*B0**2*lprime*(-2*Bbar*sG*d_B0_d_varphi**2 + B0**2 * (2*Bbar*sG*lprime*lprime*curvature*curvature + \
        Y1c*d_B0_d_varphi*d_X1s_d_varphi + X1s*d_B0_d_varphi*d_Y1c_d_varphi - X1c*d_B0_d_varphi*d_Y1s_d_varphi + \
        X1s*Y1c*d2_B0_d_varphi2 - Y1s * (d_B0_d_varphi*d_X1c_d_varphi + X1c*d2_B0_d_varphi2))))/(Bbar*G0*G0*G0))

    self.grad_grad_B_alt = grad_grad_B_alt


def Bfield_cylindrical(self, r=0, theta=0):
    '''
    Function to calculate the magnetic field vector B=(B_R,B_phi,B_Z) at
    every point along the axis (hence with nphi points) where R, phi and Z
    are the standard cylindrical coordinates for a given
    near-axis radius r and a Boozer poloidal angle vartheta (not theta).
    The formulae implemented here are eq (3.5) and (3.6) of
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    
    # Define auxiliary variables
    t = self.tangent_cylindrical.transpose()
    n = self.normal_cylindrical.transpose()
    b = self.binormal_cylindrical.transpose()
    B0 = self.B0
    sG = self.sG
    G0 = self.G0
    X1c = self.X1c
    X1s = self.X1s
    Y1c = self.Y1c
    Y1s = self.Y1s
    d_l_d_varphi = self.d_l_d_varphi
    curvature = self.curvature
    torsion = self.torsion
    iotaN = self.iotaN
    d_X1c_d_varphi = self.d_X1c_d_varphi
    d_X1s_d_varphi = self.d_X1s_d_varphi
    d_Y1s_d_varphi = self.d_Y1s_d_varphi
    d_Y1c_d_varphi = self.d_Y1c_d_varphi

    B0_vector = sG * B0 * t

    if r == 0:
        return B0_vector
    else:
        factor = B0 * B0 / G0
        B1_vector_t = factor * (X1c * np.cos(theta) + X1s * np.sin(theta)) * d_l_d_varphi * curvature
        B1_vector_n = factor * (np.cos(theta) * (d_X1c_d_varphi - Y1c * d_l_d_varphi * torsion + iotaN * X1s) \
                                + np.sin(theta) * (d_X1s_d_varphi - Y1s * d_l_d_varphi * torsion - iotaN * X1c))
        B1_vector_b = factor * (np.cos(theta) * (d_Y1c_d_varphi + X1c * d_l_d_varphi * torsion + iotaN * Y1s) \
                                + np.sin(theta) * (d_Y1s_d_varphi + X1s * d_l_d_varphi * torsion - iotaN * Y1c))

        B1_vector = B1_vector_t * t + B1_vector_n * n + B1_vector_b * b
        B_vector_cylindrical = B0_vector + r * B1_vector

        return B_vector_cylindrical

def Bfield_cartesian(self, r=0, theta=0):
    '''
    Function to calculate the magnetic field vector B=(B_x,B_y,B_z) at
    every point along the axis (hence with nphi points) where x, y and z
    are the standard cartesian coordinates for a given
    near-axis radius r and a Boozer poloidal angle vartheta (not theta).

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    B_vector_cylindrical = self.Bfield_cylindrical(r,theta)
    phi = self.phi

    B_x = np.cos(phi) * B_vector_cylindrical[0] - np.sin(phi) * B_vector_cylindrical[1]
    B_y = np.sin(phi) * B_vector_cylindrical[0] + np.cos(phi) * B_vector_cylindrical[1]
    B_z = B_vector_cylindrical[2]

    B_vector_cartesian = np.array([B_x, B_y, B_z])

    return B_vector_cartesian

def grad_B_tensor_cartesian(self):
    '''
    Function to calculate the gradient of the magnetic field vector B=(B_x,B_y,B_z)
    at every point along the axis (hence with nphi points) where x, y and z
    are the standard cartesian coordinates.
    '''

    B0, B1, B2 = self.Bfield_cylindrical()
    nablaB = self.grad_B_tensor_cylindrical
    cosphi = np.cos(self.phi)
    sinphi = np.sin(self.phi)
    R0 = self.R0

    grad_B_vector_cartesian = np.array([
[cosphi**2*nablaB[0, 0] - cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + 
   sinphi**2*nablaB[1, 1], cosphi**2*nablaB[0, 1] - sinphi**2*nablaB[1, 0] + 
   cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), cosphi*nablaB[0, 2] - 
   sinphi*nablaB[1, 2]], [-(sinphi**2*nablaB[0, 1]) + cosphi**2*nablaB[1, 0] + 
   cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), sinphi**2*nablaB[0, 0] + 
   cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + cosphi**2*nablaB[1, 1], 
  sinphi*nablaB[0, 2] + cosphi*nablaB[1, 2]], 
 [cosphi*nablaB[2, 0] - sinphi*nablaB[2, 1], sinphi*nablaB[2, 0] + cosphi*nablaB[2, 1], 
  nablaB[2, 2]]
    ])

    return grad_B_vector_cartesian

def grad_grad_B_tensor_cylindrical(self):
    '''
    Function to calculate the gradient of of the gradient the magnetic field
    vector B=(B_R,B_phi,B_Z) at every point along the axis (hence with nphi points)
    where R, phi and Z are the standard cylindrical coordinates.
    '''
    return np.transpose(self.grad_grad_B,(1,2,3,0))

def grad_grad_B_tensor_cartesian(self):
    '''
    Function to calculate the gradient of of the gradient the magnetic field
    vector B=(B_x,B_y,B_z) at every point along the axis (hence with nphi points)
    where x, y and z are the standard cartesian coordinates.
    '''
    nablanablaB = self.grad_grad_B_tensor_cylindrical()
    cosphi = np.cos(self.phi)
    sinphi = np.sin(self.phi)

    grad_grad_B_vector_cartesian = np.array([[
[cosphi**3*nablanablaB[0, 0, 0] - cosphi**2*sinphi*(nablanablaB[0, 0, 1] + 
      nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0]) + 
    cosphi*sinphi**2*(nablanablaB[0, 1, 1] + nablanablaB[1, 0, 1] + 
      nablanablaB[1, 1, 0]) - sinphi**3*nablanablaB[1, 1, 1], 
   cosphi**3*nablanablaB[0, 0, 1] + cosphi**2*sinphi*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 0, 1]) + sinphi**3*nablanablaB[1, 1, 0] - 
    cosphi*sinphi**2*(nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), cosphi**2*nablanablaB[0, 0, 2] - 
    cosphi*sinphi*(nablanablaB[0, 1, 2] + nablanablaB[1, 0, 2]) + 
    sinphi**2*nablanablaB[1, 1, 2]], [cosphi**3*nablanablaB[0, 1, 0] + 
    sinphi**3*nablanablaB[1, 0, 1] + cosphi**2*sinphi*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 1, 0]) - 
    cosphi*sinphi**2*(nablanablaB[0, 0, 1] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), cosphi**3*nablanablaB[0, 1, 1] - 
    sinphi**3*nablanablaB[1, 0, 0] + cosphi*sinphi**2*(nablanablaB[0, 0, 0] - 
      nablanablaB[1, 0, 1] - nablanablaB[1, 1, 0]) + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 1] + nablanablaB[0, 1, 0] - 
      nablanablaB[1, 1, 1]), cosphi**2*nablanablaB[0, 1, 2] - 
    sinphi**2*nablanablaB[1, 0, 2] + cosphi*sinphi*(nablanablaB[0, 0, 2] - 
      nablanablaB[1, 1, 2])], [cosphi**2*nablanablaB[0, 2, 0] - 
    cosphi*sinphi*(nablanablaB[0, 2, 1] + nablanablaB[1, 2, 0]) + 
    sinphi**2*nablanablaB[1, 2, 1], cosphi**2*nablanablaB[0, 2, 1] - 
    sinphi**2*nablanablaB[1, 2, 0] + cosphi*sinphi*(nablanablaB[0, 2, 0] - 
      nablanablaB[1, 2, 1]), cosphi*nablanablaB[0, 2, 2] - 
    sinphi*nablanablaB[1, 2, 2]]], 
 [[sinphi**3*nablanablaB[0, 1, 1] + cosphi**3*nablanablaB[1, 0, 0] + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 0] - nablanablaB[1, 0, 1] - 
      nablanablaB[1, 1, 0]) - cosphi*sinphi**2*(nablanablaB[0, 0, 1] + 
      nablanablaB[0, 1, 0] - nablanablaB[1, 1, 1]), -(sinphi**3*nablanablaB[0, 1, 0]) + 
    cosphi**3*nablanablaB[1, 0, 1] + cosphi*sinphi**2*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 1, 0]) + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 1] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), -(sinphi**2*nablanablaB[0, 1, 2]) + 
    cosphi**2*nablanablaB[1, 0, 2] + cosphi*sinphi*(nablanablaB[0, 0, 2] - 
      nablanablaB[1, 1, 2])], [-(sinphi**3*nablanablaB[0, 0, 1]) + 
    cosphi*sinphi**2*(nablanablaB[0, 0, 0] - nablanablaB[0, 1, 1] - 
      nablanablaB[1, 0, 1]) + cosphi**3*nablanablaB[1, 1, 0] + 
    cosphi**2*sinphi*(nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), sinphi**3*nablanablaB[0, 0, 0] + 
    cosphi*sinphi**2*(nablanablaB[0, 0, 1] + nablanablaB[0, 1, 0] + 
      nablanablaB[1, 0, 0]) + cosphi**2*sinphi*(nablanablaB[0, 1, 1] + 
      nablanablaB[1, 0, 1] + nablanablaB[1, 1, 0]) + cosphi**3*nablanablaB[1, 1, 1], 
   sinphi**2*nablanablaB[0, 0, 2] + cosphi*sinphi*(nablanablaB[0, 1, 2] + 
      nablanablaB[1, 0, 2]) + cosphi**2*nablanablaB[1, 1, 2]], 
  [-(sinphi**2*nablanablaB[0, 2, 1]) + cosphi**2*nablanablaB[1, 2, 0] + 
    cosphi*sinphi*(nablanablaB[0, 2, 0] - nablanablaB[1, 2, 1]), 
   sinphi**2*nablanablaB[0, 2, 0] + cosphi*sinphi*(nablanablaB[0, 2, 1] + 
      nablanablaB[1, 2, 0]) + cosphi**2*nablanablaB[1, 2, 1], 
   sinphi*nablanablaB[0, 2, 2] + cosphi*nablanablaB[1, 2, 2]]], 
 [[cosphi**2*nablanablaB[2, 0, 0] - cosphi*sinphi*(nablanablaB[2, 0, 1] + 
      nablanablaB[2, 1, 0]) + sinphi**2*nablanablaB[2, 1, 1], 
   cosphi**2*nablanablaB[2, 0, 1] - sinphi**2*nablanablaB[2, 1, 0] + 
    cosphi*sinphi*(nablanablaB[2, 0, 0] - nablanablaB[2, 1, 1]), 
   cosphi*nablanablaB[2, 0, 2] - sinphi*nablanablaB[2, 1, 2]], 
  [-(sinphi**2*nablanablaB[2, 0, 1]) + cosphi**2*nablanablaB[2, 1, 0] + 
    cosphi*sinphi*(nablanablaB[2, 0, 0] - nablanablaB[2, 1, 1]), 
   sinphi**2*nablanablaB[2, 0, 0] + cosphi*sinphi*(nablanablaB[2, 0, 1] + 
      nablanablaB[2, 1, 0]) + cosphi**2*nablanablaB[2, 1, 1], 
   sinphi*nablanablaB[2, 0, 2] + cosphi*nablanablaB[2, 1, 2]], 
  [cosphi*nablanablaB[2, 2, 0] - sinphi*nablanablaB[2, 2, 1], 
   sinphi*nablanablaB[2, 2, 0] + cosphi*nablanablaB[2, 2, 1], nablanablaB[2, 2, 2]]
      ]])

    return grad_grad_B_vector_cartesian
