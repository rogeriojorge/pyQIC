"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetic axis shape.
"""

import logging
import numpy as np
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import BSpline, make_interp_spline
from .spectral_diff_matrix import spectral_diff_matrix
from .util import fourier_minimum
from .fourier_interpolation import fourier_interpolation_matrix
from numpy import matlib as ml  

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(self,array):
    sp=spline(np.append(self.phi,2*np.pi/self.nfp+self.phi[0]), np.append(array,array[0]), bc_type='periodic')
    return sp

def convert_to_spline_tripled(self,array):
    phi_tripled = np.append(self.phi-(2*np.pi/self.nfp),self.phi)
    phi_tripled = np.append(phi_tripled,self.phi+(2*np.pi/self.nfp))
    sp=spline(np.append(phi_tripled,4*np.pi/self.nfp+self.phi[0]), np.append(array,array[0]), bc_type='periodic')
    return sp    

def init_axis(self):
    """
    Initialize the curvature, torsion, differentiation matrix, etc.
    """
    # Force curvature to be zero if omn using rc
    omn = self.omn
    rc  = self.rc
    rs  = self.rs
    nfp = self.nfp
    zs  = self.zs
    zc  = self.zc
    half_helicity = self.half_helicity #  False #self.half_helicity 
    if omn:
        ### Setting higher order rc to make sure kappa has first order zeros
        ###  at phi=0 and phi=pi/nfp
        if len(rc) == 3:
            rc[1] = 0.0
            rc[2] = -1 / (1.0 + 4 * nfp * nfp)
        elif len(rc) == 4:
            rc[2] = -1.0 / (1.0 + 4.0 * nfp * nfp)
            rc[3] = -( rc[1] +  rc[1] * nfp * nfp) / (1.0 + 9.0 * nfp * nfp)
        elif len(rc) == 5:
            rc[3] = -( rc[1] +  rc[1] * nfp * nfp) / (1.0 + 9 * nfp * nfp)
            rc[4]=-(1 + rc[2] + 4 * rc[2] * nfp * nfp) / (1.0 + 16 * nfp * nfp)
        elif len(rc) == 6:
            rc[4]=-(1 + rc[2] + 4 * rc[2] * nfp * nfp) / (1.0 + 16 * nfp * nfp)
            rc[5] = -( rc[1] +  rc[1] * nfp * nfp + rc[3] +  9*rc[3] * nfp * nfp) / (1.0 + 25 * nfp * nfp)
        elif len(rc) == 7:
            rc[5] = -( rc[1] +  rc[1] * nfp * nfp + rc[3] +  9*rc[3] * nfp * nfp) / (1.0 + 25 * nfp * nfp)
            rc[6]=-(1 + rc[2] + rc[4] + (rc[2] + 4 * rc[4]) * 4 * nfp * nfp) / (1.0 + 36 * nfp * nfp)
        elif len(rc) == 8:
            rc[6]=-(1 + rc[2] + rc[4] + (rc[2] + 4 * rc[4]) * 4 * nfp * nfp) / (1.0 + 36 * nfp * nfp)
            rc[7] = -( rc[1] +  rc[1] * nfp * nfp + rc[3] +  9*rc[3] * nfp * nfp + rc[5] +  25*rc[5] * nfp * nfp) / (1.0 + 49 * nfp * nfp)
        else:
            rc[7] = -((rc[1] + nfp*nfp*rc[1] + rc[3] + 9*nfp*nfp*rc[3] + rc[5] + 25*nfp*nfp*rc[5])/(1.0 + 49*nfp*nfp))
            rc[8] = -((1 + rc[2] + 4*nfp*nfp*rc[2] + rc[4] + 16*nfp*nfp*rc[4] + rc[6] + 36*nfp*nfp*rc[6])/(1.0+64*nfp*nfp))
    ### Conditions on rc and zs for half-helicity magnetic axis:
    ###
        if half_helicity == True:
            if len(zs) == 3:
                zs[2] = -(zs[1]*(2+nfp*nfp)) / (4.0*( 1.0 + 2.0* nfp * nfp))
            elif len(rc) == 4:
                zs[3] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp)) / (3.0*( 2.0 + 9.0* nfp * nfp))
            elif len(rc) == 5:
                zs[4] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) ) / (8.0*( 1.0 + 8.0* nfp * nfp))
            elif len(rc) == 6:
                zs[5] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) ) / (5.0*( 2.0 + 25.0* nfp * nfp))
            elif len(rc) == 7:
                zs[6] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) + 5*zs[5]*( 2.0 + 25.0* nfp * nfp) ) / (12.0*( 1.0 + 18.0* nfp * nfp))
            elif len(rc) == 8:
                zs[7] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) + 5*zs[5]*( 2.0 + 25.0* nfp * nfp) +12.0*zs[6]*( 1.0 + 18.0* nfp * nfp)   ) / (7.0*( 2.0 + 49.0* nfp * nfp))
            else:
                zs[8] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) + 5*zs[5]*( 2.0 + 25.0* nfp * nfp) +12.0*zs[6]*( 1.0 + 18.0* nfp * nfp) + 7.0*zs[7]*( 2.0 + 49.0* nfp * nfp)  ) / (16.0*( 1.0 + 32.0* nfp * nfp))
            #zs[10] = -(zs[1]*(2+nfp*nfp) + zs[2]*(4+8*nfp*nfp) + zs[3]*(6+27*nfp*nfp) + zs[4]*(8+64*nfp*nfp) + zs[5]*(10+125*nfp*nfp) \
        #+ zs[6]*(12+216*nfp*nfp) + zs[7]*(14+343*nfp*nfp) + zs[8]*(16+ 512*nfp*nfp) + zs[9]*(18+729*nfp*nfp)) / (20*(1+50*nfp*nfp))

    # Shorthand:
    nphi = self.nphi
    nfp = self.nfp

    phi = self.phi
    d_phi = self.d_phi
    R0 = np.zeros(nphi)
    Z0 = np.zeros(nphi)
    R0p = np.zeros(nphi)
    Z0p = np.zeros(nphi)
    R0pp = np.zeros(nphi)
    Z0pp = np.zeros(nphi)
    R0ppp = np.zeros(nphi)
    Z0ppp = np.zeros(nphi)
    for jn in range(0, self.nfourier):
        n = jn * nfp
        sinangle = np.sin(n * phi)
        cosangle = np.cos(n * phi)
        R0 += self.rc[jn] * cosangle + self.rs[jn] * sinangle
        Z0 += self.zc[jn] * cosangle + self.zs[jn] * sinangle
        R0p += self.rc[jn] * (-n * sinangle) + self.rs[jn] * (n * cosangle)
        Z0p += self.zc[jn] * (-n * sinangle) + self.zs[jn] * (n * cosangle)
        R0pp += self.rc[jn] * (-n * n * cosangle) + self.rs[jn] * (-n * n * sinangle)
        Z0pp += self.zc[jn] * (-n * n * cosangle) + self.zs[jn] * (-n * n * sinangle)
        R0ppp += self.rc[jn] * (n * n * n * sinangle) + self.rs[jn] * (-n * n * n * cosangle)
        Z0ppp += self.zc[jn] * (n * n * n * sinangle) + self.zs[jn] * (-n * n * n * cosangle)

    d_l_d_phi = np.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
    d3_l_d_phi3 = (R0p * R0p + R0pp * R0pp + Z0pp * Z0pp + R0 * R0pp + R0p * R0ppp + Z0p * Z0ppp - d2_l_d_phi2 * d2_l_d_phi2) / d_l_d_phi
    B0_over_abs_G0 = nphi / np.sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    self.d_l_d_varphi = abs_G0_over_B0
    G0 = self.sG * abs_G0_over_B0 * self.B0

    # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
    d_r_d_phi_cylindrical = np.array([R0p, R0, Z0p]).transpose()
    d2_r_d_phi2_cylindrical = np.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
    d3_r_d_phi3_cylindrical = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()

    tangent_cylindrical = np.zeros((nphi, 3))
    d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
    for j in range(3):
        tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
        d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                          + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

    curvature = np.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                        d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                        d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

    axis_length = np.sum(d_l_d_phi) * d_phi * nfp
    rms_curvature = np.sqrt((np.sum(curvature * curvature * d_l_d_phi) * d_phi * nfp) / axis_length)
    mean_of_R = np.sum(R0 * d_l_d_phi) * d_phi * nfp / axis_length
    mean_of_Z = np.sum(Z0 * d_l_d_phi) * d_phi * nfp / axis_length
    standard_deviation_of_R = np.sqrt(np.sum((R0 - mean_of_R) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)
    standard_deviation_of_Z = np.sqrt(np.sum((Z0 - mean_of_Z) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)

    normal_cylindrical = np.zeros((nphi, 3))
    for j in range(3):
        normal_cylindrical[:,j] = d_tangent_d_l_cylindrical[:,j] / curvature
    self.normal_cylindrical = normal_cylindrical

    # b = t x n
    binormal_cylindrical = np.zeros((nphi, 3))
    binormal_cylindrical[:,0] = tangent_cylindrical[:,1] * normal_cylindrical[:,2] - tangent_cylindrical[:,2] * normal_cylindrical[:,1]
    binormal_cylindrical[:,1] = tangent_cylindrical[:,2] * normal_cylindrical[:,0] - tangent_cylindrical[:,0] * normal_cylindrical[:,2]
    binormal_cylindrical[:,2] = tangent_cylindrical[:,0] * normal_cylindrical[:,1] - tangent_cylindrical[:,1] * normal_cylindrical[:,0]

    # If looking for omnigenity, use signed Frenet-Serret frame
    sign_curvature_change = np.ones((self.nphi,))
    if self.omn == True:
        nfp_phi_length = int(np.ceil(self.nphi/2))
        sign_curvature_change[nfp_phi_length:2*nfp_phi_length] = (-1)*np.ones((nfp_phi_length-1,))

    curvature = curvature* sign_curvature_change
    for j in range(3):
        normal_cylindrical[:,j]   =   normal_cylindrical[:,j]*sign_curvature_change
        binormal_cylindrical[:,j] = binormal_cylindrical[:,j]*sign_curvature_change

    self._determine_helicity()
    if half_helicity == True:  ### Half helicity conditions are enforced on the axis so we know helicity is 1/2
        self.helicity = 0.5
        #self.N_helicity =  -self.helicity * self.nfp
    self.N_helicity =  -self.helicity * self.nfp
    # We use the same sign convention for torsion as the
    # Landreman-Sengupta-Plunk paper, wikipedia, and
    # mathworld.wolfram.com/Torsion.html.  This sign convention is
    # opposite to Garren & Boozer's sign convention!
    torsion_numerator = (d_r_d_phi_cylindrical[:,0] * (d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,2] - d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,1]) \
                         + d_r_d_phi_cylindrical[:,1] * (d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,0] - d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,2]) 
                         + d_r_d_phi_cylindrical[:,2] * (d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,1] - d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,0]))

    torsion_denominator = (d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,2] - d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,1]) ** 2 \
        + (d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,0] - d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,2]) ** 2 \
        + (d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,1] - d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,0]) ** 2

    torsion = torsion_numerator / torsion_denominator

    # self.d_d_phi = spectral_diff_matrix(self.nphi, xmin = phi[0], xmax = phi[0] + 2*np.pi/self.nfp)
    self.d_d_phi = spectral_diff_matrix(self.nphi, xmin = 0, xmax = 2*np.pi/self.nfp)

    self.Bbar = self.spsi * np.mean(self.B0)

    self.interpolateTo0 = fourier_interpolation_matrix(nphi, -self.phi_shift*self.d_phi*self.nfp)
    # Calculate G0 and varphi
    if self.omn == False:
        # In here B0 is assumed to be given as a fourier series in phi
        # varphi starts from 0 and finishes at 2 * np.pi / nfp
        G0 = self.sG * np.sum(self.B0 * d_l_d_phi) / nphi
        self.varphi = np.zeros(nphi)
        d_l_d_phi_spline = self.convert_to_spline(d_l_d_phi)
        d_l_d_phi_from_zero = d_l_d_phi_spline(np.linspace(0,2*np.pi/self.nfp,self.nphi,endpoint=False))
        for j in range(1, nphi):
            # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
            self.varphi[j] = self.varphi[j-1] + (d_l_d_phi_from_zero[j-1] + d_l_d_phi_from_zero[j])
        self.varphi = self.varphi * (0.5 * d_phi * 2 * np.pi / axis_length)
    else:
        # In here B0 is assumed to be given as a fourier series in varphi
        # Picard iteration is used to find varphi and G0, with varphi periodic
        # but not starting necessarily at 0
        nu = np.zeros((nphi,))
        for j in range(20):
            varphi = phi + nu
            B0 =  np.array(sum([self.B0_vals[i] *np.cos(nfp*i*varphi) for i in range(len(self.B0_vals))]))
            B0 += np.array(sum([self.B0_svals[i]*np.sin(nfp*i*varphi) for i in range(len(self.B0_svals))]))
            abs_G0 = np.sum(B0 * d_l_d_phi) / nphi
            rhs = -1 + d_l_d_phi * B0 / abs_G0
            last_nu = nu
            nu = np.linalg.solve(self.d_d_phi+self.interpolateTo0, rhs)
            norm_change = np.sqrt(sum((nu-last_nu)**2)/nphi)
            logger.debug("  Iteration {}: |change to nu| = {}".format(j, norm_change))
            if norm_change < 1e-13:
                break
        varphi = phi + nu
        self.varphi = varphi
        B0 =  np.array(sum([self.B0_vals[i] *np.cos(nfp*i*varphi) for i in range(len(self.B0_vals))]))
        B0 += np.array(sum([self.B0_svals[i]*np.sin(nfp*i*varphi) for i in range(len(self.B0_svals))]))
        self.B0 = B0
        G0 = self.sG * np.sum(self.B0 * d_l_d_phi) / nphi
        dproptoB = True # makes d= sqrt(d_over_curvature/B0)*curvature, this is sometimes useful for half-helicity

        self.d = np.zeros(nphi)
        if not self.d_over_curvature == 0:
            if dproptoB == True: 
                self.d = np.sqrt(self.d_over_curvature/B0) * curvature
            else:
                self.d =self.d_over_curvature* curvature
        elif not self.d_over_curvature_cvals == []:
            if np.size(self.d_over_curvature_cvals) == self.nphi:
                self.d = self.d_over_curvature_cvals * curvature
            else:
                self.d = np.array(sum([self.d_over_curvature_cvals[i]*np.cos(nfp*i*varphi) * curvature for i in range(len(self.d_over_curvature_cvals))]))
        elif not len(self.d_over_curvature_spline)==0:
            N_points = len(self.d_over_curvature_spline)
            x_in = np.linspace(0,1,N_points)*np.pi/self.nfp
            y_in = self.d_over_curvature_spline
            x_in_periodic = np.append(x_in, 2*np.pi/self.nfp-x_in[-2::-1])
            y_in_periodic = np.append(y_in, y_in[-2::-1])
            # The order of the spline is important, as we are going to take derivatives with respect to phi
            spline_d_over_curv = make_interp_spline(x_in_periodic, y_in_periodic, bc_type = 'periodic', k = 7) 
            temp_d_over_curv = spline_d_over_curv(self.phi)
            self.d = temp_d_over_curv * curvature

        self.d -= self.k_second_order_SS * nfp * self.B0_vals[1] * np.sin(nfp * varphi) / B0
        self.d += np.array(sum([self.d_cvals[i]*np.cos(nfp*i*varphi)*curvature for i in range(len(self.d_cvals))]))
        self.d += np.array(sum([self.d_svals[i]*np.sin(nfp*i*varphi) *curvature*sign_curvature_change  for i in range(len(self.d_svals))]))

    self.d_l_d_varphi = self.sG * G0 / self.B0   
    self.d_d_varphi = np.zeros((nphi, nphi))
    for j in range(nphi):
        self.d_d_varphi[j,:] = self.d_d_phi[j,:] * self.sG * G0 / (self.B0[j] * d_l_d_phi[j])
    
    self.d_bar = self.d / (curvature + 1e-31)

    self.etabar_squared_over_curvature_squared = (self.B0  / self.Bbar) * self.d_bar**2

    # Add all results to self:
    self.phi = phi
    self.d_phi = d_phi
    self.R0 = R0
    self.Z0 = Z0
    self.R0p = R0p
    self.Z0p = Z0p
    self.R0pp = R0pp
    self.Z0pp = Z0pp
    self.R0ppp = R0ppp
    self.Z0ppp = Z0ppp
    self.G0 = G0
    self.d_l_d_phi = d_l_d_phi
    self.d2_l_d_phi2 = d2_l_d_phi2
    self.d3_l_d_phi3 = d3_l_d_phi3
    self.d_r_d_phi_cylindrical = d_r_d_phi_cylindrical
    self.d2_r_d_phi2_cylindrical = d2_r_d_phi2_cylindrical
    self.d3_r_d_phi3_cylindrical = d3_r_d_phi3_cylindrical
    self.axis_length = axis_length
    self.curvature = curvature
    self.torsion = torsion
    self.d_curvature_d_varphi = np.matmul(self.d_d_varphi, curvature)
    self.d_torsion_d_varphi = np.matmul(self.d_d_varphi, torsion)
    self.d_curvature_d_varphi_at_0 = self.d_curvature_d_varphi[0]
    self.d_d_d_varphi_at_0 = np.matmul(self.d_d_varphi, self.d)[0]
    self.sign_curvature_change = sign_curvature_change
    self.min_R0 = fourier_minimum(self.R0)
    self.min_Z0 = fourier_minimum(self.Z0)
    self.tangent_cylindrical = tangent_cylindrical
    self.normal_cylindrical = normal_cylindrical 
    self.binormal_cylindrical = binormal_cylindrical
    self.abs_G0_over_B0 = abs_G0_over_B0

    # The output is not stellarator-symmetric if (1) R0s is nonzero, (2) Z0c is nonzero, or (3) sigma_initial is nonzero
    if self.order == 'r1':
        self.lasym = np.max(np.abs(self.rs))>0 or np.max(np.abs(self.zc))>0 or np.abs(self.sigma0)>0
    else:
        self.lasym = np.max(np.abs(self.rs))>0 or np.max(np.abs(self.zc))>0 or np.abs(self.sigma0)>0 or np.any(np.array(self.B2c_svals) > 0.0) or np.any(np.array(self.B2c_svals) < 0.0)

    # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
    self.R0_func = self.convert_to_spline(sum([self.rc[i]*np.cos(i*self.nfp*self.phi) +\
                                               self.rs[i]*np.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.rc))]))
    self.Z0_func = self.convert_to_spline(sum([self.zc[i]*np.cos(i*self.nfp*self.phi) +\
                                               self.zs[i]*np.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.zs))]))

    # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
    self.normal_R_spline     = self.convert_to_spline(self.normal_cylindrical[:,0])
    self.normal_phi_spline   = self.convert_to_spline(self.normal_cylindrical[:,1])
    self.normal_z_spline     = self.convert_to_spline(self.normal_cylindrical[:,2])
    self.binormal_R_spline   = self.convert_to_spline(self.binormal_cylindrical[:,0])
    self.binormal_phi_spline = self.convert_to_spline(self.binormal_cylindrical[:,1])
    self.binormal_z_spline = self.convert_to_spline(self.binormal_cylindrical[:,2])
    self.tangent_R_spline = self.convert_to_spline(self.tangent_cylindrical[:,0])
    self.tangent_phi_spline = self.convert_to_spline(self.tangent_cylindrical[:,1])
    self.tangent_z_spline = self.convert_to_spline(self.tangent_cylindrical[:,2])

    # Spline interpolants for the cylindrical components of the Frenet-Serret frame (half helicity):
    if half_helicity == True:
        self.normal_cylindrical_tripled = np.append(-self.normal_cylindrical[:,:],self.normal_cylindrical[:,:],axis=0)
        self.normal_cylindrical_tripled = np.append(self.normal_cylindrical_tripled,-self.normal_cylindrical,axis=0) 
        self.phi_tripled = np.append(self.phi-(2*np.pi/self.nfp),self.phi,axis=0)
        self.phi_tripled = np.append(self.phi_tripled,self.phi+(2*np.pi/self.nfp),axis=0)
        self.binormal_cylindrical_tripled = np.append(-self.binormal_cylindrical,self.binormal_cylindrical,axis=0)
        self.binormal_cylindrical_tripled = np.append(self.binormal_cylindrical_tripled,-self.binormal_cylindrical,axis=0) 
        self.normal_R_spline_tripled     = self.convert_to_spline_tripled(self.normal_cylindrical_tripled[:,0])
        self.normal_phi_spline_tripled   = self.convert_to_spline_tripled(self.normal_cylindrical_tripled[:,1])
        self.normal_z_spline_tripled     = self.convert_to_spline_tripled(self.normal_cylindrical_tripled[:,2])
        self.binormal_R_spline_tripled    = self.convert_to_spline_tripled(self.binormal_cylindrical_tripled[:,0])
        self.binormal_phi_spline_tripled  = self.convert_to_spline_tripled(self.binormal_cylindrical_tripled[:,1])
        self.binormal_z_spline_tripled  = self.convert_to_spline_tripled(self.binormal_cylindrical_tripled[:,2])

    # Spline interpolant for the magnetic field on-axis as a function of phi (not varphi)
    self.B0_spline = self.convert_to_spline(self.B0)

    # Spline interpolant for nu = varphi-phi
    nu = self.varphi-self.phi
    self.nu_spline = self.convert_to_spline(nu)
    self.nu_spline_of_varphi = spline(np.append(self.varphi,self.varphi[0]+2*np.pi/self.nfp), np.append(self.varphi-self.phi,self.varphi[0]-self.phi[0]), bc_type='periodic')
