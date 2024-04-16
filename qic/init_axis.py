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
    curvature_zero_order = self.curvature_zero_order

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
            if curvature_zero_order==1:
                if len(zs) == 3:
                   zs[2] = -(zs[1]*(2+nfp*nfp)) / (4.0*( 1.0 + 2.0* nfp * nfp))
                elif len(zs) == 4:
                    #if*zs[2]==0:    ### zs(2)=0 seems to cause problems
                    #   *zs[2] = -(zs[1]*(2+nfp*nfp)) / (4.0*( 1.0 + 2.0* nfp * nfp))        
                    zs[3] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp)) / (3.0*( 2.0 + 9.0* nfp * nfp))
                elif len(zs) == 5:
                    zs[4] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) ) / (8.0*( 1.0 + 8.0* nfp * nfp))
                elif len(zs) == 6:
                    zs[5] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) ) / (5.0*( 2.0 + 25.0* nfp * nfp))
                elif len(zs) == 7:
                    zs[6] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) + 5*zs[5]*( 2.0 + 25.0* nfp * nfp) ) / (12.0*( 1.0 + 18.0* nfp * nfp))
                elif len(zs) == 8:
                    zs[7] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) + 5*zs[5]*( 2.0 + 25.0* nfp * nfp) +12.0*zs[6]*( 1.0 + 18.0* nfp * nfp)   ) / (7.0*( 2.0 + 49.0* nfp * nfp))
                else:
                    zs[8] = -(zs[1]*(2+nfp*nfp) + 4*zs[2]*(1.0+2.0*nfp*nfp) + 3*zs[3]*(2.0 +9.0*nfp*nfp) + 8*zs[4]*(1.0 +8.0*nfp*nfp) + 5*zs[5]*( 2.0 + 25.0* nfp * nfp) +12.0*zs[6]*( 1.0 + 18.0* nfp * nfp) + 7.0*zs[7]*( 2.0 + 49.0* nfp * nfp)  ) / (16.0*( 1.0 + 32.0* nfp * nfp))
                #zs[10] = -(zs[1]*(2+nfp*nfp) +*zs[2]*(4+8*nfp*nfp) + zs[3]*(6+27*nfp*nfp) + zs[4]*(8+64*nfp*nfp) + zs[5]*(10+125*nfp*nfp) \
                #+ zs[6]*(12+216*nfp*nfp) + zs[7]*(14+343*nfp*nfp) + zs[8]*(16+ 512*nfp*nfp) + zs[9]*(18+729*nfp*nfp)) / (20*(1+50*nfp*nfp)) 
            elif curvature_zero_order == 3:
                if len(zs) == 4:
                    rc[1] =  ((5 + 4*nfp**2)*(1 + 9*nfp**2))/( 2*(5 + 30*nfp**2 + 49*nfp**4 + 36*nfp**6))
                    rc[2] = -1.0 / (1.0 + 4.0 * nfp * nfp)
                    rc[3] =  -(((1 +nfp**2)*(5 + 4*nfp**2))/(2*(5 + 30*nfp**2 + 49*nfp**4 + 36*nfp**6)))
                    zs[2] = 0
                    zs[3] = -(zs[1]*(2+nfp*nfp)) / (3.0*( 2.0 + 9.0*nfp * nfp))
                
                elif len(zs) == 5:
                    rc[2] = (2*(-10 - 122*nfp**2 - 288*nfp**4 + 5*rc[1] + 90*nfp**2*rc[1] + \
                                169*nfp**4*rc[1] + 144*nfp**6*rc[1]))/(3*(5 + 65*nfp**2 + 244*nfp**4 + 576*nfp**6))
                    rc[3] =  -((rc[1] +nfp**2*rc[1])/(1 + 9*nfp**2))
                    rc[4] = -((-5 - 49*nfp**2 - 36*nfp**4 + 10*rc[1] + 60*nfp**2*rc[1] + \
                               98*nfp**4*rc[1] + 72*nfp**6*rc[1])/(3*(5 + 65*nfp**2 + 244*nfp**4 + 576*nfp**6)))
                    zs[3] = -(zs[1]*(2+nfp*nfp) ) / (3.0*( 2.0 + 9.0* nfp * nfp))
                    zs[4] = -(((1 + 2*nfp**2)*zs[2])/(2*(1 + 8*nfp**2)))

                elif len(zs) == 6:
                    rc[3] = -((-20 - 564*nfp**2 - 1600*nfp**4 + 30*rc[1] + 636*(nfp**2)*rc[1] +\
                                2646*(nfp**4)*rc[1] + 2400*(nfp**6)*rc[1] -15*rc[2] - 435*nfp**2*rc[2] - \
                                1692*nfp**4*rc[2] - 4800*nfp**6*rc[2])/(4*(5 + 114*nfp**2 + 769*nfp**4 +\
                                3600*nfp**6)))
                    rc[4] = -((1 + rc[2] + 4*nfp**2*rc[2])/(1 + 16*nfp**2))
                    rc[5] = -((20 + 244*nfp**2 + 576*nfp**4 - 10*rc[1] - 180*nfp**2*rc[1] - \
                               338*nfp**4*rc[1] - 288*nfp**6*rc[1] + 15*rc[2] + 195*nfp**2*rc[2] +\
                               732*nfp**4*rc[2] + 1728*nfp**6*rc[2])/(4*(5 + 114*nfp**2 + 769*nfp**4 + 3600*nfp**6)))
                    zs[4] = -((zs[2] + 2*(nfp**2)*zs[2])/(2*(1 + 8*nfp**2))) 
                    zs[5] = -((2*zs[1] + (nfp**2)*zs[1] + 6*zs[3] + 27*(nfp**2)*zs[3])/(5*(2 + 25*nfp**2)))               
                
                elif len(zs) == 7:
                    rc[4] = -((45 + 1449 *nfp**2 + 8100 *nfp**4 - 30*rc[1] - 1236 *nfp**2*rc[1] -\
                                5766 *nfp**4*rc[1] - 5400 *nfp**6*rc[1] + 40*rc[2] + 1320 *nfp**2*rc[2] +\
                                9152 *nfp**4*rc[2] + 28800 *nfp**6*rc[2] - 20*rc[3] - 856 *nfp**2*rc[3] -\
                                5796 *nfp**4*rc[3] - 32400 *nfp**6*rc[3])/(5*(5 + 177 *nfp**2 +\
                                1876 *nfp**4 + 14400 *nfp**6)))
                    rc[5] = -((rc[1] + nfp**2*rc[1] +rc[3] + 9*nfp**2*rc[3])/(1 + 25 *nfp**2))
                    rc[6] = -((-20 - 564*nfp**2 - 1600*nfp**4 + 30*rc[1] + 636 *nfp**2*rc[1] +\
                                2646*nfp**4*rc[1] + 2400*nfp**6*rc[1] - 15*rc[2] - 435*nfp**2*rc[2] -\
                                1692*nfp**4*rc[2] - 4800*nfp**6*rc[2] + 20*rc[3] + 456*nfp**2*rc[3] +\
                                3076*nfp**4*rc[3] + 14400*nfp**6*rc[3])/(5*(5 + 177 *nfp**2 +\
                                1876*nfp**4 + 14400 *nfp**6)))
                    zs[5] = -((2*zs[1] + (nfp**2)*zs[1] + 6*zs[3] + 27*(nfp**2)*zs[3])/(5*(2 + 25*nfp**2)))
                    zs[6] = -((zs[2] + 2*(nfp**2)*zs[2] + 2*zs[4] + 16*(nfp**2)*zs[4])/(3*(1 + 18*nfp**2)))
                
                elif len(zs) == 8:
                    rc[5] = -((-45 - 2529*nfp**2 - 15876*nfp**4 + 60*rc[1] + 2760*nfp**2*rc[1] +\
                                22188*nfp**4*rc[1] + 21168*nfp**6*rc[1] - 40*rc[2] - 2280*nfp**2*rc[2] -\
                                16832*nfp**4*rc[2] - 56448*nfp**6*rc[2] + 50*rc[3] + 2380*nfp**2*rc[3] +\
                                25290*nfp**4*rc[3] + 158760*nfp**6*rc[3] - 25*rc[4] - 1485*nfp**2*rc[4] -\
                                15620*nfp**4*rc[4] - 141120*nfp**6*rc[4])/(6*(5 + 254*nfp**2 + 3889*nfp**4 +\
                                44100*nfp**6)))
                    rc[6] = -((1 + rc[2] + 4*nfp**2*rc[2] +rc[4] + 16*nfp**2*rc[4])/(1 + 36*nfp**2))
                    rc[7] = -((45 + 1449*nfp**2 + 8100*nfp**4 - 30*rc[1] - 1236*nfp**2*rc[1] -\
                                5766*nfp**4*rc[1] - 5400*nfp**6*rc[1] + 40*rc[2] + 1320*nfp**2*rc[2] +\
                                9152*nfp**4*rc[2] + 28800*nfp**6*rc[2] - 20*rc[3] - 856*nfp**2*rc[3] -\
                                5796*nfp**4*rc[3] - 32400*nfp**6*rc[3] + 25*rc[4] + 885*nfp**2*rc[4] +\
                                9380*nfp**4*rc[4] + 72000*nfp**6*rc[4])/(6*(5 + 254*nfp**2 + 3889*nfp**4 +\
                                44100*nfp**6)))
                       
                    zs[6] = -((zs[2] + 2*(nfp**2)*zs[2] + 2*zs[4] + 16*(nfp**2)*zs[4])/(3*(1 + 18*nfp**2)))
                    zs[7] = -((2*zs[1] + (nfp**2)*zs[1] + 6*zs[3] + 27*(nfp**2)*zs[3] + 10*zs[5] +\
                                125*(nfp**2)*zs[5])/(7*(2 + 49*nfp**2)))

                else:
                    rc[6] = -((80 + 4944*nfp**2 + 50176*nfp**4 - 60*rc[1] - 4440*nfp**2*rc[1] -\
                                38988*nfp**4*rc[1] - 37632*nfp**6*rc[1] + 75*rc[2] + 4695*nfp**2*rc[2] +\
                                53820*nfp**4*rc[2] + 188160*nfp**6*rc[2] - 50*rc[3] - 3780*nfp**2*rc[3] -\
                                41530*nfp**4*rc[3] - 282240*nfp**6*rc[3] + 60*rc[4] + 3900*nfp**2*rc[4] +\
                                59328*nfp**4*rc[4] + 602112*nfp**6*rc[4] - 30*rc[5] - 2364*nfp**2*rc[5] -\
                                35766*nfp**4*rc[5] - 470400*nfp**6*rc[5])/(7*(5 + 345*nfp**2 + 7204*nfp**4 +\
                                112896*nfp**6)))
                    rc[7] = -((rc[1] + nfp**2*rc[1] + rc[3] + 9*nfp**2*rc[3] + rc[5] +\
                                25*nfp**2*rc[5])/(1 + 49*nfp**2))
                    rc[8] = -((-45 - 2529*nfp**2 - 15876*nfp**4 + 60*rc[1] + 2760*nfp**2*rc[1] +\
                                22188*nfp**4*rc[1] + 21168*nfp**6*rc[1] - 40*rc[2] - 2280*nfp**2*rc[2] -\
                                16832*nfp**4*rc[2] - 56448*nfp**6*rc[2] + 50*rc[3] + 2380*nfp**2*rc[3] +\
                                25290*nfp**4*rc[3] + 158760*nfp**6*rc[3] - 25*rc[4] - 1485*nfp**2*rc[4] -\
                                15620*nfp**4*rc[4] - 141120*nfp**6*rc[4] + 30*rc[5] + 1524*nfp**2*rc[5] +\
                                23334*nfp**4*rc[5] + 264600*nfp**6*rc[5])/(7*(5 + 345*nfp**2 + 7204*nfp**4 +\
                                112896*nfp**6)))
                    zs[7] = -((2*zs[1] + (nfp**2)*zs[1] + 6*zs[3] + 27*(nfp**2)*zs[3] + 10*zs[5] +\
                                125*(nfp**2)*zs[5])/(7*(2 + 49*nfp**2)))
                    zs[8] = -((zs[2] + 2*(nfp**2)*zs[2] + 2*zs[4] + 16*(nfp**2)*zs[4] + 3*zs[6] +\
                                54*(nfp**2)*zs[6])/(4*(1 + 32*nfp**2)))            
            elif curvature_zero_order == 5:
                if len(zs) == 5:
                    rc[1] = (4*(1 + 9*(nfp**2))*(61 + 100*(nfp**2) + 64*(nfp**4)))/(5*(61 +\
                                458*(nfp**2) + 749*(nfp**4) + 820*(nfp**6) + 576*(nfp**8))) 
                    rc[2] = -((4*(61 + 46*(nfp**2) + 109*(nfp**4) + 144*(nfp**6)))/(5*(61 +\
                                458*(nfp**2) + 749*(nfp**4) + 820*(nfp**6) + 576*(nfp**8))))
                    rc[3] = -((4*(1 + (nfp**2))*(61 + 100*(nfp**2) + 64*(nfp**4)))/(5*(61 +\
                                458*(nfp**2) + 749*(nfp**4) + 820*(nfp**6) + 576*(nfp**8)))) 
                    rc[4] = -((61 + 154*(nfp**2) + 109*(nfp**4) + 36*(nfp**6))/(5*(61 +\
                                458*(nfp**2) + 749*(nfp**4) + 820*(nfp**6) + 576*(nfp**8)))) 
                    zs[2] = ((1 + 8*(nfp**2))*(16*zs[1] + 20*(nfp**2)*zs[1] + 9*(nfp**4)*zs[1]))/(12*(4 +\
                                28*(nfp**2) + 61*(nfp**4) + 72*(nfp**6))) 
                    zs[3] = -((2*zs[1] + (nfp**2)*zs[1])/(3*(2 + 9*(nfp**2)))) 
                    zs[4] = -(((1 + 2*(nfp**2))*(16*zs[1] + 20*(nfp**2)*zs[1] + 9*(nfp**4)*zs[1]))/(24*(4 +\
                                28*(nfp**2) + 61*(nfp**4) + 72*(nfp**6))))
                elif len(zs) == 6:
                    rc[2] = (4*(-183 - 2850*(nfp**2) - 7407*(nfp**4) - 10800*(nfp**6) + 122*rc[1] +\
                                2302*(nfp**2)*rc[1] + 6118*(nfp**4)*rc[1] + 8738*(nfp**6)*rc[1] +\
                                7200*(nfp**8)*rc[1]))/(7*(61 + 914*(nfp**2) + 2789*(nfp**4) +\
                                                           6676*(nfp**6) + 14400*(nfp**8))) 
                    rc[3] = -((244 + 6500*(nfp**2) + 10256*(nfp**4) + 6400*(nfp**6) + 549*rc[1] -\
                              1206*(nfp**2)*rc[1] + 1701*(nfp**4)*rc[1] + 19476*(nfp**6)*rc[1] +\
                                  14400*(nfp**8)*rc[1])/(14*(61 + 914*(nfp**2) + 2789*(nfp**4) +\
                                                              6676*(nfp**6) + 14400*(nfp**8)))) 
                    rc[4] = -((-305 - 3050*(nfp**2) - 6905*(nfp**4) - 4500*(nfp**6) + 488*rc[1] +\
                              3352*(nfp**2)*rc[1] + 7672*(nfp**4)*rc[1] + 10088*(nfp**6)*rc[1] +\
                              7200*(nfp**8)*rc[1])/(7*(61 + 914*(nfp**2) + 2789*(nfp**4) +\
                                                       6676*(nfp**6) + 14400*(nfp**8)))) 
                    rc[5] = -((-244 - 2596*(nfp**2) - 3856*(nfp**4) - 2304*(nfp**6) + 305*rc[1] +\
                               2290*(nfp**2)*rc[1] + 3745*(nfp**4)*rc[1] + 4100*(nfp**6)*rc[1] +\
                                  2880*(nfp**8)*rc[1])/(14*(61 + 914*(nfp**2) + 2789*(nfp**4) +\
                                                             6676*(nfp**6) + 14400*(nfp**8))))
                    zs[3] = -((16*zs[1] + 180*(nfp**2)*zs[1] + 441*(nfp**4)*zs[1] +\
                                200*(nfp**6)*zs[1] - 16*zs[2] - 240*(nfp**2)*zs[2] -\
                                      564*(nfp**4)*zs[2] - 800*(nfp**6)*zs[2])/(2*(16 + 196*(nfp**2) + 769*(nfp**4) + 1800*(nfp**6))))
                    zs[4] = -((zs[2] + 2*(nfp**2)*zs[2])/(2*(1 + 8*(nfp**2)))) 
                    zs[5] = -((-16*zs[1] - 148*(nfp**2)*zs[1] - 169*(nfp**4)*zs[1] -\
                                2*(nfp**6)*zs[1] + 48*zs[2] + 336*(nfp**2)*zs[2] + 732*(nfp**4)*zs[2] +\
                                864*(nfp**6)*zs[2])/(10*(16 + 196*(nfp**2) + 769*(nfp**4) + 1800*(nfp**6))))
                 
                elif len(zs) == 7:
                    rc[3] = -((-732 - 21420*(nfp**2) - 84912*(nfp**4) - 172800*(nfp**6) + 793*rc[1] +\
                                20498*(nfp**2)*rc[1] + 83993*(nfp**4)*rc[1] + 211588*(nfp**6)*rc[1] +\
                                187200*(nfp**8)*rc[1] - 488*rc[2] - 14440*(nfp**2)*rc[2] -\
                                      62272*(nfp**4)*rc[2] - 175232*(nfp**6)*rc[2] - \
                                        460800*(nfp**8)*rc[2])/(6*(61 + 1522*(nfp**2) + 7525*(nfp**4) +\
                                                                   31284*(nfp**6) + 129600*(nfp**8))))
                    rc[4] = -((183 - 17850*(nfp**2) - 41553*(nfp**4) + 24300*(nfp**6) + 488*rc[1] +\
                                18968*(nfp**2)*rc[1] + 52472*(nfp**4)*rc[1] + 76392*(nfp**6)*rc[1] +\
                                    64800*(nfp**8)*rc[1] + 488*rc[2] - 3088*(nfp**2)*rc[2] +\
                                    3752*(nfp**4)*rc[2] + 111168*(nfp**6)*rc[2] +\
                                          259200*(nfp**8)*rc[2])/(15*(61 + 1522*(nfp**2) + 7525*(nfp**4) +\
                                                                    31284*(nfp**6) + 129600*(nfp**8)))) 
                    rc[5] = -((732 + 9708*(nfp**2) + 34992*(nfp**4) + 62208*(nfp**6) - 427*rc[1] -\
                                7462*(nfp**2)*rc[1] - 27643*(nfp**4)*rc[1] - 43596*(nfp**6)*rc[1] -\
                                36288*(nfp**8)*rc[1] + 488*rc[2] + 6632*(nfp**2)*rc[2] +\
                                      26432*(nfp**4)*rc[2] + 74880*(nfp**6)*rc[2] +\
                                          165888*(nfp**8)*rc[2])/(6*(61 + 1522*(nfp**2) + 7525*(nfp**4) +\
                                                                    31284*(nfp**6) + 129600*(nfp**8)))) 
                    rc[6] = -((732 + 11400*(nfp**2) + 29628*(nfp**4) + 43200*(nfp**6) - 488*rc[1] -\
                                9208*(nfp**2)*rc[1] - 24472*(nfp**4)*rc[1] - 34952*(nfp**6)*rc[1] -\
                                28800*(nfp**8)*rc[1] + 427*rc[2] + 6398*(nfp**2)*rc[2] +\
                                      19523*(nfp**4)*rc[2] + 46732*(nfp**6)*rc[2] +\
                                          100800*(nfp**8)*rc[2])/(15*(61 + 1522*(nfp**2) + 7525*(nfp**4) +\
                                                                    31284*(nfp**6) + 129600*(nfp**8)))) 
                    zs[4] = -((-48*zs[1] - 1020*(nfp**2)*zs[1] - 2883*(nfp**4)*zs[1] - 1350*(nfp**6)*zs[1] +\
                                128*zs[2] + 2240*(nfp**2)*zs[2] + 9152*(nfp**4)*zs[2] + 14400*(nfp**6)*zs[2] -\
                                96*zs[3] - 2136*(nfp**2)*zs[3] - 8694*(nfp**4)*zs[3] -\
                                      24300*(nfp**6)*zs[3])/(40*(4 + 76*(nfp**2) + 469*(nfp**4) + 1800*(nfp**6)))) 
                    zs[5] = -((2*zs[1] + (nfp**2)*zs[1] + 6*zs[3] + 27*(nfp**2)*zs[3])/(5*(2 + 25*(nfp**2)))) 
                    zs[6] = -((16*zs[1] + 180*(nfp**2)*zs[1] + 441*(nfp**4)*zs[1] + 200*(nfp**6)*zs[1] -\
                                16*zs[2] - 240*(nfp**2)*zs[2] - 564*(nfp**4)*zs[2] - 800*(nfp**6)*zs[2] +\
                                      32*zs[3] + 392*(nfp**2)*zs[3] + 1538*(nfp**4)*zs[3] +\
                                          3600*(nfp**6)*zs[3])/(20*(4 + 76*(nfp**2) + 469*(nfp**4) + 1800*(nfp**6))))

                elif len(zs) == 8:
                    rc[4] = -((10431 + 402750*(nfp**2) + 2296719*(nfp**4) + 7541100*(nfp**6) -\
                                8784*rc[1] - 370224*(nfp**2)*rc[1] - 2131056*(nfp**4)*rc[1] -\
                                6910416*(nfp**6)*rc[1] - 6350400*(nfp**8)*rc[1] + 8296*rc[2] +\
                                319664*(nfp**2)*rc[2] + 1946984*(nfp**4)*rc[2] + 8113216*(nfp**6)*rc[2] +\
                                23990400*(nfp**8)*rc[2] - 4880*rc[3] - 208880*(nfp**2)*rc[3] -\
                                      1346480*(nfp**4)*rc[3] - 6328080*(nfp**6)*rc[3] -\
                                        31752000*(nfp**8)*rc[3])/(55*(61 + 2282*(nfp**2) +\
                                        16685*(nfp**4) + 106324*(nfp**6) + 705600*(nfp**8)))) 
                    rc[5] = -((732 + 38988*(nfp**2) + 159792*(nfp**4) + 338688*(nfp**6) - 122*rc[1] -\
                                30532*(nfp**2)*rc[1] - 116458*(nfp**4)*rc[1] - 62696*(nfp**6)*rc[1] -\
                                56448*(nfp**8)*rc[1] + 488*rc[2] + 26152*(nfp**2)*rc[2] +\
                                      116032*(nfp**4)*rc[2] + 325760*(nfp**6)*rc[2] + 903168*(nfp**8)*rc[2] +\
                                        305*rc[3] - 3550*(nfp**2)*rc[3] + 5585*(nfp**4)*rc[3] +\
                                        281700*(nfp**6)*rc[3] + 70080*(nfp**8)*rc[3])/(11*(61 +\
                                        2282*(nfp**2) + 16685*(nfp**4) + 106324*(nfp**6) + 705600*(nfp**8)))) 
                    rc[6] = -((-7076 - 189400*(nfp**2) - 1004644*(nfp**4) - 2273600*(nfp**6) + 8784*rc[1] +\
                                194544*(nfp**2)*rc[1] + 1051056*(nfp**4)*rc[1] + 3169296*(nfp**6)*rc[1] +\
                                2822400*(nfp**8)*rc[1] - 4941*rc[2] - 135594*(nfp**2)*rc[2] -\
                                760509*(nfp**4)*rc[2] - 2368116*(nfp**6)*rc[2] - 6350400*(nfp**8)*rc[2] +\
                                4880*rc[3] + 111280*(nfp**2)*rc[3] + 682480*(nfp**4)*rc[3] +\
                                    3302480*(nfp**6)*rc[3] + 14112000*(nfp**8)*rc[3])/(55*(61 +\
                                    2282*(nfp**2) + 16685*(nfp**4) + 106324*(nfp**6) + 705600*(nfp**8)))) 
                    rc[7] = -((-732 - 21420*(nfp**2) - 84912*(nfp**4) - 172800*(nfp**6) + 793*rc[1] +\
                                20498*(nfp**2)*rc[1] + 83993*(nfp**4)*rc[1] + 211588*(nfp**6)*rc[1] +\
                                187200*(nfp**8)*rc[1] - 488*rc[2] - 14440*(nfp**2)*rc[2] -\
                                62272*(nfp**4)*rc[2] - 175232*(nfp**6)*rc[2] - 460800*(nfp**8)*rc[2] +\
                                366*rc[3] + 9132*(nfp**2)*rc[3] + 45150*(nfp**4)*rc[3] +\
                                187704*(nfp**6)*rc[3] + 777600*(nfp**8)*rc[3])/(11*(61 + 2282*(nfp**2) +\
                                    16685*(nfp**4) + 106324*(nfp**6) + 705600*(nfp**8)))) 
                    zs[5] = -((96*zs[1] + 2328*(nfp**2)*zs[1] + 11094*(nfp**4)*zs[1] + 5292*(nfp**6)*zs[1] -\
                                128*zs[2] - 3776*(nfp**2)*zs[2] - 16832*(nfp**4)*zs[2] - 28224*(nfp**6)*zs[2] +\
                                240*zs[3] + 6060*(nfp**2)*zs[3] + 37935*(nfp**4)*zs[3] + 119070*(nfp**6)*zs[3] -\
                                160*zs[4] - 4960*(nfp**2)*zs[4] - 31240*(nfp**4)*zs[4] -\
                                      141120*(nfp**6)*zs[4])/(15*(16 + 436*(nfp**2) + 3889*(nfp**4) + 22050*(nfp**6)))) 
                    zs[6] = -((zs[2] + 2*(nfp**2)*zs[2] + 2*zs[4] + 16*(nfp**2)*zs[4])/(3*(1 +\
                                                                                18*(nfp**2)))) 
                    zs[7] = -((-48*zs[1] - 1020*(nfp**2)*zs[1] - 2883*(nfp**4)*zs[1] - 1350*(nfp**6)*zs[1] +\
                                128*zs[2] + 2240*(nfp**2)*zs[2] + 9152*(nfp**4)*zs[2] + 14400*(nfp**6)*zs[2] -\
                                96*zs[3] - 2136*(nfp**2)*zs[3] - 8694*(nfp**4)*zs[3] - 24300*(nfp**6)*zs[3] +\
                                    160*zs[4] + 3040*(nfp**2)*zs[4] + 18760*(nfp**4)*zs[4] +\
                                    72000*(nfp**6)*zs[4])/(21*(16 + 436*(nfp**2) + 3889*(nfp**4) + 22050*(nfp**6))))
                else:  
                    rc[5] = -((-2928 - 167472*(nfp**2) - 1286592*(nfp**4) - 5419008*(nfp**6) +\
                                3050*rc[1] + 163780*(nfp**2)*rc[1] + 1279450*(nfp**4)*rc[1] +\
                                6005000*(nfp**6)*rc[1] + 5644800*(nfp**8)*rc[1] - 2440*rc[2] -\
                                140360*(nfp**2)*rc[2] - 1127360*(nfp**4)*rc[2] - 5668480*(nfp**6)*rc[2] -\
                                18063360*(nfp**8)*rc[2] + 2135*rc[3] + 114590*(nfp**2)*rc[3] +\
                                981815*(nfp**4)*rc[3] + 6220620*(nfp**6)*rc[3] + 35562240*(nfp**8)*rc[3] -\
                                1220*rc[4] - 71380*(nfp**2)*rc[4] - 646480*(nfp**4)*rc[4] -\
                                4563200*(nfp**6)*rc[4] - 36126720*(nfp**8)*rc[4])/(13*(61 + 3194*(nfp**2) +\
                                                32429*(nfp**4) + 292996*(nfp**6) + 2822400*(nfp**8)))) 
                    rc[6] = -((-4880 - 604000*(nfp**2) - 3377680*(nfp**4) - 6272000*(nfp**6) +\
                                8784*rc[1] + 616176*(nfp**2)*rc[1] + 3643056*(nfp**4)*rc[1] +\
                                12147984*(nfp**6)*rc[1] + 11289600*(nfp**8)*rc[1] - 2745*rc[2] -\
                                438930*(nfp**2)*rc[2] - 2490345*(nfp**4)*rc[2] - 4600980*(nfp**6)*rc[2] -\
                                14112000*(nfp**8)*rc[2] + 4880*rc[3] + 345520*(nfp**2)*rc[3] +\
                                2276080*(nfp**4)*rc[3] + 10563920*(nfp**6)*rc[3] + 56448000*(nfp**8)*rc[3] +\
                                2196*rc[4] - 39816*(nfp**2)*rc[4] + 84564*(nfp**4)*rc[4] +\
                                6255936*(nfp**6)*rc[4] + 45158400*(nfp**8)*rc[4])/(91*(61 + 3194*(nfp**2) +\
                                            32429*(nfp**4) + 292996*(nfp**6) + 2822400*(nfp**8)))) 
                    rc[7] = -((2928 + 97200*(nfp**2) + 710592*(nfp**4) + 2764800*(nfp**6) - 2257*rc[1] -\
                                87122*(nfp**2)*rc[1] - 641873*(nfp**4)*rc[1] - 2308948*(nfp**6)*rc[1] -\
                                2131200*(nfp**8)*rc[1] + 2440*rc[2] + 81800*(nfp**2)*rc[2] +\
                                628160*(nfp**4)*rc[2] + 3072640*(nfp**6)*rc[2] + 9216000*(nfp**8)*rc[2] -\
                                1342*rc[3] - 53548*(nfp**2)*rc[3] - 427438*(nfp**4)*rc[3] -\
                                2218392*(nfp**6)*rc[3] - 11404800*(nfp**8)*rc[3] + 1220*rc[4] +\
                                42100*(nfp**2)*rc[4] + 368080*(nfp**4)*rc[4] + 2689280*(nfp**6)*rc[4] +\
                                18432000*(nfp**8)*rc[4])/(13*(61 + 3194*(nfp**2) + 32429*(nfp**4) +\
                                                               292996*(nfp**6) + 2822400*(nfp**8)))) 
                    rc[8] = -((10431 + 402750*(nfp**2) + 2296719*(nfp**4) + 7541100*(nfp**6) -\
                                8784*rc[1] - 370224*(nfp**2)*rc[1] - 2131056*(nfp**4)*rc[1] -\
                                6910416*(nfp**6)*rc[1] - 6350400*(nfp**8)*rc[1] + 8296*rc[2] +\
                                319664*(nfp**2)*rc[2] + 1946984*(nfp**4)*rc[2] + 8113216*(nfp**6)*rc[2] +\
                                23990400*(nfp**8)*rc[2] - 4880*rc[3] - 208880*(nfp**2)*rc[3] -\
                                1346480*(nfp**4)*rc[3] - 6328080*(nfp**6)*rc[3] -\
                                31752000*(nfp**8)*rc[3] + 3355*rc[4] + 125510*(nfp**2)*rc[4] +\
                                917675*(nfp**4)*rc[4] + 5847820*(nfp**6)*rc[4] +\
                                38808000*(nfp**8)*rc[4])/(91*(61 + 3194*(nfp**2) + 32429*(nfp**4) +\
                                                               292996*(nfp**6) + 2822400*(nfp**8)))) 
                    zs[6] = -((-32*zs[1] - 1224*(nfp**2)*zs[1] - 6498*(nfp**4)*zs[1] - 3136*(nfp**6)*zs[1] +\
                                80*zs[2] + 2640*(nfp**2)*zs[2] + 17940*(nfp**4)*zs[2] + 31360*(nfp**6)*zs[2] -\
                                80*zs[3] - 3140*(nfp**2)*zs[3] - 20765*(nfp**4)*zs[3] - 70560*(nfp**6)*zs[3] +\
                                128*zs[4] + 4416*(nfp**2)*zs[4] + 39552*(nfp**4)*zs[4] + 200704*(nfp**6)*zs[4] -\
                                80*zs[5] - 3300*(nfp**2)*zs[5] - 29805*(nfp**4)*zs[5] -\
                                196000*(nfp**6)*zs[5])/(28*(4 + 148*(nfp**2) + 1801*(nfp**4) + 14112*(nfp**6)))) 
                    zs[7] = -((2*zs[1] + (nfp**2)*zs[1] + 6*zs[3] + 27*(nfp**2)*zs[3] + 10*zs[5] +\
                                125*(nfp**2)*zs[5])/(7*(2 + 49*(nfp**2))))
                    zs[8] = -((96*zs[1] + 2328*(nfp**2)*zs[1] + 11094*(nfp**4)*zs[1] + 5292*(nfp**6)*zs[1] -\
                                128*zs[2] - 3776*(nfp**2)*zs[2] - 16832*(nfp**4)*zs[2] -\
                                28224*(nfp**6)*zs[2] + 240*zs[3] + 6060*(nfp**2)*zs[3] +\
                                37935*(nfp**4)*zs[3] + 119070*(nfp**6)*zs[3] - 160*zs[4] -\
                                4960*(nfp**2)*zs[4] - 31240*(nfp**4)*zs[4] - 141120*(nfp**6)*zs[4] +\
                                240*zs[5] + 6540*(nfp**2)*zs[5] + 58335*(nfp**4)*zs[5] +\
                                330750*(nfp**6)*zs[5])/(112*(4 + 148*(nfp**2) + 1801*(nfp**4) + 14112*(nfp**6))))


            #     rc[4] = -((45 + 1449*nfp*nfp + 8100*nfp*nfp*nfp*nfp - 30 *rc[1] - 1236*nfp*nfp*rc[1] - 5766*(nfp*nfp*nfp*nfp)*rc[1] - 5400*(nfp*nfp*nfp*nfp*nfp*nfp)*rc[1] + 40 *rc[2] \
            #    + 1320 *nfp*nfp*rc[2] + 9152 *nfp*nfp*nfp*nfp *rc[2] + 28800 *nfp*nfp*nfp*nfp*nfp*nfp *rc[2]- 20 *rc[3] - 856 *nfp*nfp*rc[3] - 5796* nfp*nfp*nfp*nfp *rc[3] \
            #     - 32400 *nfp*nfp*nfp*nfp*nfp*nfp *rc[3])/(5*(5 + 177*nfp*nfp + 1876*nfp*nfp*nfp*nfp + 14400*nfp*nfp*nfp*nfp*nfp*nfp)))

            #     rc[5] = -((rc[1] + nfp*nfp*rc[1] + rc[3]+ 9*nfp*nfp *rc[3])/(1 + 25*nfp*nfp))

            #     rc[6] = -((-20 - 564*nfp*nfp - 1600*nfp*nfp*nfp*nfp + 30*rc[1] + 636*nfp*nfp *rc[1]+ 2646 *nfp*nfp*nfp*nfp *rc[1] \
            #    + 2400 *nfp*nfp*nfp*nfp*nfp*nfp *rc[1] - 15 *rc[2] - 435 *nfp*nfp *rc[2] - 1692 *nfp*nfp*nfp*nfp *rc[2] \
            #     - 4800 *nfp*nfp*nfp*nfp*nfp*nfp *rc[2] + 20 *rc[3]+ 456 *nfp*nfp *rc[3]+ 3076 *nfp*nfp*nfp*nfp *rc[3] \
            #     + 14400 *nfp*nfp*nfp*nfp*nfp*nfp *rc[3])/(5*(5 + 177 *nfp*nfp + 1876 *nfp*nfp*nfp*nfp \
            #     + 14400 *nfp*nfp*nfp*nfp*nfp*nfp)))

            #     zs[5] = -((2*zs[1] + nfp*nfp*zs[1] + 6 *zs[3] + 27*nfp*nfp*zs[3])/(5*(2 + 25*nfp*nfp)))

            #     zs[6] = -((zs[2] + 2*nfp*nfp *zs[2] + 2 *zs[4] + 16*nfp*nfp *zs[4])/(3*(1 + 18*nfp*nfp)))

    temp2 = np.zeros(self.nfourier)
    temp2[:len(zs)] = zs
    zs = temp2

    temp3 = np.zeros(self.nfourier)
    temp3[:len(rc)] = rc
    rc = temp3

    self.rc = rc
    self.zs = zs     
            
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
    d2_tangent_d_l2_cylindrical = np.zeros((nphi, 3))
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
        sign_halfhelicity = np.sign(self.helicity)
        if sign_halfhelicity==0: sign_halfhelicity =1
        self.helicity = 0.5*sign_halfhelicity
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
        #d_proptoB = False # makes d= sqrt(d_over_curvature/B0)*curvature, this is sometimes useful for half-helicity

        self.d = np.zeros(nphi)
        if not self.d_over_curvature == 0:
            if self.d_propToB == True: 
                self.d = np.sqrt(self.d_over_curvature/B0) * curvature
            else:
                self.d =self.d_over_curvature *curvature
#        elif not self.d_over_curvature_cvals == []:
#            if np.size(self.d_over_curvature_cvals) == self.nphi:
#                self.d = self.d_over_curvature_cvals * curvature
#            else:
#                self.d = np.array(sum([self.d_over_curvature_cvals[i]*np.cos(nfp*i*varphi) * curvature for i in range(len(self.d_over_curvature_cvals))]))
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
        self.d += np.array(sum([self.d_over_curvature_cvals[i]*np.cos(nfp*i*varphi)*curvature for i in range(len(self.d_over_curvature_cvals))]))
        self.d += np.array(sum([self.d_over_curvature_svals[i]*np.sin(nfp*i*varphi) *curvature*sign_curvature_change  for i in range(len(self.d_over_curvature_svals))]))
        self.d += np.array(sum([self.d_cvals[i]*np.cos(nfp*i*varphi) for i in range(len(self.d_cvals))]))
        self.d += np.array(sum([self.d_svals[i]*np.sin(nfp*i*varphi) for i in range(len(self.d_svals))]))

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
        self.B0_tripled = np.append(self.B0,self.B0,axis=0)
        self.B0_tripled = np.append(self.B0_tripled,self.B0,axis=0)
        self.B0_spline_tripled = self.convert_to_spline_tripled(self.B0_tripled)         
    # Spline interpolant for the magnetic field on-axis as a function of phi (not varphi)
    self.B0_spline = self.convert_to_spline(self.B0)

    # Spline interpolant for nu = varphi-phi
    nu = self.varphi-self.phi
    self.nu_spline = self.convert_to_spline(nu)
    self.nu_spline_of_varphi = spline(np.append(self.varphi,self.varphi[0]+2*np.pi/self.nfp), np.append(self.varphi-self.phi,self.varphi[0]-self.phi[0]), bc_type='periodic')
