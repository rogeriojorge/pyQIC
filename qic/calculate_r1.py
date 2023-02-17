"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import logging
import numpy as np
from .util import fourier_minimum
from .newton import newton
from scipy.interpolate import CubicSpline as spline

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _residual(self, x):
    """
    Residual in the sigma equation, used for Newton's method.  x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x[1::])
    iota = x[0]
    if self.omn == True:
        helicity = - self.helicity
        self.alpha_no_buffer = np.pi*(2*helicity+1/2) + iota * (self.varphi - np.pi/self.nfp)
        self.alpha_notIota = 0 * self.varphi # added varphi term to get proper sized array
        self.alpha_iota = self.varphi - np.pi/self.nfp

        if self.omn_method == 'buffer':
            location_section_I_II   = np.argmin(np.abs(self.varphi-self.delta))
            location_section_II_III = np.argmin(np.abs(self.varphi-(2*np.pi/self.nfp-self.delta)))+1
            varphiI   = self.varphi[0:location_section_I_II]
            varphiIII = self.varphi[location_section_II_III::]

            # Calculate alpha_iota on buffer regions
            alpha_I_II = self.delta - np.pi/self.nfp
            alpha_0 = 0
            alpha_1 = -(2*alpha_0 - 2*alpha_I_II + 1*self.delta) / self.delta
            alpha_3 = 2*(alpha_0 - alpha_I_II + 1*self.delta) / (self.delta * self.delta * self.delta)
            alpha_4 = -(alpha_0 - alpha_I_II + 1*self.delta) / (self.delta * self.delta * self.delta * self.delta)
            self.alpha_iota[0:location_section_I_II]   =   alpha_0 + alpha_1 * varphiI + alpha_3 * (varphiI **3) + alpha_4 * (varphiI **4)
            self.alpha_iota[location_section_II_III::] = -(alpha_0 + alpha_1 * (2*np.pi/self.nfp-varphiIII) + alpha_3 * ((2*np.pi/self.nfp-varphiIII) **3) + alpha_4 * ((2*np.pi/self.nfp-varphiIII) **4))

            alpha_shift = np.pi*(2*helicity+1/2)
            n_for_alpha = helicity

            # Calculate alpha_notIota on buffer regions
            alpha_I_II = 0
            alpha_0 =  - np.pi * n_for_alpha
            alpha_1 = -(2*alpha_0 - 2*alpha_I_II + 0*self.delta) / self.delta
            alpha_3 = 2*(alpha_0 - alpha_I_II + 0*self.delta) / (self.delta * self.delta * self.delta) 
            alpha_4 = -(alpha_0 - alpha_I_II + 0*self.delta) / (self.delta * self.delta * self.delta * self.delta)
            self.alpha_notIota[0:location_section_I_II] =     alpha_0 + alpha_1 * varphiI + alpha_3 * (varphiI **3) + alpha_4 * (varphiI **4)
            self.alpha_notIota[location_section_II_III::] = -(alpha_0 + alpha_1 * (2*np.pi/self.nfp-varphiIII) + alpha_3 * ((2*np.pi/self.nfp-varphiIII) **3) + alpha_4 * ((2*np.pi/self.nfp-varphiIII) **4))
            self.alpha_notIota = self.alpha_notIota + alpha_shift

            self.alpha_at_zero = alpha_shift + alpha_0

            self.d_alpha_iota_d_varphi = np.matmul(self.d_d_varphi, self.alpha_iota)
            self.d_alpha_notIota_d_varphi = n_for_alpha * self.nfp + np.matmul(self.d_d_varphi, self.alpha_notIota - self.varphi * n_for_alpha * self.nfp) # We have to treat the secular part separately here since d_d_varphi assumes periodicity
            
        elif self.omn_method == 'non-zone':
            k = self.k_buffer
            self.alpha_iota += -(np.pi / self.nfp) * (self.varphi*self.nfp/np.pi - 1)**(2*k+1)
            self.alpha_notIota += np.pi*(2*helicity + 1/2 + helicity * (self.varphi*self.nfp/np.pi - 1)**(2*k+1))

            self.d_alpha_iota_d_varphi = 1 - (np.pi / self.nfp) * (2*k+1) * (self.nfp/np.pi) * (self.varphi*self.nfp/np.pi - 1)**(2*k)
            self.d_alpha_notIota_d_varphi = np.pi * helicity * (2*k+1) * (self.nfp/np.pi) * (self.varphi*self.nfp/np.pi - 1)**(2*k)

        elif self.omn_method == 'non-zone-smoother':
            k = self.k_buffer
            p = self.p_buffer
            nu = (2*k+1)*(2*k)/((2*p+1)*(2*p))
            m = helicity
            n = self.nfp
            a_not_iota =  ((np.pi * m) * (np.pi/n)**(-2*k-1)) * (1/(1-nu))
            a_iota     = -((np.pi / n) * (np.pi/n)**(-2*k-1)) * (1/(1-nu))
            b_not_iota = -((np.pi * m) * (np.pi/n)**(-2*p-1)) * (nu/(1-nu))
            b_iota     =  ((np.pi / n) * (np.pi/n)**(-2*p-1)) * (nu/(1-nu))
            self.alpha_iota    += a_iota     * (self.varphi-np.pi/self.nfp)**(2*k+1) + b_iota     * (self.varphi-np.pi/self.nfp)**(2*p+1)
            self.alpha_notIota += a_not_iota * (self.varphi-np.pi/self.nfp)**(2*k+1) + b_not_iota * (self.varphi-np.pi/self.nfp)**(2*p+1)
            self.alpha_notIota += np.pi*(2*helicity + 1/2)

            self.d_alpha_iota_d_varphi    = 1 + a_iota     * (2*k+1) * (self.varphi-np.pi/self.nfp)**(2*k) + b_iota     * (2*p+1) * (self.varphi-np.pi/self.nfp)**(2*p)
            self.d_alpha_notIota_d_varphi =     a_not_iota * (2*k+1) * (self.varphi-np.pi/self.nfp)**(2*k) + b_not_iota * (2*p+1) * (self.varphi-np.pi/self.nfp)**(2*p)

        elif self.omn_method == 'non-zone-fourier':
            x = self.varphi
            Pi = np.pi
            if self.nfp==1:
                if self.k_buffer==1:
                    self.alpha_iota = -(125*(1728*np.sin(x) + 216*np.sin(2*x) + 64*np.sin(3*x) + 27*np.sin(4*x)) + 1728*np.sin(5*x))/(18000*np.pi*np.pi)
                    self.alpha_notIota =  -(125*(72*np.pi*np.pi*(np.pi + 2*x) + 1728*np.sin(x) + 216*np.sin(2*x) + 64*np.sin(3*x) + 27*np.sin(4*x)) + 1728*np.sin(5*x))/(18000*np.pi*np.pi)
                elif self.k_buffer==3:
                    self.alpha_iota = (-21*(8*(120 - 20*Pi**2 + Pi**4) + (15 + 2*Pi**2*(-5 + Pi**2))*np.cos(x))*np.sin(x))/(2.*Pi**6) - \
                        (28*(40 - 60*Pi**2 + 27*Pi**4)*np.sin(3*x))/(243.*Pi**6) - \
                        (21*(15 - 40*Pi**2 + 32*Pi**4)*np.sin(4*x))/(512.*Pi**6) - \
                        (84*(24 + 25*Pi**2*(-4 + 5*Pi**2))*np.sin(5*x))/(15625.*Pi**6)
                    self.alpha_notIota =          -Pi/2. - x - (84*(120 - 20*Pi**2 + Pi**4)*np.sin(x))/Pi**6 - \
                        (21*(15 + 2*Pi**2*(-5 + Pi**2))*np.sin(2*x))/(4.*Pi**6) - \
                        (28*(40 - 60*Pi**2 + 27*Pi**4)*np.sin(3*x))/(243.*Pi**6) - \
                        (21*(15 - 40*Pi**2 + 32*Pi**4)*np.sin(4*x))/(512.*Pi**6) - \
                        (84*(24 + 25*Pi**2*(-4 + 5*Pi**2))*np.sin(5*x))/(15625.*Pi**6)
                else: 
                    logging.raiseExceptions("Not implemented yet")
            elif self.nfp==2:
                if self.k_buffer==1:
                    self.alpha_iota = -(6*np.sin(self.nfp*x) + 3*np.sin(2*self.nfp*x)/4 + 2*np.sin(3*self.nfp*x)/9 + 3*np.sin(4*self.nfp*x)/32 + 6*np.sin(5*self.nfp*x)/125) / (np.pi * np.pi)
                    self.alpha_notIota = helicity * self.nfp * x + np.pi/2 * (1 + 2*helicity) - helicity * 2 * self.alpha_iota
                else: 
                    logging.raiseExceptions("Not implemented yet")
            elif self.nfp==3:
                if self.k_buffer==1:
                    self.alpha_iota = -(4*np.sin(self.nfp*x) + np.sin(2*self.nfp*x)/2 + 4*np.sin(3*self.nfp*x)/27 + np.sin(4*self.nfp*x)/16 + 4*np.sin(5*self.nfp*x)/125) / (np.pi * np.pi)
                    self.alpha_notIota = helicity * self.nfp * x + np.pi/2 * (1 + 2*helicity) - helicity * 2 * self.alpha_iota
                else: 
                    logging.raiseExceptions("Not implemented yet")
            else: 
                logging.raiseExceptions("Not implemented yet")
            self.d_alpha_iota_d_varphi = np.matmul(self.d_d_varphi, self.alpha_iota)
            self.d_alpha_notIota_d_varphi = helicity * self.nfp + np.matmul(self.d_d_varphi, self.alpha_notIota - self.varphi * helicity * self.nfp)

        # Calculate alpha
        self.alpha = self.alpha_iota * iota + self.alpha_notIota

        # Calculate gamma
        self.gamma_iota    = 1 - self.d_alpha_iota_d_varphi
        self.gamma_notIota = - self.d_alpha_notIota_d_varphi
        self.gamma         = self.gamma_iota * iota + self.gamma_notIota
    else:
        self.gamma = iota + self.helicity * self.nfp - np.matmul(self.d_d_varphi, self.alpha)
    r = np.matmul(self.d_d_varphi, sigma) + self.gamma * \
        (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * self.etabar_squared_over_curvature_squared * (-self.spsi * self.torsion + self.I2 / self.Bbar) * self.G0 / self.B0
    #logger.debug("_residual called with x={}, r={}".format(x, r))

    sigma_at_0 = self.interpolateTo0 @ sigma
    return np.append(r,sigma_at_0-self.sigma0)

def _jacobian(self, x):
    """
    Compute the Jacobian matrix for solving the sigma equation. x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x[1::])

    # d (Riccati equation) / d sigma:
    jac = np.copy(self.d_d_varphi)
    for j in range(self.nphi):
        jac[j, j] += self.gamma [j] * 2 * sigma[j]

    # d (Riccati equation) / d iota:
    if self.omn == True:
        gamma_iota = self.gamma_iota
    else:
        gamma_iota = 1
    jac = np.append(np.transpose([gamma_iota * (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma)]),jac,axis=1)

    # d (sigma[0]-sigma0) / dsigma:
    jac = np.append(jac,[np.append(0,self.interpolateTo0)],axis=0)

    #logger.debug("_jacobian called with x={}, jac={}".format(x, jac))
    return jac

def solve_sigma_equation(self):
    """
    Solve the sigma equation.
    """
    x0 = np.full(self.nphi+1, self.sigma0)
    x0[0] = 0 # Initial guess for iota
    """
    soln = scipy.optimize.root(self._residual, x0, jac=self._jacobian, method='lm')
    self.iota = soln.x[0]
    self.sigma = np.copy(soln.x)
    self.sigma[0] = self.sigma0
    """
    self.sigma = newton(self._residual, x0, jac=self._jacobian)
    self.iota = self.sigma[0]
    self.iotaN = self.iota + self.helicity * self.nfp
    self.sigma = self.sigma[1::]

def _determine_helicity(self):
    """
    Determine the integer N associated with the type of quasisymmetry
    by counting the number of times the normal vector rotates
    poloidally as you follow the axis around toroidally.
    """
    quadrant = np.zeros(self.nphi + 1)
    for j in range(self.nphi):
        if self.normal_cylindrical[j,0] >= 0:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3
    quadrant[self.nphi] = quadrant[0]

    counter = 0
    for j in range(self.nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
    counter *= self.spsi * self.sG
    self.helicity = counter / 4

def r1_diagnostics(self):
    """
    Compute various properties of the O(r^1) solution, once sigma and
    iota are solved for.
    """
    # Spline interpolant for the first order components of the magnetic field
    # as a function of phi, not varphi
    self.d_spline = self.convert_to_spline(self.d)
    if self.omn == True:
        self.alpha_tilde = self.alpha#-self.N_helicity*self.varphi
        self.cos_alpha_tilde_spline = self.convert_to_spline(np.cos(self.alpha_tilde))
        self.sin_alpha_tilde_spline = self.convert_to_spline(np.sin(self.alpha_tilde))

        angle = self.alpha - (-self.helicity * self.nfp * self.varphi)
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        B1sQI    = 0    # B=B0*(1+d cos(theta-alpha)), B1cQI = B0*d
        B1cQI    = self.B0 * self.d
        self.B1c = (B1cQI * cosangle - B1sQI * sinangle)
        self.B1s = (B1sQI * cosangle + B1cQI * sinangle)
    else:
        self.alpha_tilde = self.alpha
        self.cos_alpha_tilde_spline = self.convert_to_spline(np.cos(self.alpha))
        self.sin_alpha_tilde_spline = self.convert_to_spline(np.sin(self.alpha))

    self.B1c_spline = self.convert_to_spline(self.B1c)
    self.B1s_spline = self.convert_to_spline(self.B1s)
    # X = X1c*cos(theta-N*varphi)+X1s*sin(theta-N*varphi)
    # theta=0 -> X(theta=0)=X1c*cos(N*varphi)-X1s*sin(N*varphi)
    # theta=0 -> Y(theta=0)=Y1c*cos(N*varphi)-Y1s*sin(N*varphi)
    self.X1c = self.B1c / (self.curvature * self.B0)# * (self.sign_curvature_change)
    self.X1s = self.B1s / (self.curvature * self.B0)# * (self.sign_curvature_change)
    self.Y1s = self.sG * self.Bbar * self.curvature * ( self.B1c + self.B1s * self.sigma) / (self.B1c * self.B1c + self.B1s * self.B1s)# * (self.sign_curvature_change)# + 1e-30) 
    self.Y1c = self.sG * self.Bbar * self.curvature * (-self.B1s + self.B1c * self.sigma) / (self.B1c * self.B1c + self.B1s * self.B1s)# * (self.sign_curvature_change)# + 1e-30)

    # self.X1c = self.convert_to_spline(self.B1c / (self.curvature * self.B0))(self.phi - self.phi_shift*self.d_phi)
    # self.X1s = self.convert_to_spline(self.B1s / (self.curvature * self.B0))(self.phi - self.phi_shift*self.d_phi)
    # self.Y1s = self.convert_to_spline(self.sG * self.Bbar * self.curvature * ( self.B1c + self.B1s * self.sigma) / (self.B1c * self.B1c + self.B1s * self.B1s))(self.phi - self.phi_shift*self.d_phi)
    # self.Y1c = self.convert_to_spline(self.sG * self.Bbar * self.curvature * (-self.B1s + self.B1c * self.sigma) / (self.B1c * self.B1c + self.B1s * self.B1s))(self.phi - self.phi_shift*self.d_phi)
    
    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
    if self.helicity == 0:
        self.X1s_untwisted = self.X1s
        self.X1c_untwisted = self.X1c
        self.Y1s_untwisted = self.Y1s
        self.Y1c_untwisted = self.Y1c
    else:
        # import matplotlib.pyplot as plt
        # plt.plot(self.varphi)
        # plt.plot(self.phi - self.phi_shift*self.d_phi + self.nu_spline(self.phi - self.phi_shift*self.d_phi))
        # plt.show()
        # exit()
        # angle = -self.helicity * self.nfp * (self.phi - self.phi_shift*self.d_phi + self.nu_spline(self.phi - self.phi_shift*self.d_phi))
        # print(self.phi - self.phi_shift*self.d_phi + self.nu_spline(self.phi - self.phi_shift*self.d_phi) - self.varphi)
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X1s_untwisted = (self.X1s *   cosangle  + self.X1c * sinangle)# * self.sign_curvature_change
        self.X1c_untwisted = (self.X1s * (-sinangle) + self.X1c * cosangle)# * self.sign_curvature_change
        self.Y1s_untwisted = (self.Y1s *   cosangle  + self.Y1c * sinangle)# * self.sign_curvature_change
        self.Y1c_untwisted = (self.Y1s * (-sinangle) + self.Y1c * cosangle)# * self.sign_curvature_change
    # Use (R,Z) for elongation in the (R,Z) plane,
    # or use (X,Y) for elongation in the plane perpendicular to the magnetic axis.
    p = self.X1s * self.X1s + self.X1c * self.X1c + self.Y1s * self.Y1s + self.Y1c * self.Y1c
    q = self.X1s * self.Y1c - self.X1c * self.Y1s
    self.elongation = (p + np.sqrt(p * p - 4 * q * q)) / (2 * np.abs(q))
    self.mean_elongation = np.sum(self.elongation * self.d_l_d_phi) / np.sum(self.d_l_d_phi)
    index = np.argmax(self.elongation)
    self.max_elongation = -fourier_minimum(-self.elongation)
    # Area of the ellipse in the plane perpendicular to the magnetic axis
    self.ellipse_area = np.pi * self.sG * self.Bbar / self.B0 + 2 * (self.X1c * self.Y1c - self.X1s * self.Y1s)

    self.d_X1c_d_varphi = np.matmul(self.d_d_varphi, self.X1c)
    self.d_X1s_d_varphi = np.matmul(self.d_d_varphi, self.X1s)
    self.d_Y1s_d_varphi = np.matmul(self.d_d_varphi, self.Y1s)
    self.d_Y1c_d_varphi = np.matmul(self.d_d_varphi, self.Y1c)

    self.calculate_grad_B_tensor()


