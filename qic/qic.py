"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qic():
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis, convert_to_spline
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics
    from .grad_B_tensor import calculate_grad_B_tensor, calculate_grad_grad_B_tensor, \
        Bfield_cylindrical, Bfield_cartesian, grad_B_tensor_cartesian, \
        grad_grad_B_tensor_cylindrical, grad_grad_B_tensor_cartesian
    from .calculate_r2 import calculate_r2, construct_qi_r2
    from .calculate_r3 import calculate_r3
    from .mercier import mercier
    from .plot import plot, get_boundary, B_fieldline, B_contour, plot_axis
    from .r_singularity import calculate_r_singularity
    from .plot import plot, plot_boundary, get_boundary, B_fieldline, B_contour, plot_axis, B_densityplot
    from .fourier_interpolation import fourier_interpolation
    from .Frenet_to_cylindrical import Frenet_to_cylindrical, to_RZ
    from .optimize_nae import optimise_params, min_geo_qi_consistency
    from .to_vmec import to_vmec
    from .util import B_mag
    
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 B0_vals=[], B0_svals=[], d_cvals=[], d_svals=[], alpha_cvals=[0.], alpha_svals=[0.], phi_shift=0,
                 omn = False, omn_method='buffer', k_buffer=1, p_buffer=2, delta  = np.pi/5, k_second_order_SS = 0, I2=0., sG=1, spsi=1, nphi=31,
                 B2s=0., B2c=0., B2s_cvals=[], B2s_svals=[], B2c_cvals=[], B2c_svals=[], p2=0., order="r1", d_over_curvature=0, d_over_curvature_cvals = []):
        """
        Create a near-axis stellarator.
        """
        # First, force {rc, zs, rs, zc} to have the same length, for
        # simplicity.
        nfourier = np.max([len(rc), len(zs), len(rs), len(zc)])
        self.nfourier = nfourier
        self.rc = np.zeros(nfourier)
        self.zs = np.zeros(nfourier)
        self.rs = np.zeros(nfourier)
        self.zc = np.zeros(nfourier)
        self.rc[:len(rc)] = rc
        self.zs[:len(zs)] = zs
        self.rs[:len(rs)] = rs
        self.zc[:len(zc)] = zc

        # Solve for omnigenity
        self.omn = omn
        self.delta  = min(abs(delta),0.95*np.pi/nfp)
        self.omn_method = omn_method
        self.k_buffer = k_buffer
        self.p_buffer = p_buffer
        self.k_second_order_SS = k_second_order_SS
        self.d_over_curvature = d_over_curvature
        self.d_over_curvature_cvals = d_over_curvature_cvals

        # Force nphi to be odd:
        if np.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        self.nfp = nfp
        self.etabar = etabar
        self.sigma0 = sigma0
        phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        self.d_phi = phi[1] - phi[0]
        if omn==True and phi_shift==0:
            phi_shift=1/3
        self.phi_shift = phi_shift
        self.phi = phi + self.phi_shift*self.d_phi
        if B0_vals==[]:
            self.B0_vals = [B0]
            self.B0_well_depth = 0
        else:
            self.B0_vals = B0_vals
            self.B0_well_depth = B0_vals[1]
        self.B0 =  np.array(sum([self.B0_vals[i]*np.cos(nfp*i*self.phi) for i in range(len(self.B0_vals))]))
        self.B0_svals = B0_svals
        self.B0 += np.array(sum([self.B0_svals[i]*np.sin(nfp*i*self.phi) for i in range(len(self.B0_svals))]))
        if d_cvals==[] and d_svals == [] and d_over_curvature_cvals == [] and d_over_curvature == 0:
            self.d_cvals = [etabar]
            self.d_svals = d_svals
        else:
            self.d_cvals = d_cvals
            self.d_svals = d_svals
            if self.d_cvals == []:
                self.etabar = 0
                self.d_cvals = [0]
            else:
                self.etabar = d_cvals[0]
        if d_over_curvature_cvals == [] and not d_over_curvature == 0:
            self.d_over_curvature_cvals == [d_over_curvature]
        if not d_over_curvature_cvals == [] and not d_over_curvature == 0:
            raise ValueError('Only one of d_over_curvature_cvals or d_over_curvature should be specified.')
        self.d = np.array(sum([self.d_cvals[i]*np.cos(nfp*i*self.phi) for i in range(len(self.d_cvals))]))
        self.d = self.d + np.array(sum([self.d_svals[i]*np.sin(nfp*i*self.phi) for i in range(len(self.d_svals))]))
        self.alpha_cvals = alpha_cvals
        self.alpha_svals = alpha_svals
        self.alpha = np.array(sum([self.alpha_cvals[i]*np.cos(nfp*i*self.phi) for i in range(len(self.alpha_cvals))]))
        self.alpha = self.alpha + np.array(sum([self.alpha_svals[i]*np.sin(nfp*i*self.phi) for i in range(len(self.alpha_svals))]))
        self.B1s = self.B0 * self.d * np.sin(self.alpha)
        self.B1c = self.B0 * self.d * np.cos(self.alpha)
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        if B2c_cvals==[] and B2c_svals == []:
            self.B2c_cvals = [B2c]
            self.B2c_svals = B2c_svals
            self.B2c = B2c
        else:
            self.B2c_cvals = B2c_cvals
            self.B2c_svals = B2c_svals
            if self.B2c_cvals == []:
                self.B2c = 0
                self.B2c_cvals = [0]
            else:
                self.B2c = B2c_cvals[0]
        if B2s_cvals==[] and B2s_svals == []:
            self.B2s_cvals = [B2s]
            self.B2s_svals = B2s_svals
            self.B2s = B2s
        else:
            self.B2s_cvals = B2s_cvals
            self.B2s_svals = B2s_svals
            if self.B2s_cvals == []:
                self.B2s = 0
                self.B2s_cvals = [0]
            else:
                self.B2s = B2s_cvals[0]
        self.p2 = p2
        self.order = order
        self.min_R0_threshold = 0.3
        self.min_Z0_threshold = 0.3
        self._set_names()
        self.calculate()

    def change_nfourier(self, nfourier_new):
        """
        Resize the arrays of Fourier amplitudes. You can either increase
        or decrease nfourier.
        """
        rc_old = self.rc
        rs_old = self.rs
        zc_old = self.zc
        zs_old = self.zs
        index = np.min((self.nfourier, nfourier_new))
        self.rc = np.zeros(nfourier_new)
        self.rs = np.zeros(nfourier_new)
        self.zc = np.zeros(nfourier_new)
        self.zs = np.zeros(nfourier_new)
        self.rc[:index] = rc_old[:index]
        self.rs[:index] = rs_old[:index]
        self.zc[:index] = zc_old[:index]
        self.zs[:index] = zs_old[:index]
        nfourier_old = self.nfourier
        self.nfourier = nfourier_new
        self._set_names()
        # No need to recalculate if we increased the Fourier
        # resolution, only if we decreased it.
        self.calculate()

    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        self.solve_sigma_equation()
        self.r1_diagnostics()
        if self.order != 'r1':
            self.calculate_r2()
            if self.order == 'r3':
                self.calculate_r3()
    
    def get_dofs(self):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        return np.concatenate((self.rc, self.zs, self.rs, self.zc,
                               np.array([self.etabar, self.sigma0, self.B2s, self.B2c, self.p2, self.I2, self.delta]),
                               self.B0_vals, self.B0_svals, self.d_cvals, self.d_svals, self.alpha_cvals, self.alpha_svals,
                               self.B2c_cvals, self.B2c_svals, self.B2s_cvals, self.B2s_svals, self.d_over_curvature_cvals,
                               np.array([self.k_second_order_SS,self.d_over_curvature])))

    def set_dofs(self, x):
        """
        For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.
        """
        assert len(x) == self.nfourier * 4 + 7 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals)\
                       + len(self.alpha_cvals) + len(self.alpha_svals)\
                       + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + len(self.B2s_svals)\
                       + len(self.d_over_curvature_cvals) + 2
        self.rc = x[self.nfourier * 0 : self.nfourier * 1]
        self.zs = x[self.nfourier * 1 : self.nfourier * 2]
        self.rs = x[self.nfourier * 2 : self.nfourier * 3]
        self.zc = x[self.nfourier * 3 : self.nfourier * 4]
        self.etabar = x[self.nfourier * 4 + 0]
        self.sigma0 = x[self.nfourier * 4 + 1]
        self.B2s = x[self.nfourier * 4 + 2]
        self.B2c = x[self.nfourier * 4 + 3]
        self.p2 = x[self.nfourier * 4 + 4]
        self.I2 = x[self.nfourier * 4 + 5]
        self.delta = x[self.nfourier * 4 + 6]
        self.B0_vals  = x[self.nfourier * 4 + 6 + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + 1]
        self.B0_svals = x[self.nfourier * 4 + 6 + len(self.B0_vals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + 1]
        self.d_cvals  = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + 1]
        self.d_svals  = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + 1]
        self.alpha_cvals = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + 1]
        self.alpha_svals = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + 1]
        self.B2c_cvals   = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + 1]
        self.B2c_svals   = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + 1]
        self.B2s_cvals   = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + 1]
        self.B2s_svals               = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + len(self.B2s_svals) + 1]
        self.d_over_curvature_cvals  = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + len(self.B2s_svals) + 1 : self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + len(self.B2s_svals) + len(self.d_over_curvature_cvals) + 1]
        self.k_second_order_SS       = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + len(self.B2s_svals) + len(self.d_over_curvature_cvals) + 1]
        self.d_over_curvature        = x[self.nfourier * 4 + 6 + len(self.B0_vals) + len(self.B0_svals) + len(self.d_cvals) + len(self.d_svals) + len(self.alpha_cvals) + len(self.alpha_svals) + len(self.B2c_cvals) + len(self.B2c_svals) + len(self.B2s_cvals) + len(self.B2s_svals) + len(self.d_over_curvature_cvals) + 2]
        # Set new B0, d, alpha, B1 and B2
        if len(self.B0_vals)>1:
            self.B0_well_depth = self.B0_vals[1]
        self.B0 = sum([self.B0_vals[i]*np.cos(self.nfp*i*self.phi) for i in range(len(self.B0_vals))])
        self.B0_svals = self.B0_svals
        self.B0 += np.array(sum([self.B0_svals[i]*np.sin(self.nfp*i*self.phi) for i in range(len(self.B0_svals))]))
        self.d  = np.array(sum([self.d_cvals[i]*np.cos(self.nfp*i*self.phi) for i in range(len(self.d_cvals))]))
        self.d += np.array(sum([self.d_svals[i]*np.sin(self.nfp*i*self.phi) for i in range(len(self.d_svals))]))
        self.alpha  = np.array(sum([self.alpha_cvals[i]*np.cos(self.nfp*i*self.phi) for i in range(len(self.alpha_cvals))]))
        self.alpha += np.array(sum([self.alpha_svals[i]*np.sin(self.nfp*i*self.phi) for i in range(len(self.alpha_svals))]))
        self.B1s = self.B0 * self.d * np.sin(self.alpha)
        self.B1c = self.B0 * self.d * np.cos(self.alpha)
        self.calculate()
        logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))
        
    def _set_names(self):
        """
        For simsopt, sets the list of names for each degree of freedom.
        """
        names = []
        names += ['rc({})'.format(j) for j in range(self.nfourier)]
        names += ['zs({})'.format(j) for j in range(self.nfourier)]
        names += ['rs({})'.format(j) for j in range(self.nfourier)]
        names += ['zc({})'.format(j) for j in range(self.nfourier)]
        names += ['etabar', 'sigma0', 'B2s', 'B2c', 'p2', 'I2', 'delta']
        names += ['B0({})'.format(j) for j in range(len(self.B0_vals))]
        names += ['B0s({})'.format(j) for j in range(len(self.B0_svals))]
        names += ['dc({})'.format(j) for j in range(len(self.d_cvals))]
        names += ['ds({})'.format(j) for j in range(len(self.d_svals))]
        names += ['alphac({})'.format(j) for j in range(len(self.alpha_cvals))]
        names += ['alphas({})'.format(j) for j in range(len(self.alpha_svals))]
        names += ['B2cc({})'.format(j) for j in range(len(self.B2c_cvals))]
        names += ['B2cs({})'.format(j) for j in range(len(self.B2c_svals))]
        names += ['B2sc({})'.format(j) for j in range(len(self.B2s_cvals))]
        names += ['B2ss({})'.format(j) for j in range(len(self.B2s_svals))]
        names += ['d_over_curvaturec({})'.format(j) for j in range(len(self.d_over_curvature_cvals))]
        names += ['k_second_order_SS','d_over_curvature']
        self.names = names

    @classmethod
    def from_paper(cls, name, **kwargs):
        """
        Get one of the configurations that has been used in our papers.
        Available values for ``name`` are
        ``"QI"``,
        ``"r1 section 5.1"``,
        ``"r1 section 5.2"``,
        ``"r1 section 5.3"``,
        ``"r2 section 5.1"``,
        ``"r2 section 5.2"``,
        ``"r2 section 5.3"``,
        ``"r2 section 5.4"``,
        ``"r2 section 5.5"``,
        These last 5 configurations can also be obtained by specifying an integer 1-5 for ``name``.
        The configurations that begin with ``"r1"`` refer to sections in 
        Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).
        The configurations that begin with ``"r2"`` refer to sections in 
        Landreman and Sengupta, Journal of Plasma Physics 85, 815850601 (2019).
        The QI configuration refers to section 8.2 of
        Plunk, Landreman, and Helander, Journal of Plasma Physics 85, 905850602 (2019)

        You can specify any other arguments of the ``qic`` constructor
        in ``kwargs``. You can also use ``kwargs`` to override any of
        the properties of the configurations from the papers. For
        instance, you can modify the value of ``etabar`` in the first
        example using

        .. code-block::

          q = qic.Qic.from_paper('r1 section 5.1', etabar=1.1)
        """

        def add_default_args(kwargs_old, **kwargs_new):
            """
            Take any key-value arguments in ``kwargs_new`` and treat them as
            defaults, adding them to the dict ``kwargs_old`` only if
            they are not specified there.
            """
            for key in kwargs_new:
                if key not in kwargs_old:
                    kwargs_old[key] = kwargs_new[key]

                    
        if name == "r1 section 5.1":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
            add_default_args(kwargs, rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9)
                
        elif name == "r1 section 5.2":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
            add_default_args(kwargs, rc=[1, 0.265], zs=[0, -0.21], nfp=4, etabar=-2.25)
                
        elif name == "r1 section 5.3":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
            add_default_args(kwargs, rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, etabar=-1.1, sigma0=-0.6)
                
        elif name == "r2 section 5.1" or name == '5.1' or name == 1:
            """ The configuration from Landreman & Sengupta (2019), section 5.1 """
            add_default_args(kwargs, rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r3', B2c=-0.00322)
            
        elif name == "r2 section 5.2" or name == '5.2' or name == 2:
            """ The configuration from Landreman & Sengupta (2019), section 5.2 """
            add_default_args(kwargs, rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, etabar=0.632, order='r3', B2c=-0.158)
                             
        elif name == "r2 section 5.3" or name == '5.3' or name == 3:
            """ The configuration from Landreman & Sengupta (2019), section 5.3 """
            add_default_args(kwargs, rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r3', B2c=-0.7, p2=-600000.)
                             
        elif name == "r2 section 5.4" or name == '5.4' or name == 4:
            """ The configuration from Landreman & Sengupta (2019), section 5.4 """
            add_default_args(kwargs, rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, etabar=1.569, order='r3', B2c=0.1348)
                             
        elif name == "r2 section 5.5" or name == '5.5' or name == 5:
            """ The configuration from Landreman & Sengupta (2019), section 5.5 """
            add_default_args(kwargs, rc=[1, 0.3], zs=[0, 0.3], nfp=5, etabar=2.5, sigma0=0.3, I2=1.6, order='r3', B2c=1., B2s=3., p2=-0.5e7)
        elif name == "QI" or name == "QI r1 Plunk" or name == "QI Plunk":
            """ The configuration from Plunk, Landreman & Helander (2019), section 8.2 """
            rc      = [ 1.0, 0.0,-0.2 ]
            zs      = [ 0.0, 0.0, 0.35]
            B0_vals = [ 1.0, 0.1 ]
            d_svals = [ 0.0, 1.08, 0.26, 0.46]
            delta   = 0.1 * 2*np.pi
            nphi    = 151
            add_default_args(kwargs, rc=rc, zs=zs, nfp=1, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta)
        elif name == "QI r1 Jorge" or name == "QI NFP1 r1 Jorge" or name == "QI Jorge":
            """ The configuration from Jorge et al (2022) """
            rc      = [ 1.0,0.0,-0.4056622889934463,0.0,0.07747378220100756,0.0,-0.007803860877024245,0.0,0.0,0.0,0.0,0.0,0.0 ]
            zs      = [ 0.0,0.0,-0.24769666390049602,0.0,0.06767352436978152,0.0,-0.006980621303449165,0.0,-0.0006816270917189934,0.0,-1.4512784317099981e-05,0.0,-2.839050532138523e-06 ]
            B0_vals = [ 1.0,0.16915531046156507 ]
            omn_method ='non-zone'
            k_buffer = 3
            k_second_order_SS   = 0.0
            d_over_curvature   = 0.5183783762725197
            d_svals = [ 0.0,0.003563114185517955,0.0002015921485566435,-0.0012178616509882368,-0.00011629450296628697,-8.255825435616736e-07,3.2011540526397e-06 ]
            delta   = 0.1
            nfp     = 1
            nphi    = 201
            add_default_args(kwargs, omn_method = omn_method, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, d_over_curvature=d_over_curvature, k_second_order_SS=k_second_order_SS)
        elif name == "QI NFP1 r2":
            rc      = [ 1.0,0.0,-0.41599809655680886,0.0,0.08291443961920232,0.0,-0.008906891641686355,0.0,0.0,0.0,0.0,0.0,0.0 ]
            zs      = [ 0.0,0.0,-0.28721210154364263,0.0,0.08425262593215394,0.0,-0.010427621520053335,0.0,-0.0008921610906627226,0.0,-6.357200965811029e-07,0.0,2.7316247301500753e-07 ]
            rs      = [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
            zc      = [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
            sigma0  =  0.0
            B0_vals = [ 1.0,0.15824229612567256 ]
            omn_method ='non-zone'
            k_buffer = 3
            p_buffer = 2
            k_second_order_SS   = 0.0
            d_over_curvature   = 0.48654821249917474
            d_svals = [ 0.0,-0.00023993050759319644,1.6644294162908823e-05,0.00012071143120099562,-1.1664837950174757e-05,-2.443821681789672e-05,2.0922298879435957e-06 ]
            delta   = 0.1
            nfp     = 1
            B2s_svals = [ 0.0,0.27368018673599265,-0.20986698715787325,0.048031641735420336,0.07269565329289157,1.3981498114634812e-07,-9.952017662433159e-10 ]
            B2c_cvals = [ -0.0007280714400220894,0.20739775852289746,0.05816363701644946,0.06465766308954603,0.006987357785313118,1.2229700694973357e-07,-3.057497440766065e-09,0.0 ]
            B2s_cvals = [ 0.0,0.0,0.0,0.0,0.0 ]
            B2c_svals = [ 0.0,0.0,0.0,0.0 ]
            p2      =  0.0
            nphi    = 201
            add_default_args(kwargs, sigma0 = sigma0, omn_method = omn_method, p_buffer = p_buffer, k_buffer=k_buffer, rs=rs,zc=zc, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, p2=p2, order='r3', k_second_order_SS=k_second_order_SS, d_over_curvature=d_over_curvature, B2s_cvals=B2s_cvals, B2c_svals=B2c_svals)
        elif name == "QI NFP2 r2":
            rc      = [ 1.0,0.0,-0.07764451554933544,0.0,0.005284971468552636,0.0,-0.00016252676632564814,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
            zs      = [ 0.0,0.0,-0.06525233925323416,0.0,0.005858113288916291,0.0,-0.0001930489465183875,0.0,-1.21045713465733e-06,0.0,-6.6162738585035e-08,0.0,-1.8633251242689778e-07,0.0,1.4688345268925702e-07,0.0,-8.600467886165271e-08,0.0,4.172537468496238e-08,0.0,-1.099753830863863e-08 ]
            rs      = [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
            zc      = [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
            sigma0  =  0.0
            B0_vals = [ 1.0,0.12735237900304514 ]
            omn_method ='non-zone-fourier'
            k_buffer = 1
            p_buffer = 2
            k_second_order_SS   = -25.137439389881692
            d_over_curvature   = -0.14601620836497467
            d_svals = [ 0.0,-5.067489975338647,0.2759212337742016,-0.1407115065170644,0.00180521570352059,-0.03135134464554904,0.009582569807320895,-0.004780243312143034,0.002241790407060276,-0.0006738437017134619,0.00031559081192998053 ]
            delta   = 0.8
            nfp     = 2
            B2s_svals = [ 0.0,0.0012174780422017702,0.00026317725313621535,0.0002235661375254599,0.0006235230087895861,0.00021429298911807877,8.428032911991958e-05,-0.000142566391046771,-3.194627950185967e-05,-0.0001119389848119665,-6.226472957451552e-05 ]
            B2c_cvals = [ 0.0018400322140812674,-0.0013637739279265815,-0.0017961063281748597,-0.000855123667865997,-0.001412983361026517,-0.0010676686588779228,-0.0008117922713651492,-0.0002878689335032291,-0.0002515272886665927,-7.924709175875918e-05,-4.919421452969814e-05,0.0,0.0,0.0,0.0 ]
            B2s_cvals = [ 0.4445604502180231,0.13822067284200223,-0.561756934579829,0.2488873179399463,-0.14559282723014635,0.020548052084815048,-0.011070304464557718,0.004342889373034949,-0.0015730819049237866,0.0035406584522436986,0.002831887060104115,0.0,0.0,0.0,0.0 ]
            B2c_svals = [ 0.0,2.7062914673236698,-0.9151373916194634,0.021394010521077745,-0.017469913902854437,0.03186670312840335,0.021102584055813403,0.0024194864183551515,-0.0059152315287890125,0.003709416127750524,0.010027743000785166,0.0,0.0,0.0,0.0 ]
            p2      =  0.0
            nphi    = 201
            add_default_args(kwargs, sigma0 = sigma0, omn_method = omn_method, p_buffer = p_buffer, k_buffer=k_buffer, rs=rs,zc=zc, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, p2=p2, order='r3', k_second_order_SS=k_second_order_SS, d_over_curvature=d_over_curvature, B2s_cvals=B2s_cvals, B2c_svals=B2c_svals)
        elif name == "LandremanPaul2021QA" or name == "precise QA":
            """
            A fit of the near-axis model to the quasi-axisymmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=2,
                             rc=[1.0038581971135636, 0.18400998741139907, 0.021723381370503204, 0.0025968236014410812, 0.00030601568477064874, 3.5540509760304384e-05, 4.102693907398271e-06, 5.154300428457222e-07, 4.8802742243232844e-08, 7.3011320375259876e-09],
                             zs=[0.0, -0.1581148860568176, -0.02060702320552523, -0.002558840496952667, -0.0003061368667524159, -3.600111450532304e-05, -4.174376962124085e-06, -4.557462755956434e-07, -8.173481495049928e-08, -3.732477282851326e-09],
                             B0=1.006541121335688,
                             etabar=-0.6783912804454629,
                             B2c=0.26859318908803137,
                             nphi=99,
                             order='r3')

        elif name == "precise QA+well":
            """
            A fit of the near-axis model to the precise quasi-axisymmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=2,
                             rc=[1.0145598919163676, 0.2106377247598754, 0.025469267136340394, 0.0026773601516136727, 0.00021104172568911153, 7.891887175655046e-06, -8.216044358250985e-07, -2.379942694112007e-07, -2.5495108673798585e-08, 1.1679227114962395e-08, 8.961288962248274e-09],
                             zs=[0.0, -0.14607192982551795, -0.021340448470388084, -0.002558983303282255, -0.0002355043952788449, -1.2752278964149462e-05, 3.673356209179739e-07, 9.261098628194352e-08, -7.976283362938471e-09, -4.4204430633540756e-08, -1.6019372369445714e-08],
                             B0=1.0117071561808106,
                             etabar=-0.5064143402495729,
                             B2c=-0.2749140163639202,
                             nphi=99,
                             order='r3')
            
        elif name == "LandremanPaul2021QH" or name == "precise QH":
            """
            A fit of the near-axis model to the quasi-helically symmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=4,
                             rc=[1.0033608429348413, 0.19993025252481125, 0.03142704185268144, 0.004672593645851904, 0.0005589954792333977, 3.298415996551805e-05, -7.337736061708705e-06, -2.8829857667619663e-06, -4.51059545517434e-07],
                             zs=[0.0, 0.1788824025525348, 0.028597666614604524, 0.004302393796260442, 0.0005283708386982674, 3.5146899855826326e-05, -5.907671188908183e-06, -2.3945326611145963e-06, -6.87509350019021e-07],
                             B0=1.003244143729638,
                             etabar=-1.5002839921360023,
                             B2c=0.37896407142157423,
                             nphi=99,
                             order='r3')

        elif name == "precise QH+well":
            """
            A fit of the near-axis model to the precise quasi-helically symmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=4,
                             rc=[1.000474932581454, 0.16345392520298313, 0.02176330066615466, 0.0023779201451133163, 0.00014141976024376502, -1.0595894482659743e-05, -2.9989267970578764e-06, 3.464574408947338e-08],
                             zs=[0.0, 0.12501739099323073, 0.019051257169780858, 0.0023674771227236587, 0.0001865909743321566, -2.2659053455802824e-06, -2.368335337174369e-06, -1.8521248561490157e-08],
                             B0=0.999440074325872,
                             etabar=-1.2115187546668142,
                             B2c=0.6916862277166693,
                             nphi=99,
                             order='r3')
        else:
            raise ValueError('Unrecognized configuration name')

        return cls(**kwargs)

    @classmethod
    def from_cxx(cls, filename):
        """
        Load a configuration from a ``qsc_out.<extension>.nc`` output file
        that was generated by the C++ version of QSC. Almost all the
        data will be taken from the output file, over-writing any
        calculations done in python when the new Qic object is
        created.
        """
        def to_string(nc_str):
            """ Convert a string from the netcdf binary format to a python string. """
            temp = [c.decode('UTF-8') for c in nc_str]
            return (''.join(temp)).strip()
        
        f = netcdf.netcdf_file(filename, mmap=False)
        nfp = f.variables['nfp'][()]
        nphi = f.variables['nphi'][()]
        rc = f.variables['R0c'][()]
        rs = f.variables['R0s'][()]
        zc = f.variables['Z0c'][()]
        zs = f.variables['Z0s'][()]
        I2 = f.variables['I2'][()]
        B0 = f.variables['B0'][()]
        spsi = f.variables['spsi'][()]
        sG = f.variables['sG'][()]
        etabar = f.variables['eta_bar'][()]
        sigma0 = f.variables['sigma0'][()]
        order_r_option = to_string(f.variables['order_r_option'][()])
        if order_r_option == 'r2.1':
            order_r_option = 'r3'
        if order_r_option == 'r1':
            p2 = 0.0
            B2c = 0.0
            B2s = 0.0
        else:
            p2 = f.variables['p2'][()]
            B2c = f.variables['B2c'][()]
            B2s = f.variables['B2s'][()]

        q = cls(nfp=nfp, nphi=nphi, rc=rc, rs=rs, zc=zc, zs=zs,
                B0=B0, sG=sG, spsi=spsi,
                etabar=etabar, sigma0=sigma0, I2=I2, p2=p2, B2c=B2c, B2s=B2s, order=order_r_option)
        
        def read(name, cxx_name=None):
            if cxx_name is None: cxx_name = name
            setattr(q, name, f.variables[cxx_name][()])

        [read(v) for v in ['R0', 'Z0', 'R0p', 'Z0p', 'R0pp', 'Z0pp', 'R0ppp', 'Z0ppp',
                           'sigma', 'curvature', 'torsion', 'X1c', 'Y1c', 'Y1s', 'elongation']]
        if order_r_option != 'r1':
            [read(v) for v in ['X20', 'X2c', 'X2s', 'Y20', 'Y2c', 'Y2s', 'Z20', 'Z2c', 'Z2s', 'B20']]
            if order_r_option != 'r2':
                [read(v) for v in ['X3c1', 'Y3c1', 'Y3s1']]
                    
        f.close()
        return q
        
    def min_R0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(R0) < min_R0_constraint.
        """
        return np.max((0, self.min_R0_threshold - self.min_R0)) ** 2

    def min_Z0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(Z0) < min_Z0_constraint.
        """
        return np.max((0, self.min_Z0_threshold - self.min_Z0)) ** 2
        
    @classmethod
    def from_boozxform(cls, booz_xform_file, max_s_for_fit = 0.4, N_phi = 200, max_n_to_plot = 2, show=False,
                         vmec_file=None, rc=[], rs=[], zc=[], zs=[], nNormal=None, input_stel=None, savefig=False):#, order='r2', sigma0=0, I2=0, p2=0, omn=False):
        """
        Load a configuration from a VMEC and a BOOZ_XFORM output files
        """
        # Read properties of BOOZ_XFORM output file
        f = netcdf.netcdf_file(booz_xform_file,'r',mmap=False)
        bmnc = f.variables['bmnc_b'][()]
        ixm = f.variables['ixm_b'][()]
        ixn = f.variables['ixn_b'][()]
        jlist = f.variables['jlist'][()]
        ns = f.variables['ns_b'][()]
        nfp = f.variables['nfp_b'][()]
        Psi = f.variables['phi_b'][()]
        Psi_a = np.abs(Psi[-1])
        iotaVMECt=f.variables['iota_b'][()][1]
        f.close()

        if vmec_file!=None:
            # Read axis-shape from VMEC output file
            f = netcdf.netcdf_file(vmec_file,'r',mmap=False)
            am = f.variables['am'][()]
            rc = f.variables['raxis_cc'][()]
            zs = -f.variables['zaxis_cs'][()]
            try:
                rs = -f.variables['raxis_cs'][()]
                zc = f.variables['zaxis_cc'][()]
                logger.info('Non stellarator symmetric configuration')
            except:
                rs=[]
                zc=[]
                logger.info('Stellarator symmetric configuration')
            f.close()
        elif rc!=[]:
            # Read axis-shape from input parameters
            rc=rc
            rs=rs
            zc=zc
            zs=zs
        else:
            print("Axis shape not specified")
            # Calculate nNormal
        if nNormal==None:
            stel = Qic(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp)
            nNormal = stel.iotaN - stel.iota
        else:
            nNormal = nNormal

        # Prepare coordinates for fit
        s_full = np.linspace(0,1,ns)
        ds = s_full[1] - s_full[0]
        #s_half = s_full[1:] - 0.5*ds
        s_half = s_full[jlist-1] - 0.5*ds
        mask = s_half < max_s_for_fit
        s_fine = np.linspace(0,1,400)
        sqrts_fine = s_fine
        phi = np.linspace(0,2*np.pi / nfp, N_phi)
        B0  = np.zeros(N_phi)
        B1s = np.zeros(N_phi)
        B1c = np.zeros(N_phi)
        B20 = np.zeros(N_phi)
        B2s = np.zeros(N_phi)
        B2c = np.zeros(N_phi)

        # Perform fit
        numRows=3
        numCols=max_n_to_plot*2+1
        if show: fig=plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
        for jmn in range(len(ixm)):
            m = ixm[jmn]
            n = ixn[jmn] / nfp
            if m>2:
                continue
            doplot = (np.abs(n) <= max_n_to_plot) & show
            row = m
            col = n+max_n_to_plot
            if doplot:
                plt.subplot(int(numRows),int(numCols),int(row*numCols + col + 1))
                plt.plot(np.sqrt(s_half), bmnc[:,jmn],'.-')
                # plt.xlabel(r'$\sqrt{s}$')
                plt.title('bmnc(m='+str(m)+' n='+str(n)+')')
            if m==0:
                # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
                degree = 4
                p = np.polyfit(s_half[mask], bmnc[mask,jmn], degree)
                B0 += p[-1] * np.cos(n*nfp*phi)
                B20 += p[-2] * np.cos(n*nfp*phi)
                if doplot:
                    plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
            if m==1:
                # For m=1, fit a polynomial in sqrt(s) to an odd function
                x1 = np.sqrt(s_half[mask])
                y1 = bmnc[mask,jmn]
                x2 = np.concatenate((-x1,x1))
                y2 = np.concatenate((-y1,y1))
                degree = 5
                p = np.polyfit(x2,y2, degree)
                B1c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
                B1s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
                if doplot:
                    plt.plot(sqrts_fine, np.polyval(p, sqrts_fine),'r')
            if m==2:
                # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
                x1 = s_half[mask]
                y1 = bmnc[mask,jmn]
                degree = 4
                p = np.polyfit(x1,y1, degree)
                B2c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
                B2s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
                if doplot:
                    plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
        if show:
            plt.show()
        # Convert expansion in sqrt(s) to an expansion in r
        BBar = np.mean(B0)
        sqrt_s_over_r = np.sqrt(np.pi * BBar / Psi_a)
        B1s *= sqrt_s_over_r
        B1c *= -sqrt_s_over_r
        B20 *= sqrt_s_over_r*sqrt_s_over_r
        B2c *= sqrt_s_over_r*sqrt_s_over_r
        B2s *= sqrt_s_over_r*sqrt_s_over_r
        eta_bar = np.mean(B1c) / BBar

        # NEEDS A WAY TO READ I2 FROM VMEC OR BOOZ_XFORM

        # if p2==0 and vmec_file!=None:
        #     r  = np.sqrt(Psi_a/(np.pi*BBar))
        #     p2 = - am[0]/r/r

        # if omn:
        #     if order=='r1':
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0,I2=I2)
        #     else:
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0,I2=I2, B2c=np.mean(B2c), B2s=np.mean(B2s), order=order, p2=p2)
        # else:
        #     if order=='r1':
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0, I2=I2)
        #     else:
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0, I2=I2, B2c=np.mean(B2c), B2s=np.mean(B2s), order=order, p2=p2)

        # q.B0_boozxform_array=B0
        # q.B1c_boozxform_array=B1c
        # q.B1s_boozxform_array=B1s
        # q.B20_boozxform_array=B20
        # q.B2c_boozxform_array=B2c
        # q.B2s_boozxform_array=B2s
        # q.iotaVMEC = iotaVMECt
        try:
            name = vmec_file[5:-3]

            figB0=plt.figure(figsize=(5, 5), dpi=80)
            plt.plot(input_stel.phi, input_stel.B0, 'r--', label=r'$B_0$ Near-axis')
            plt.plot(input_stel.phi, B0,            'b-' , label=r'$B_0$ VMEC')
            plt.xlabel(r'$\phi$', fontsize=18)
            plt.legend(fontsize=14)
            if savefig: figB0.savefig('B0_VMEC'+name+'.pdf')

            figB1=plt.figure(figsize=(5, 5), dpi=80)
            plt.plot(input_stel.phi, input_stel.B1c, 'r--', label=r'$B_{1c}$ Near-axis')
            plt.plot(input_stel.phi, B1c,            'r-' , label=r'$B_{1c}$ VMEC')
            plt.plot(input_stel.phi, input_stel.B1s, 'b--', label=r'$B_{1s}$ Near-axis')
            plt.plot(input_stel.phi, B1s,            'b-' , label=r'$B_{1s}$ VMEC')
            plt.xlabel(r'$\phi$', fontsize=18)
            plt.legend(fontsize=14)
            if savefig: figB1.savefig('B1_VMEC'+name+'.pdf')

            figB2=plt.figure(figsize=(5, 5), dpi=80)
            if input_stel.order != 'r1':
                plt.plot(input_stel.phi, input_stel.B20, 'r--', label=r'$B_{20}$ Near-axis')
                plt.plot(input_stel.phi, input_stel.B2c_array, 'b--', label=r'$B_{2c}$ Near-axis')
                plt.plot(input_stel.phi, input_stel.B2s_array, 'g--', label=r'$B_{2s}$ Near-axis')
            plt.plot(input_stel.phi, B20,            'r-' , label=r'$B_{20}$ VMEC')
            plt.plot(input_stel.phi, B2c,            'b-' , label=r'$B_{2c}$ VMEC')
            plt.plot(input_stel.phi, B2s,            'g-' , label=r'$B_{2s}$ VMEC')
            plt.xlabel(r'$\phi$', fontsize=18)
            plt.legend(fontsize=14)
            if savefig: figB2.savefig('B2_VMEC'+name+'.pdf')

            if show: plt.show()
        except Exception as e:
            print(e)

        plt.close(figB0)
        plt.close()

        return [B0,B1c,B1s,B20,B2c,B2s,iotaVMECt]