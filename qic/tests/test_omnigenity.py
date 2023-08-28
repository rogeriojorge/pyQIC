#!/usr/bin/env python3

import unittest
from matplotlib.pyplot import savefig
import numpy as np
from qic.qic import Qic
import logging
import os, sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewtonTests(unittest.TestCase):

    def test_paper_III(self):
        """
        Compare with Paper III
        """
        # rc      = [ 1.0, 0.0,-0.2 ]
        # zs      = [ 0.0, 0.0, 0.35]
        # B0_vals = [ 1.0, 0.1 ]
        # d_svals = [ 0.0, 1.08, 0.26, 0.46]
        # delta   = 0.1 * 2*np.pi
        # nphi    = 151
        # stel = Qic(rc=rc,zs=zs, nfp=1, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta)
        stel = Qic.from_paper("QI")
        print('iota  =', stel.iota)
        print('max elongation  =', stel.max_elongation)
        print('mean elongation =', stel.mean_elongation)

        logger.info('Comparing to matlab from Paper III')
        matlab_results_folder = "matlab_data/"
        abs_filename_matlab_results_folder = os.path.join(os.path.dirname(__file__), matlab_results_folder)
        def compare_field(field, py_field, rtol=1e-12, atol=1e-11):
            data = np.genfromtxt(abs_filename_matlab_results_folder+field+".txt")
            if field=='X1s' or field=='X1c' or field=='Y1s' or field=='Y1c':
                py_field = py_field[int(len(data)/2)+1:]
                data = data[int(len(data)/2)+1:]
            logger.info('max difference in {}: {}'.format(field, np.max(np.abs(data - py_field))))
            np.testing.assert_allclose(data, py_field, rtol=rtol, atol=atol)

        np.testing.assert_allclose(stel.iota, 0.716646377954308)

        compare_field('torsion', stel.torsion)
        compare_field('curvature', stel.curvature * stel.sign_curvature_change)
        compare_field('curvature_alt', stel.curvature * stel.sign_curvature_change)
        compare_field('d_l_d_phi', stel.d_l_d_phi)
        compare_field('d2_l_d_phi2', stel.d2_l_d_phi2)
        compare_field('d_d_varphi', stel.d_d_varphi)
        compare_field('phi', stel.phi)
        compare_field('varphi', stel.varphi)
        compare_field('d', stel.d)
        compare_field('B0', stel.B0)
        compare_field('axis_length', stel.axis_length)
        compare_field('G0', stel.G0)
        compare_field('d_r_d_phi_cylindrical', stel.d_r_d_phi_cylindrical)
        compare_field('d2_r_d_phi2_cylindrical', stel.d2_r_d_phi2_cylindrical)
        compare_field('d3_r_d_phi3_cylindrical', stel.d3_r_d_phi3_cylindrical)
        compare_field('d_phi', stel.d_phi)
        compare_field('alpha_iota', stel.alpha_iota)
        compare_field('alpha_notIota', stel.alpha_notIota-3*np.pi)
        compare_field('d_alpha_iota_d_varphi', stel.d_alpha_iota_d_varphi)
        compare_field('d_alpha_notIota_d_varphi', stel.d_alpha_notIota_d_varphi)
        compare_field('sigma', stel.sigma)
        compare_field('alpha', stel.alpha-3*np.pi)
        # Need to untwist B, X and Y
        compare_field('B1c_over_B0', -(stel.B1s * -np.sin(stel.varphi) + stel.B1c * np.cos(stel.varphi)) / stel.B0)
        compare_field('B1s_over_B0', -(stel.B1s *  np.cos(stel.varphi) + stel.B1c * np.sin(stel.varphi)) / stel.B0)
        compare_field('X1s', stel.X1s_untwisted)
        compare_field('X1c', stel.X1c_untwisted)
        compare_field('Y1c', stel.Y1c_untwisted)
        compare_field('Y1s', stel.Y1s_untwisted)

    def test_multiple_NFP(self):
        """
        Create multiple NFP configurations
        """
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
        nphi    =  301
        stel    =  Qic(sigma0 = sigma0, omn_method = omn_method, p_buffer = p_buffer, k_buffer=k_buffer, rs=rs,zc=zc, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, p2=p2, order='r3', k_second_order_SS=k_second_order_SS, d_over_curvature=d_over_curvature, B2s_cvals=B2s_cvals, B2c_svals=B2c_svals)
        iota    =  -0.718394753879415

        print('iota  =', stel.iota)
        print('max elongation  =', stel.max_elongation)
        print('mean elongation =', stel.mean_elongation)
        print('N_helicity =', stel.N_helicity)
        # stel.plot_axis()
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(stel.alpha_notIota, label='alpha not iota')
        # plt.legend()
        # plt.figure()
        # plt.plot(stel.alpha_iota, label='alpha iota')
        # plt.legend()
        # plt.figure()
        # plt.plot(stel.alpha, label='alpha')
        # plt.plot(stel.alpha_no_buffer, label='alpha no buffer')
        # plt.legend()
        # plt.show()

    def test_non_zone(self):
        rc      = [ 1.0, 0.0,-0.2 ]
        zs      = [ 0.0, 0.0, 0.35]
        B0_vals = [ 1.0, 0.1 ]
        d_svals = [ 0.0, 1.08, 0.26, 0.46]
        delta   = 0.1 * 2*np.pi
        nphi    = 151
        stel = Qic(omn_method = 'non-zone', rc=rc,zs=zs, nfp=1, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta)
        print('iota  =', stel.iota)
        print('max elongation  =', stel.max_elongation)
        print('mean elongation =', stel.mean_elongation)

if __name__ == "__main__":
    unittest.main()
