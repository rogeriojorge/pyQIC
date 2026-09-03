#!/usr/bin/env python3

import unittest

import numpy as np

from qic.qic import Qic


class DOverCurvatureTests(unittest.TestCase):

    rc = [1.0, 0.0, -0.3, 0.0, 0.01, 0.0, 0.001]
    zs = [0.0, 0.0, -0.2, 0.0, 0.01, 0.0, 0.001]
    cvals = [0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    def make_qic(self, **kwargs):
        return Qic(
            omn_method='non-zone-fourier',
            delta=0.1,
            p_buffer=2,
            k_buffer=1,
            rc=self.rc,
            zs=self.zs,
            nfp=1,
            B0_vals=[1.0, 0.16],
            nphi=31,
            omn=True,
            order='r1',
            **kwargs
        )

    def test_set_dofs_with_d_over_curvature_cvals(self):
        stel = self.make_qic(d_over_curvature_cvals=self.cvals)
        d_before = stel.d.copy()

        stel.set_dofs(stel.get_dofs())

        np.testing.assert_allclose(stel.d_over_curvature_cvals, self.cvals)
        np.testing.assert_allclose(stel.d, d_before)

    def test_scalar_d_over_curvature_initializes_cvals(self):
        stel = self.make_qic(d_over_curvature=0.5)

        self.assertEqual(stel.d_over_curvature_cvals, [0.5])

    def test_construct_qi_r2_initializes_cvals(self):
        stel = self.make_qic()
        expected = np.array([0.5, 0.0, 0.0, 0.0])
        expected_params = ['d_over_curvaturec({})'.format(j) for j in range(4)]

        class OptimizationReached(Exception):
            pass

        def check_initialized_cvals(params, **kwargs):
            np.testing.assert_allclose(stel.d_over_curvature_cvals, expected)
            self.assertEqual(params, expected_params)
            raise OptimizationReached

        stel.calculate = lambda: None
        stel.optimise_params = check_initialized_cvals
        with self.assertRaises(OptimizationReached):
            stel.construct_qi_r2()

    def test_construct_qi_r2_pads_short_cvals(self):
        stel = self.make_qic(d_over_curvature_cvals=[0.5, 0.1])
        expected = np.array([0.5, 0.1, 0.0, 0.0])
        expected_params = ['d_over_curvaturec({})'.format(j) for j in range(4)]

        class OptimizationReached(Exception):
            pass

        def check_padded_cvals(params, **kwargs):
            np.testing.assert_allclose(stel.d_over_curvature_cvals, expected)
            self.assertEqual(params, expected_params)
            raise OptimizationReached

        stel.calculate = lambda: None
        stel.optimise_params = check_padded_cvals
        with self.assertRaises(OptimizationReached):
            stel.construct_qi_r2()

    def test_numpy_coefficient_inputs(self):
        stel = Qic(
            rc=np.asarray(self.rc),
            zs=np.asarray(self.zs),
            B0_vals=np.array([1.0, 0.16]),
            d_cvals=np.array([0.5]),
            d_svals=np.array([]),
            d_over_curvature_cvals=np.array([]),
            B2c_cvals=np.array([]),
            B2c_svals=np.array([]),
            B2s_cvals=np.array([]),
            B2s_svals=np.array([]),
            nphi=31,
        )

        np.testing.assert_allclose(stel.B0_vals, [1.0, 0.16])


if __name__ == '__main__':
    unittest.main()
