"""Gauge and tensor checks independent of the VMEC integration tests."""
import unittest
import numpy as np
from qic.qic import Qic
from qic.calculate_r2 import _solve_periodic_beta_0
from qic.spectral_diff_matrix import spectral_diff_matrix


class PeriodicBetaGaugeTests(unittest.TestCase):
    def test_exactly_singular_periodic_derivative(self):
        derivative = np.array([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])
        # Row scaling represents a nonuniform Boozer-to-grid angle map.
        derivative *= np.array([1., 1.2, .7])[:, None]
        expected = np.array([0., 1., -1.])
        rhs = derivative @ expected
        original_matrix, original_rhs = derivative.copy(), rhs.copy()
        result = _solve_periodic_beta_0(derivative, rhs)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(derivative @ result, rhs, atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(derivative, original_matrix)
        np.testing.assert_array_equal(rhs, original_rhs)

    def test_manufactured_periodic_solution(self):
        phi = 2*np.pi*np.arange(31)/31
        derivative = spectral_diff_matrix(len(phi))
        expected = np.sin(2*phi+.3) + .2*np.cos(3*phi)
        rhs = 2*np.cos(2*phi+.3) - .6*np.sin(3*phi)
        expected -= expected[0]
        result = _solve_periodic_beta_0(derivative, rhs)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(derivative @ result, rhs, atol=1e-12, rtol=1e-12)

    def test_pressure_configurations_without_vmec(self):
        # These configurations reach beta_0 during the VMEC integration test.
        # With constant B0, quadrature roundoff can make its analytic zero RHS
        # nonzero and expose the periodic derivative's constant nullspace.
        for name in ['r2 section 5.3', 'r2 section 5.5']:
            q = Qic.from_paper(name, nphi=151, order='r2')
            self.assertTrue(np.all(np.isfinite(q.beta_0)))
            np.testing.assert_allclose(q.beta_0, 0, atol=1e-12, rtol=0)


class ToroidalTensorTests(unittest.TestCase):
    def test_axisymmetric_grad_grad_B_components(self):
        # A tiny current regularizes the second-order solve; the analytic
        # vacuum toroidal-field limit is accurate to this test's tolerance.
        radius, field = 1.2, .9
        for sign_G in [-1, 1]:
            q = Qic(rc=[radius], zs=[0], nphi=15, B0=field,
                    etabar=.8, I2=1e-8, sG=sign_G, order='r2')
            q.calculate_grad_grad_B_tensor(two_ways=True)
            expected = np.zeros((q.nphi, 3, 3, 3))
            value = 2*sign_G*field/radius**2
            for index in [(2, 0, 0), (0, 2, 0), (0, 0, 2)]:
                expected[(slice(None),) + index] = value
            expected[:, 2, 2, 2] = -value
            np.testing.assert_allclose(q.grad_grad_B, expected, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(q.grad_grad_B, q.grad_grad_B_alt,
                                       atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
