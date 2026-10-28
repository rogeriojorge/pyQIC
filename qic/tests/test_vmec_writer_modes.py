"""Writer-only regressions; no VMEC, MPI, or equilibrium calculation needed."""
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from qic.to_vmec import to_vmec


def synthetic_configuration(nphi=17):
    q = SimpleNamespace(nfp=1, nphi=nphi, spsi=1, Bbar=1., p2=0.,
                        I2=0., etabar=1., sigma0=0., order='r1', lasym=False,
                        rc=np.array([1.]), zs=np.array([0.]))
    def boundary(r, ntheta):
        theta = np.arange(ntheta)[:, None] * 2*np.pi/ntheta
        phi = np.arange(nphi)[None, :] * 2*np.pi/nphi
        # Both retained modes and deliberately out-of-range modes are present.
        R = 1 + .1*np.cos(theta) + .02*np.cos(4*theta) + .03*np.cos(6*phi)
        Z = np.broadcast_to(.1*np.sin(theta), R.shape)
        return R, Z, np.zeros_like(R)
    q.Frenet_to_cylindrical = boundary
    return q


class WriterModeTests(unittest.TestCase):
    def write(self, params, ntor_max=14):
        q = synthetic_configuration()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'input.test'
            to_vmec(q, str(path), params=params, ntheta=24, ntorMax=ntor_max)
            return path.read_text(), q

    def test_mpol_is_number_of_modes(self):
        text, q = self.write(dict(mpol=4, ntor=2))
        modes = re.findall(r'RBC\(\s*(-?\d+),\s*(\d+)\)', text)
        self.assertTrue(modes)
        self.assertTrue(all(int(m) < 4 for _, m in modes))
        self.assertEqual(q.RBC.shape[0], 4)
        # Preserve a physical retained coefficient, not just the array shape.
        self.assertAlmostEqual(q.RBC[1, 2], .1)

    def test_ntor_cap_applies_to_boundary_coefficients(self):
        text, q = self.write(dict(mpol=4, ntor=6), ntor_max=2)
        self.assertIn('  NTOR = 2\n', text)
        modes = re.findall(r'RBC\(\s*(-?\d+),\s*(\d+)\)', text)
        self.assertTrue(all(abs(int(n)) <= 2 for n, _ in modes))
        self.assertEqual(q.RBC.shape[1], 5)


if __name__ == '__main__':
    unittest.main()
