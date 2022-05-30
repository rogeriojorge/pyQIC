#!/usr/bin/env python3

import cProfile
from qic import Qic

cProfile.run("""
for j in range(10):
    s = Qic(rc=[1, 0.045], zs=[0, 0.045], etabar=0.9, nphi=31, order="r2")
""", sort='tottime')
