#!/usr/bin/env python3

import logging
import os
import unittest
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt
from qic.qic import Qic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cases = ['r1 section 5.1', 'r1 section 5.2', 'r1 section 5.3',
         'r2 section 5.1', 'r2 section 5.2', 'r2 section 5.3', 'r2 section 5.4', 'r2 section 5.5']

# Calling plt.close() after each matplotlib figure is not essential,
# but it eliminates a matplotlib warning about opening too many
# figures at once, and it reduces memory usage.

class PlotTests(unittest.TestCase):

    def test_plot(self):
        for case in cases:
            s = Qic.from_paper(case, order='r1')
            s.plot(show=False)
            plt.close()
        
            s = Qic.from_paper(case, order='r2')
            s.plot(show=False)
            plt.close()
        
            s = Qic.from_paper(case, order='r3')
            s.plot(show=False)
            plt.close()
        
    def test_axis(self):
        """
        Test call to plot axis shape
        """
        stel = Qic.from_paper('r1 section 5.2')
        stel.plot_axis(frenet=False, show=False)
        plt.close()

        stel = Qic.from_paper(4, order='r2')
        stel.plot_axis(frenet=False, show=False)
        plt.close()

        stel = Qic.from_paper(4, order='r3')
        stel.plot_axis(frenet=False, show=False)
        plt.close()
                
    def test_other_plots(self):
        """
        Call other plotting functions to check that they work with a first
        order case and a second order one
        """
        stel = Qic.from_paper("r1 section 5.1")
        # stel.plot(fieldlines=True)
        stel.plot_boundary(show=False)
        stel.B_fieldline(show=False)
        plt.close()
        stel.B_contour(show=False)
        plt.close()
        
        stel = Qic.from_paper(4, order='r2')
        # stel.plot(fieldlines=True)
        stel.plot_boundary(show=False)
        stel.B_fieldline(show=False)
        plt.close()
        stel.B_contour(show=False)
        plt.close()
        
        stel = Qic.from_paper(4, order='r3')
        # stel.plot(fieldlines=True)
        stel.plot_boundary(show=False)
        stel.B_fieldline(show=False)
        plt.close()
        stel.B_contour(show=False)
        plt.close()

if __name__ == "__main__":
    unittest.main()
