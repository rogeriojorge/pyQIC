#!/usr/bin/env python3

import unittest
import numpy as np
from qic.qic import Qic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewtonTests(unittest.TestCase):

    def test_omnigenity(self):
        # Order r2
        nphi=351
        # rc      = [ 1.0,0.0,-0.2,0.0,0.0,0.0,0.0,0.0,0.0 ]
        # zs      = [ 0.0,0.0,0.3052828826418424,0.0,-0.0062592338816644544,0.0,-0.0013608816560281526,0.0,0.00034499224916859685 ]
        # B0_vals = [ 1.0,0.20356847942029987 ]
        # d_svals = [ 0.0,0.0,0.0,0.0 ]
        # k_second_order_SS   = 10.767959843661382
        # delta   = 1.756279248056422
        # nfp     = 1
        # B2s_svals = [ 0.0,-0.2890116275360059,0.42365891429851427,0.19804239404493473,-0.04647331768552959 ]
        # B2c_cvals = [ -0.07443435202732489,0.059675356537100585,-0.42166691507263726,0.10299024291977077,-0.2222054479974802 ]
        # p2      =  0.0
        # stel    =  Qic(rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, p2=p2, order='r2', k_second_order_SS=k_second_order_SS)
        rc      = [ 1.0,0.0,-0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
        zs      = [ 0.0,0.0,0.3515256195485459,0.0,-0.025760473197283567,0.0,0.0012843427458089259,0.0,-0.0010382838326711225,0.0,5.8831297609230655e-05,0.0,-2.5206215753153992e-05,0.0,1.0528739193321092e-05,0.0,1.6498256568592327e-06,0.0,7.231278664950175e-07,0.0,6.236575496049474e-08 ]
        B0_vals = [ 1.0,0.2898599196464543 ]
        k_second_order_SS   = 3.8393349190026678
        d_svals = [0.]
        delta   = 1.657971751685515
        nfp     = 1
        B2s_svals = [ 0.0,0.02516989814033407,0.03283462033677047,0.004837653606915959,0.03466436789051,0.012588451775448475,0.0034798016087720404,0.012368193922088775,0.03844118710392006,0.002403469150739275,0.01163670504914525 ]
        B2c_cvals = [ 0.0206121680245761,0.0602654567737221,0.06344485624821924,0.026187767224228442,0.05839086377958894,0.007915937165665798,0.036743548309213274,0.036842540401104525,0.0142687978041074,-0.013198287181027281,-0.0072079191108149875,0.0 ]
        B2s_cvals = [ -0.1318900156971933,0.5649273497420052,-0.022473047295254987,-0.08265049514305536,0.41342725761575694,-0.2931657967710277,0.31542024309141736,-0.21005748857764805,0.6787636103131851,-0.1240125056565712,0.1242848766908088,0.0 ]
        B2c_svals = [ 0.0,1.5753225284964127,-0.57516292896785,0.8948531647751348,-3.0875825530983967,-0.06329756906736621,-1.7890702692964318,1.039759175233141,-0.1690212827616995,-0.16762613422413772,0.015322897225364228,0.0,0.0 ]
        p2      =  0.0
        stel    =  Qic(rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, B2c_cvals=B2c_cvals, B2s_svals=B2s_svals, p2=p2, order='r3', k_second_order_SS=k_second_order_SS, B2s_cvals=B2s_cvals, B2c_svals=B2c_svals)
        import matplotlib.pyplot as plt
        # plt.plot(stel.X1c, label='X1c')
        # plt.plot(stel.X1s, label='X1s')
        # plt.plot(stel.Y1c, label='Y1c')
        # plt.plot(stel.Y1s, label='Y1s')
        # plt.plot(stel.sigma, label='sigma')
        # plt.plot(stel.d, label='d')
        # plt.plot(stel.d_bar, label='d_bar')
        # plt.plot(stel.alpha, label='alpha')
        # plt.plot(stel.X20, label='X20')
        # plt.plot(stel.X2c, label='X2c')
        # plt.plot(stel.X2s, label='X2s')
        # plt.plot(stel.Y20, label='Y20')
        # plt.plot(stel.Y2c, label='Y2c')
        # plt.plot(stel.Y2s, label='Y2s')
        # plt.plot(stel.B20, label='B20')
        # plt.plot(stel.B2c_array, label='B2c_array')
        # plt.plot(stel.B2s_array, label='B2s_array')
        plt.figure()
        plt.plot(stel.B20, label='B20')
        plt.plot(stel.B20[::-1], label='B20 reversed')
        plt.plot(stel.B20QI_deviation, label='B20 deviation')
        plt.legend()
        plt.figure()
        plt.plot(stel.B2cQI, label='B2cQI')
        plt.plot(stel.B2cQI[::-1], label='B2cQI reversed')
        plt.plot(stel.B2cQI_deviation, label='B2cQI deviation')
        plt.legend()
        plt.figure()
        plt.plot(stel.B2sQI, label='B2sQI')
        plt.plot(stel.B2sQI[::-1], label='B2sQI reversed')
        plt.plot(stel.B2sQI_deviation, label='B2sQI deviation')
        plt.legend()
        plt.show()
        # stel.plot()
        # stel.plot_boundary(r=0.05,fieldlines=True)
        
if __name__ == "__main__":
    unittest.main()
