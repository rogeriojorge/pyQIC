#!/usr/bin/env python3
# Example of the optimisation of d_bar for second order QI
import os
import shutil
from qic import Qic
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import optimize, interpolate
this_path = Path(__file__).parent.resolve()

# Print the resulting stellarator solution
def print_results(stel):
    out_txt  = f'from qic import Qic\n'
    out_txt += f'def optimized_configuration_nfp{stel.nfp}(nphi=131,order = "r2"):\n'
    out_txt += f'    rc      = [{",".join([str(elem) for elem in stel.rc])}]\n'
    out_txt += f'    zs      = [{",".join([str(elem) for elem in stel.zs])}]\n'
    out_txt += f'    B0_vals = [{",".join([str(elem) for elem in stel.B0_vals])}]\n'
    out_txt += f'    omn_method = "{stel.omn_method}"\n'
    out_txt += f'    k_buffer = {stel.k_buffer}\n'
    out_txt += f'    p_buffer = {stel.p_buffer}\n'
    out_txt += f'    d_over_curvature_cvals = [{",".join([str(elem) for elem in stel.d_over_curvature_cvals])}]\n'
    out_txt += f'    delta   = {stel.delta}\n'
    out_txt += f'    d_svals = [{",".join([str(elem) for elem in stel.d_svals])}]\n'
    out_txt += f'    nfp     = {stel.nfp}\n'
    out_txt += f'    iota    = {stel.iota}\n'
    if not stel.order=='r1':
        out_txt += f'    X2s_svals = [{",".join([str(elem) for elem in stel.B2s_svals])}]\n'
        out_txt += f'    X2c_cvals = [{",".join([str(elem) for elem in stel.B2c_cvals])}]\n'
        out_txt += f'    X2s_cvals = [{",".join([str(elem) for elem in stel.B2s_cvals])}]\n'
        out_txt += f'    X2c_svals = [{",".join([str(elem) for elem in stel.B2c_svals])}]\n'
        out_txt += f'    p2      = {stel.p2}\n'
        if not stel.p2==0:
            out_txt += f'    # DMerc mean  = {np.mean(stel.DMerc_times_r2)}\n'
            out_txt += f'    # DWell mean  = {np.mean(stel.DWell_times_r2)}\n'
            out_txt += f'    # DGeod mean  = {np.mean(stel.DGeod_times_r2)}\n'
        out_txt += f'    # B20QI_deviation_max = {stel.B20QI_deviation_max}\n'
        out_txt += f'    # B2cQI_deviation_max = {stel.B2cQI_deviation_max}\n'
        out_txt += f'    # B2sQI_deviation_max = {stel.B2sQI_deviation_max}\n'
        out_txt += f'    # Max |X20| = {max(abs(stel.X20))}\n'
        out_txt += f'    # Max |Y20| = {max(abs(stel.Y20))}\n'
        if stel.order == 'r3':
            out_txt += f'    # Max |X3c1| = {max(abs(stel.X3c1))}\n'
        out_txt += f'    # gradgradB inverse length: {stel.grad_grad_B_inverse_scale_length}\n'
        out_txt += f'    # d2_volume_d_psi2 = {stel.d2_volume_d_psi2}\n'
    out_txt += f'    # max curvature_d(0) = {stel.d_curvature_d_varphi_at_0}\n'
    out_txt += f'    # max d_d(0) = {stel.d_d_d_varphi_at_0}\n'
    out_txt += f'    # max gradB inverse length: {np.max(stel.inv_L_grad_B)}\n'
    out_txt += f'    # Max elongation = {stel.max_elongation}\n'
    out_txt += f'    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)'
    with open(os.path.join(this_path,f'optimized_configuration_nfp{stel.nfp}.py'), 'w') as f:
        f.write(out_txt)
    print(out_txt)
    return out_txt

# Define a function that in addition to the condition of second order QI, it also requires some elongation constraint
def geo_condition_and_elongation(stel, order):
    res = 0
    if np.max(stel.elongation)>8.0:
        res += np.max(stel.elongation)
    elif np.min(stel.elongation)<1/8.0:
        res += 1/np.min(stel.elongation)
    return stel.min_geo_qi_consistency(order = order) + res

# Some arbitrary initial case with an appropriate axis (with order 1 curvature)
# nphi = 201
# order = 'r1'
# nfp=1
# rc      = [ 1.0,0.0,-0.3,0.0,0.01,0.0,0.001 ]
# zs      = [ 0.0,0.0,-0.2,0.0,0.01,0.0,0.001 ]
# B0_vals = [ 1.0,0.16 ]
# omn_method ='non-zone-fourier'
# k_buffer = 1
# p_buffer = 2
# delta = 0.1
# d_over_curvature_cvals = [0.5,0.01,0.01,0.01,0.01,0.01,0.01] # Initial case, it sets the length of the search
# stel = Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, 
#     nphi=nphi, omn=True, order=order, d_over_curvature_cvals=d_over_curvature_cvals)
from optimized_configuration_nfp1 import optimized_configuration_nfp1
stel = optimized_configuration_nfp1()

# Optimise for d_bar without worrying about d_bar crossing the zero and elongation becoming too large
# res = stel.construct_qi_r2(verbose=1, show=False)
# stel.plot()

# Optimise for d_bar, but taking into account elongation
# stel = Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, 
#     nphi=nphi, omn=True, order=order, d_over_curvature_cvals=d_over_curvature_cvals)
params = [  'zs(2)','B0(1)','B2cs(1)','B2sc(0)',
            'zs(4)','rc(2)','B2cs(2)','B2sc(1)',
            'zs(6)','rc(4)','B2cs(3)','B2sc(2)',
            'B2cs(4)','B2sc(3)','B2cs(5)','B2sc(4)',
            'd_over_curvaturec(0)','d_over_curvaturec(1)','d_over_curvaturec(2)',
            'd_over_curvaturec(3)','d_over_curvaturec(4)','d_over_curvaturec(5)']
        #'B2cs(1)','B2sc(0)','B2cs(2)','B2sc(1)','B2cs(3)','B2sc(2)','B2cs(4)','B2sc(3)',
# Just to show that it can be done externally telling what parameters to use
maxiter = 300
maxfev = maxiter
res = stel.construct_qi_r2(verbose=1, method="Nelder-Mead", params=params,fun_opt = geo_condition_and_elongation, maxiter=maxiter, maxfev=maxfev, show=False)
res = stel.construct_qi_r2(verbose=1, method="BFGS", params=params,fun_opt = geo_condition_and_elongation, maxiter=maxiter, maxfev=maxfev, show=False) # Refine the solution
# stel.plot()
print_results(stel)