"""
This module optimises parameters of a NAE
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
from scipy import optimize
from .util import mu0
from .fourier_interpolation import fourier_interpolation

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def opt_fun_stel(x_iter, stel, x_param_label, fun_opt, info = {'Nfeval':0}, res_history = [],verbose =0, extras =[], x_orig = [], thresh = 1.5):
    """
    Function that is passed to the optimiser which sets some properties of Qic object and
    evaluates a function handle that returns a residual.

    x_iter = arguments of the optimisation.
    stel = Qic object to evaluate things on
    x_param_label = name of the parameters of the Qic object that the x_iter values refer to
        (if x_orig is not empty, then x_iter correspond to scale factors).
    fun_opt = function handle of the form fun_opt(stel, extra) supplied.
    info = inside information from the optimisation loop (defaulted to the number of
         evaluations).
    res_history = external object to which the value of the residual is attached
    verbose = if set to 1, it prints the details of the evaluation
    extras = optional object to pass on to fun_opt
    x_orig = if given, then x_iter are not the arguments of the parameters denoted by 
        x_param_label, but rather x_iter*x_orig is
    thresh = limit on the value of x_iter for x_orig!=0. x_iter must satisfy 1/thresh<x_iter<thresh,
        so must be thresh > 1.
    """
    info['Nfeval'] += 1
    # Set degrees of freedom to input values 
    x_iter_all = stel.get_dofs()
    if len(x_orig): # If x_orig given, interpret x_iter as scales 
        for ind, label in enumerate(x_param_label):
            # Put a threshold to the allowable scale x_iter
            if np.any(x_iter[ind]>thresh) or np.any(x_iter[ind]<1/thresh):
                res = 1000 # Chosen a value arbitrarily
                res_history.append(res)
                return res
            # Construct new string of parameter values
            x_iter_all[stel.names.index(label)] = x_iter[ind]*x_orig[stel.names.index(label)]
    else:
        for ind, label in enumerate(x_param_label):
            x_iter_all[stel.names.index(label)] = x_iter[ind]
    # Construct the new nae solution
    stel.set_dofs(x_iter_all)
    # Evaluate the residual
    res = fun_opt(stel, extras)
    if verbose:
        print(f"{info['Nfeval']} -", x_iter)
        if stel.order == 'r2':
            print(f"\N{GREEK CAPITAL LETTER DELTA}B2c = {stel.B2cQI_deviation_max:.4f},",
                # f"1/rc = {1/stel.r_singularity:.2f},",
                # f"1/L\N{GREEK CAPITAL LETTER DELTA}B = {np.max(stel.inv_L_grad_B):.2f},",
                f"Residual = {res:.4f}")
        else: 
            print(f"Residual = {res:.4f}")
    res_history.append(res) # Attach residual value to history
    return res

def fun(stel, extras):
    # Second order QI quality residual for B2c (simple case as an example)
    res = stel.B2cQI_deviation_max
    return res

def min_geo_qi_consistency(stel, order = 1):
    """
    Function that computes the consistency conditions of a first order construction with 
    second order QI at the points phi=0,pi/nfp, and returns the mismatch.

    stel = the stellarator construction, Qic.
    order = order of the zeroes of curvature at the points where the consistency conditions
        are to be evaluated. This determines the number of constraints.
    
    """
    # Define various quantities obtained from Qic
    d_d_varphi = stel.d_d_varphi
    B0 = stel.B0
    dB0 = np.matmul(d_d_varphi, B0)
    B0_over_abs_G0 = stel.B0 / np.abs(stel.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    dldphi = abs_G0_over_B0
    X1c = stel.X1c
    X1s = stel.X1s
    Y1s = stel.Y1s
    Y1c = stel.Y1c
    iota_N = stel.iotaN
    torsion = stel.torsion

    # Construct second order quantities (like in calculate_r2), but only needing
    # order = 'r1' quantities
    V2 = 2 * (Y1s * Y1c + X1s * X1c)
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s - X1s * X1s

    factor = - B0_over_abs_G0 / 8
    Z2s = factor*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    dZ2s = np.matmul(d_d_varphi,Z2s)
    Z2c = factor*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)
    dZ2c = np.matmul(d_d_varphi,Z2c)

    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * torsion * abs_G0_over_B0
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * torsion * abs_G0_over_B0
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    # Construct the expressions necessary for the conditions
    Tc = B0/dldphi*(dZ2c + 2*iota_N*Z2s + (qc*qc-qs*qs+rc*rc-rs*rs)/4/dldphi)
    Ts = B0/dldphi*(dZ2s - 2*iota_N*Z2c + (qc*qs+rc*rs)/2/dldphi)

    angle = stel.alpha - (-stel.helicity * stel.nfp * stel.varphi)
    c2a1 = np.cos(2*angle)
    s2a1 = np.sin(2*angle)

    # It is unnecessary to compute this everywhere, but the way implemented this is the 
    # easy way
    cond_1st = (Tc*c2a1 + Ts*s2a1 + B0*B0*np.matmul(d_d_varphi,stel.d*stel.d/dB0)/4)
    pos = [0, np.pi] # Not necessary to impose it at 0, but it makes it easier to construct X2s and X2c later
    # Interpolate to the points required (especially because the original grid is shifted)
    cond_1st_eval= fourier_interpolation(cond_1st, pos-stel.phi_shift*stel.d_phi*stel.nfp)
    res = np.sum(cond_1st_eval*cond_1st_eval)
    der_cond = cond_1st
    # For higher order zeroes, we have additional conditions
    if order and order>1:
        for i in range(order-1):
            der_cond = np.matmul(d_d_varphi,der_cond)
            der_cond_eval = fourier_interpolation(der_cond, pos-stel.phi_shift*stel.d_phi*stel.nfp)
            res_add = np.sum(der_cond_eval*der_cond_eval)
            res += res_add

    # Return the mismatch
    return res

def optimise_params(stel, x_param_label, fun_opt = fun, verbose = 0, maxiter = 200, maxfev  = 200, method = 'Nelder-Mead', scale = 0, extras = [], thresh = 1.5):
    """
    Method that optimises stel wrt x_param_label for minimising the function fun_opt.

    stel = the nae Qic object to optimise
    x_param_label = name of the parameters of the Qic object to optimise. Ex.: (['etabar', 'rc(1)'])
    fun_opt = function handle of the form fun_opt(stel, extra) supplied as the measure of
        the residual.
    verbose = if set to anything different from 0, it prints the details of the optimisation
    maxiter = maximum number of iterations in the optimisation
    maxfev = maximum number of function evaluations in the optimisation
    method = method of the optimisation (possibilities are the methods that can be passed to
        scipy.optimize.minimize)
    scale = if it is set to anything other than 0, then it performs optimisation by considering 
        variations of the initial parameters by no more than 100*(thresh-1 or 1-1/thresh)%
    extras = optional object to pass on to fun_opt
    thresh = limit the value by which the optimised parameters may deviate from their initial 
        value (only works if scale!=0): 100*(thresh-1 or 1-1/thresh)%. It must be thresh>1.
    """
    # Extract initial values for the parameters to optimise
    x0_all = stel.get_dofs()
    x_parameter_label_checked = [] # List of available parameters to change
    x0 = []
    if scale:
        for ind, label in enumerate(x_param_label):
            # Check if parameter exists
            if label in stel.names:
                x0.append(1.0)
                x_parameter_label_checked.append(label)
            else:
                print('The label ',label, 'does not exist, and will be ignored.')
    else:
        for ind, label in enumerate(x_param_label):
            if label in stel.names:
                x0.append(x0_all[stel.names.index(label)])
                x_parameter_label_checked.append(label)
            else:
                print('The label ',label, 'does not exist, and will be ignored.')

    # Initialise the residual history 
    res_history = []
    res_history.append(fun_opt(stel, extras))
    
    # Optimisation (using scipy.optimize.minimize)
    if scale:
        # If scale, then limited optimisation relative to initial values
        opt = optimize.minimize(opt_fun_stel, x0, args=(stel, x_parameter_label_checked, fun_opt, {'Nfeval':0}, res_history, verbose, extras, x0_all, thresh), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev})
    else:
        opt = optimize.minimize(opt_fun_stel, x0, args=(stel, x_parameter_label_checked, fun_opt, {'Nfeval':0}, res_history, verbose, extras), method=method, tol=1e-3, options={'maxiter': maxiter, 'maxfev': maxfev})

    # Make sure that the final form of stel is with the optimised paramters
    x = stel.get_dofs()
    if scale:
        for ind, label in enumerate(x_parameter_label_checked):
            x[stel.names.index(label)] = opt.x[ind]*x0_all[stel.names.index(label)]
    else:
        for ind, label in enumerate(x_parameter_label_checked):
            x[stel.names.index(label)] = opt.x[ind]
    stel.set_dofs(x)
    
    # Plot the residual history if verbose
    if verbose:
        print(opt.x)
        plt.plot(res_history)
        plt.xlabel('Nb. Iterations')
        plt.ylabel('Objective function')
        plt.show()
    
    # Returns the residual in the optimisation
    return opt.fun
