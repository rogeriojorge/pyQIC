"""
This module optimises parameters of a NAE
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
from scipy import optimize
from .util import mu0

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
        print(f"\N{GREEK CAPITAL LETTER DELTA}B2c = {stel.B2cQI_deviation_max:.4f},",
            # f"1/rc = {1/stel.r_singularity:.2f},",
            # f"1/L\N{GREEK CAPITAL LETTER DELTA}B = {np.max(stel.inv_L_grad_B):.2f},",
            f"Residual = {res:.4f}")
    res_history.append(res) # Attach residual value to history
    return res

def fun(stel, extras):
    # Second order QI quality residual for B2c (simple case as an example)
    res = stel.B2cQI_deviation_max
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
    
    # Returns the exit flag of the optimisation
    return opt.message
