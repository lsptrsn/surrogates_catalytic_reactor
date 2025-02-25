#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:28:13 2024

@author: peterson
"""
import scipy
from scipy.interpolate import interp1d
import numpy as np

import opinf.training


def integrate(t_span, y0, t, entries, model):
    """
    Integrate the Reduced Order Model (ROM) using scipy's solve_ivp.

    Parameters:
    - t_span (tuple): Time span for integration.
    - y0 (np.ndarray): Initial values.
    - t (np.ndarray): Time points at which to evaluate the solution.
    - entries (np.ndarray): Vector of control values
    - model (torch.nn.Module): The trained ROM model.

    Returns:
    - sol_ROM (scipy.integrate.OdeResult): Solution of the ROM integration.
    """

    def model_quad_OpInf(t, x):
        """
        Function defining the ROM model for scipy's solve_ivp.

        Parameters:
        - t (float): Time.
        - x (np.ndarray): State vector.
        - u (np.ndarray): Control input.
        - *args: Additional arguments (not used).

        Returns:
        - np.ndarray: Derivative of the state vector.
        """
        u = np.atleast_1d(u_func(t))
        return A_OpInf @ x + H_OpInf @ np.kron(x, x) + C_OpInf + B_OpInf @ u


    A_OpInf, B_OpInf, C_OpInf, H_OpInf = opinf.training.learned_model(model)
    u_func = interp1d(t, entries.flatten(), bounds_error=False, fill_value="extrapolate")
    dt = t[1] - t[0]
    sol_ROM = scipy.integrate.solve_ivp(model_quad_OpInf, t_span, y0,
                                        method='RK45', t_eval=t,  rtol=1e-7,
                                        atol=1e-9, max_step=dt)
    return sol_ROM
