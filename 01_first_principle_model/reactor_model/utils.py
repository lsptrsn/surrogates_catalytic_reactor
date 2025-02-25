#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:33:33 2023

@author: peterson
"""

###########################################################################
# Import packages
###########################################################################
from functools import partial
import jax
import jax.numpy as np

from scipy.optimize import fsolve
import thermodynamics as thermo
from parameter import data

jax.config.update("jax_enable_x64", True)
eps = np.finfo(float).eps

@jax.jit
def replace_out_of_bounds_values(array, threshold_value=eps, is_lower_bound=True):
    """
    Replace values in an array based on a threshold.

    Parameters:
    - array: np.ndarray, the input array to modify.
    - threshold_value: float, the value to replace out-of-bound elements with.
    - is_lower_bound: bool, True to replace values below the threshold,
                           False to replace values above the threshold.

    Returns:
    - np.ndarray with modified values.
    """
    updated_array = jax.lax.cond(
        is_lower_bound,
        # If True, replace values lower than threshold
        lambda arr: np.where(arr < threshold_value, threshold_value, arr),
        # If False, replace values higher than threshold
        lambda arr: np.where(arr > threshold_value, threshold_value, arr),
        operand=array
    )

    return updated_array

def format_time(seconds):
    """
    Convert seconds to a string format of hours, minutes, and seconds.

    Parameters:
    seconds (float): Time in seconds.

    Returns:
    str: Formatted time string in hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f"{hours}h {minutes}m {seconds}s"


def get_center_temperature(X_CO2, T, data):
    """Get temperature at the reactor center based on average temperature."""
    T_cool = data["T_cool"]
    v_gas_in = data["v_gas_in"]
    X_IV = X_CO2[0]
    T_IV = T[0]
    X_CO2 = X_CO2[1:]
    T = T[1:]

    ###########################################################################
    # INITIAL CONDITIONS
    ###########################################################################
    # pressure drop in Pa
    p_loss = 0
    p_R_loss = data["p_R"]-p_loss*data["zeta"]*data["L_R"]

    ###########################################################################
    # CONVERSIONS
    ###########################################################################
    # get part of fractions
    n_in_1 = data["n_in"][1]
    n_i = data["n_in"][:, None, None] + data["nue"][:, None, None] * X_CO2[None, :] * n_in_1
    n = np.sum(n_i, axis=0)
    x_i = n_i / n
    # M_gas
    # M_gas = np.sum(x_i.T*data["Molar_Mass"], axis=2).T
    M_gas = np.einsum('ijk,i->jk', x_i, data["Molar_Mass"])
    # Gas densitity by ideal gas law in kg/m^3 (validiert)
    density_fluid = (p_R_loss[:, None] * M_gas) / (data["R"] * T)
    # Mass flow (axial mass flow remains constant) - quasistationary
    v_gas = v_gas_in * data["density_fluid_in"] / density_fluid

    ###########################################################################
    # THERMODYNAMICS
    ###########################################################################
    expanded_X_prop = np.tile(data["X_CO2_prop"][:, np.newaxis],
                              (1, X_CO2.shape[1]))
    n_i_prop = data["n_in"][:, None, None] \
        + data["nue"][:, None, None] * expanded_X_prop[None, :] * n_in_1
    n_prop = np.sum(n_i_prop, axis=0)
    x_i_prop = n_i_prop / n_prop

    # Initialize empty lists to store the results
    U_list = []
    U_slash_list = []
    # List comprehension to compute U and U_slash
    for i in range(X_CO2.shape[1]):
        U_i, U_slash_i, _ = thermo.get_heat_transfer_coeff(
            T[:, i], x_i_prop[:, :, i], v_gas[:, i], density_fluid[:, i])
        U_list.append(U_i.flatten())
        U_slash_list.append(U_slash_i.flatten())
    # Convert lists to numpy arrays and concatenate along the appropriate axis
    U = np.stack(U_list, axis=1)
    U_slash = np.stack(U_slash_list, axis=1)

    temperature_difference = T - T_cool
    T_center = U/U_slash*(temperature_difference) + T_cool
    T_center = np.concatenate([T_IV[np.newaxis, :], T_center], axis=0)
    return T_center


def get_equilibrium_conversion(T, X_CO2, p):
    """
    Estimate equilibrium conversion.

    Parameters:
    T : array
        Temperature in K.
    X_CO2 : float
            Coversion of CO2.
    p : float
        Prssure in Pa.


    Returns:
    r : array
        Reaction rates.
    """
    n_CO2_0 = 1
    n_H2_0 = 4
    n_CH4_0 = 0
    n_H2O_0 = 0

    # Calculate updated molar amounts
    n_CO2 = n_CO2_0 + data["nue"][1]*X_CO2*n_CO2_0
    n_H2 = n_H2_0 + data["nue"][0]*X_CO2*n_CO2_0
    n_CH4 = n_CH4_0 + data["nue"][2]*X_CO2*n_CO2_0
    n_H2O = n_H2O_0 + data["nue"][3]*X_CO2*n_CO2_0
    n_ges = n_CO2 + n_H2 + n_CH4 + n_H2O

    # Calculate mole fractions
    x_CO2 = n_CO2/n_ges
    x_H2 = n_H2/n_ges
    x_CH4 = n_CH4/n_ges
    x_H2O = n_H2O/n_ges

    # Partial pressures
    p_H2 = x_H2*p
    p_CO2 = x_CO2*p
    p_CH4 = x_CH4*p
    p_H2O = x_H2O*p

    # Equilibrium constant
    Keq_LHS = 137 * T**(-3.998) * np.exp(158.7E3 / (data["R"] * T))
    Keq_LHS *= (1.01325 * 1e5)**-2  # Convert to Pa
    Keq_RHS = (p_CH4 * p_H2O**2) / (p_CO2 * p_H2**4)

    # Calculate the equilibrium conversion (in Pa)
    sol = Keq_RHS - Keq_LHS
    return sol


def get_equilibrium_curve(p):
    """
    Calculates the equilibrium conversion and temperature for a stoichiometric
    reaction mixture of methane synthesis.

    Args:
        p (float): Pressure in pascal.

    Returns:
        tuple: A tuple containing equilibrium temperature (Temp_GG) and
        equilibrium conversion (X_GG).
    """
    T = np.linspace(500, 1000, 500)

    X_CO2_save = []
    X_CO2 = 0.99  # Starting guess

    # Iterate over conversion
    for T_iter in T:
        def equation(X_CO2_i):
            return get_equilibrium_conversion(T_iter, X_CO2_i, p)
        # Use solution as initial value for the next temperature
        X_CO2 = fsolve(equation, X_CO2)[0]
        X_CO2_save.append(X_CO2)

    Temp_GG = np.array(T)
    X_GG = np.array(X_CO2_save)

    return Temp_GG, X_GG


def interpolate_equilibrium_curve(Temp_GG, X_GG, T_interp):
    """
    Get interpolated equilibrium conversion for any temperatures.
    """
    X_GG_interp = np.interp(T_interp, Temp_GG, X_GG)
    return X_GG_interp
