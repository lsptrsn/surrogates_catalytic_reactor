#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 08:42:30 2023

@author: peterson
"""
import jax.numpy as np
import jax

from parameter import data
eps = np.finfo(float).eps


@jax.jit
def reaction_rate_LHHW(T, x, p):
    """
    Estimate reaction rate according to Koschany et al.

    Parameters:
    T : array
        Temperature in K.
    x : array
        Mole fraction array.
    p : float
        Pressure in Pa.

    Returns:
    r : array
        Reaction rates.
    """
    # Kinetics data and constants
    T_ref = 555  # Reference Temperature in Kelvin

    # Other constants (detailed comments or explanations for each)
    eq_const_H2_0 = 0.00139140217047409  # Eq. Konstant Hydrogen in Pa^-0.5
    eq_const_OH_0 = 0.00158113883008419  # Eq. Constant Hydroxyl in Pa^-0.5
    eq_const_mix_0 = 0.00278280434094817  # Eq. Constant Mix in Pa^-0.5
    dH_H2 = -6.2E3  # Enthalpy Difference Hydrogen in J mol^-1
    dH_OH = 22.4E3  # Enthalpy Difference Hydroxyl in J mol^-1
    dH_mix = -10E3  # Enthalpy Difference Mix in J mol^-1

    temp_dependence = (1 / T_ref - 1 / T)

    # Arrhenius and van't Hoff dependencies
    scale_k0 = 6.86222794
    scale_energy_act = 1.22963487
    k0 = data["k0_LHHW"]*scale_k0
    energy_act = data["energy_act_LHHW"]*scale_energy_act
    k = k0 * np.exp(energy_act / data["R"] * temp_dependence)
    K_H2 = eq_const_H2_0 * np.exp(dH_H2 / data["R"] * temp_dependence)
    K_OH = eq_const_OH_0 * np.exp(dH_OH / data["R"] * temp_dependence)
    K_mix = eq_const_mix_0 * np.exp(dH_mix / data["R"] * temp_dependence)

    # Equilibrium constant
    Keq = 137 * T**(-3.998) * np.exp(158.7E3 / data["R"] / T)
    Keq *= (1.01325 * 1e5)**-2  # Convert to Pa

    # Partial pressures
    p_i = x * p
    p_H2, p_CO2, p_CH4, p_H2O = p_i[0:4]

    sqrt_H2 = (p_H2+eps)**0.5
    sqrt_CO2 = (p_CO2+eps)**0.5
    r_NUM = k * sqrt_H2 * sqrt_CO2 * (1 - (p_CH4 * p_H2O**2) / (p_CO2 * p_H2**4 * Keq + eps))
    r_DEN = (1 + K_OH * p_H2O / (sqrt_H2 + eps) + K_H2 * sqrt_H2 + K_mix * sqrt_CO2)
    r = r_NUM / (r_DEN**2)
    return r


@jax.jit
def reaction_rate_PL(T, x, p):
    """
    Estimate LHHW reaction rate according to Koschany et al.

    Parameters:
    T : array
        Temperature in K.
    x : array
        Mole fraction array.
    p : float
        Pressure in Pa.

    Returns:
    r : array
        Reaction rates.
    """
    # Kinetics data and constants
    T_ref = 555  # Reference Temperature in Kelvin

    temp_dependence = (1 / T_ref - 1 / T)

    # scaling factors
    scale_k0 = 1.34512735
    scale_energy_act = 2.86483796
    scale_n_H2 = 0.13088832
    scale_n_CO2 = 1.39879318
    # Arrhenius and van't Hoff dependencies
    k0 = data["k0_PL"]*scale_k0
    energy_act = data["energy_act_PL"]*scale_energy_act
    n_H2 = data["n_H2"]*scale_n_H2
    n_CO2 = data["n_CO2"]*scale_n_CO2
    k = k0 * np.exp(energy_act / data["R"] * temp_dependence)

    # Equilibrium constant
    Keq = 137 * T**(-3.998) * np.exp(158.7E3 / data["R"] / T)
    Keq *= (1.01325 * 1e5)**-2  # Convert to Pa

    # Partial pressures
    p_i = x * p
    p_H2, p_CO2, p_CH4, p_H2O = p_i[0:4]

    # Reaction equation
    r = k * p_H2**n_H2 * p_CO2**n_CO2 \
        * (1 - (p_CH4 * p_H2O**2) / (p_CO2 * p_H2**4 * Keq + eps))
    return r
