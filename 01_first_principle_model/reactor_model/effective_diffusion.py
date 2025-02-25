#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:13:19 2023

@author: peterson
"""
import jax
import jax.numpy as np

from parameter import data
import thermodynamics as thermo
from utils import replace_out_of_bounds_values

eps = np.finfo(float).eps


@jax.jit
def effective_diffusion(T, p, x_i, v_gas, r_int):
    # Calculate bulk concentrations
    c_ges_bulk = p / (data["R"] * T)
    c_bulk = c_ges_bulk * x_i

    # Calculate effective diffusion coefficients for CO2
    D_Knudsen_helper = np.sqrt(8 / np.pi * data["R"] * T / data["Molar_Mass"][1])
    D_Knudsen_pore = data["d_pore_core"] / 3 * D_Knudsen_helper
    D_eff_core = data["epsilon_core"] / data["tau_core"] * D_Knudsen_pore
    D_Knudsen_shell = (data["d_pore_shell"]) / 3  * D_Knudsen_helper
    D_eff_shell = data["epsilon_shell"] / data["tau_shell"] * D_Knudsen_shell

    # Calculate transfer coefficients for CO2
    ke_CO2, _ = transfer_coefficients(T, p, c_bulk, v_gas)

    # Calculate equilibrium constant and partial pressures
    Keq = 137 * T**(-3.998) * np.exp(158700 / (data["R"] * T))
    Keq = Keq * (1.01325 * 1e5)**-2  # Convert to Pa
    p_i = x_i * p
    c_CO2_GG = p_i[3]**2 * p_i[2] / (p_i[0]**4 * Keq + eps) / (data["R"] * T)

    # Calculate additional parameters
    n = 2
    Psi = 1 / data["r_core"] - 1 / data["r_shell"]
    # thiele modulus
    term = r_int / (D_eff_core * (c_bulk[1] - c_CO2_GG)+eps)
    term = replace_out_of_bounds_values(term, eps)
    Phi = data["r_core"] / (n + 1) * np.sqrt(term)
    delta = (data["r_shell"] - data["r_core"]) / data["r_core"]
    Bi_ext = (ke_CO2 * data["r_core"]) / (D_eff_core * (n + 1))
    Bi_int_inv = (D_eff_core / D_eff_shell) * (data["r_core"]**(n - 1) * Psi * (n + 1))

    # Calculate efficiency factor
    eta_inv = (1 + delta)**3 * (Phi / (np.tanh(Phi)+eps) + Phi**2 / ((1 + delta)**n * Bi_ext) + Phi**2 * Bi_int_inv)
    eta = (eta_inv+eps)**-1
    return eta


@jax.jit
def transfer_coefficients(T_surf, p_surf, c_surf, v_gas):
    c_surf_sum = np.sum(c_surf, axis=0)
    x_surf = c_surf / c_surf_sum
    rho_surf = np.sum(c_surf * data["Molar_Mass"][:, np.newaxis], axis=0)

    # Calculate D_bin, viscosity, cp, and heat_conductivity
    D_bin = diffusion_coefficients_fuller(T_surf, p_surf, x_surf)
    viscosity_fluid = thermo.get_viscosity_fluid(T_surf, x_surf)
    cp_fluid = thermo.get_cp_fluid(T_surf, x_surf)
    heat_conductivity_fluid = thermo.get_lambda_fluid(T_surf, x_surf)

    Re = rho_surf * v_gas * (2 * data["r_shell"]) / viscosity_fluid
    Pr = cp_fluid * viscosity_fluid / heat_conductivity_fluid
    Nu = 2 + 1.1 * np.power(Pr, 1/3) * np.power(Re, 0.6)
    h = Nu * heat_conductivity_fluid / (2 * data["r_shell"])

    Sc = viscosity_fluid / (rho_surf * D_bin + eps)
    Sh = 2 + 1.1 * np.power(Sc, 1/3) * np.power(Re, 0.6)
    ke = Sh * D_bin / (2 * data["r_shell"])
    ke_CO2 = ke[1]
    return ke_CO2, h


@jax.jit
def diffusion_coefficients_fuller(T, p, x):
    num_species = len(data["species"])
    # num_species = 5
    Molar_Mass = data["Molar_Mass"] * 10 ** 3
    diffusion_volume = data["diffusion_volume"]

    t_ = 0.00143 * np.power(T[np.newaxis, np.newaxis, :], 1.75)
    p_ = p[np.newaxis, np.newaxis, :] / (10 ** 5) * (2 ** 0.5)
    m_ = np.power(1 / Molar_Mass[:, np.newaxis, np.newaxis] +
                  1 / Molar_Mass[np.newaxis, :, np.newaxis], 0.5)
    d_ = np.power(np.power(diffusion_volume[:, np.newaxis, np.newaxis], (1/3)) +
                  np.power(diffusion_volume[np.newaxis, :, np.newaxis], (1/3)), 2)

    D_BIN = t_ * m_ / (p_ * d_) * 10 ** -4

    N = np.arange(num_species)
    DENOM = np.zeros((num_species, len(T)))

    # TODO: vectorise this loop utilising masking and p.r.n. numpy.ix_
    for k in range(num_species):
        N = np.arange(num_species)
        N = np.delete(N, k)
        DENOM = DENOM.at[k, :].set(np.sum(x[N, :] / D_BIN[k, N, :], axis=0))

    D_eff= (1 - x) / (DENOM + eps)
    return D_eff
