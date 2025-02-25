"""
Solving Partial-Differential Equation 1D-PFTR for Methanantion
Author: Luisa Peterson
"""

###########################################################################
# Import packages
###########################################################################
import jax
import jax.numpy as np
from jax import lax

import thermodynamics as thermo
import pressure_drop as pd
from utils import replace_out_of_bounds_values
from reaction_rate import reaction_rate_PL
from effective_diffusion import effective_diffusion

jax.config.update("jax_enable_x64", True)
eps = np.finfo(float).eps

###########################################################################
# location-discretized partial differential equations
###########################################################################
# @jax.jit
def PDE_sys(t, x, data):
    """Build up mass and energy balances for the methanation reaction."""
    slice_size = data["Nz"]
    X_CO2 = lax.dynamic_slice(x, (0,), (slice_size,))
    X_CO2 = replace_out_of_bounds_values(X_CO2, eps)
    X_CO2 = replace_out_of_bounds_values(X_CO2,
                                          threshold_value=1,
                                          is_lower_bound=False)
    tau = lax.dynamic_slice(x, (slice_size,), (slice_size,))
    T = tau*data["T_scale"]+data["T_scale_add"]
    # T = replace_out_of_bounds_values(T, data["T_min"])

    ###########################################################################
    # TIME-CHANGING PARAMATERS
    ###########################################################################
    T_cool = np.interp(t, data["t_points"], data["T_cool"],
                    left='extrapolate', right='extrapolate')
    v_gas_in = np.interp(t, data["t_points"], data["v_gas_in"],
                          left='extrapolate', right='extrapolate')

    ###########################################################################
    # INITIAL CONDITIONS
    ###########################################################################
    # pressure drop in Pa
    # p_loss = pd.pressure_drop(data["T_gas_in"], data["x_in"],
    #                           data["density_fluid_in"], v_gas_in)
    # we don't need a pressure drop to describe our reactor -> set to 0
    p_loss = 0
    p_R_loss = data["p_R"]-p_loss*data["zeta"]*data["L_R"]

    ###########################################################################
    # CONVERSIONS
    ###########################################################################
    # get part of fractions
    n_i = data["n_in"][:, None] \
        + data["nue"][:, None]*X_CO2[None, :]*data["n_in"][1]
    n_i = replace_out_of_bounds_values(n_i, eps)
    # mole fraction
    n = np.sum(n_i, axis=0)
    x_i = (n_i/n)
    # x_i = np.where(x_i < 0, 1e-10, x_i)
    # M_gas
    M_gas = np.sum(x_i.T*data["Molar_Mass"], axis=1)
    # Gas densitity by ideal gas law in kg/m^3 (validiert)
    density_fluid = (p_R_loss*M_gas/data["R"]/T).T
    # Mass flow (axial mass flow remains constant) - quasistationary
    v_gas = v_gas_in*data["density_fluid_in"]/(density_fluid)

    ###########################################################################
    # THERMODYNAMICS
    ###########################################################################
    n_i_prop = data["n_in"][:, None] \
        + data["nue"][:, None]*data["X_CO2_prop"][None, :]*data["n_in"][1]
    n_prop = np.sum(n_i_prop, axis=0)
    # mole fraction (validiert)
    # TODO nur einmal berechnen
    x_i_prop = (n_i_prop/n_prop)
    # get thermodynamics
    cp_fluid = thermo.get_cp_fluid(data["T_prop"], x_i_prop)
    H_r = thermo.get_reaction_enthalpy(T)
    U, _, lambda_ax = thermo.get_heat_transfer_coeff(
        T, x_i_prop, v_gas, density_fluid)

    ###########################################################################
    # REACTION RATE
    ###########################################################################
    # mass based reactino rate in  mol/(s*kg_cat)
    def no_reaction(_):
        # molar Reaction Rate in mol/(s*mcat^3)
        return np.zeros(data["Nz"])

    def reaction(_):
        # molar Reaction Rate in mol/(s*mcat^3)
        r = reaction_rate_PL(T, x_i, p_R_loss)
        r_int = (1-data["epsilon_core"])*data["density_cat_core"]*r
        eta = effective_diffusion(T, p_R_loss, x_i, v_gas, r_int)
        return data["cat_frac"]*(1-data["epsilon"])*eta*r_int

    all_inputs_available = np.min(x_i[0, :])*np.min(x_i[1, :])
    r_eff = lax.cond(all_inputs_available < 1e-5, no_reaction, reaction, operand=None)
    ###########################################################################
    # BALANCES
    ###########################################################################
    # Initialize composition matrix
    dxdt = np.zeros((2, data["Nz"]))
    dz = data["D_zeta"] * data["L_R"]

    # Mass Balance
    dxdt_mass_0 = -v_gas[0] * (X_CO2[0]-data["X_in"])/data["epsilon"]/dz[0] \
        + data["Molar_Mass"][1]*r_eff[0]/(data["w_in"][1]+eps) / \
        density_fluid[0]/data["epsilon"]
    dxdt = dxdt.at[0, 0].set(dxdt_mass_0)
    # inner control volumes
    dxdt_mass = -v_gas[1:data["Nz"]]*(X_CO2[1:data["Nz"]]-X_CO2[0:data["Nz"]-1]) \
        / data["epsilon"] / dz[1:data["Nz"]] \
        + data["Molar_Mass"][1]*r_eff[1:data["Nz"]] \
        / ((data["w_in"][1]+eps)*density_fluid[1:data["Nz"]]*data["epsilon"]+eps)
    dxdt = dxdt.at[0, 1:data["Nz"]].set(dxdt_mass)

    # Pre-compute constants and repeated expressions
    lambda_ax_mean = 0.5 * (lambda_ax[:-1] + lambda_ax[1:])

    # expressions for Axial Dispersion
    Ax_Disp = np.zeros(data["Nz"])
    # Vectorized computation for the inner elements
    i = np.arange(1, data["Nz"]-1)
    Ax_Disp_at_inner = lambda_ax_mean[i]*(T[i+1]-T[i])/((data["L_R"]*data["d_zeta"][i])*dz[i]) - \
        lambda_ax_mean[i-1]*(T[i]-T[i-1]) / \
        ((data["L_R"]*data["d_zeta"][i-1])*dz[i])
    Ax_Disp = Ax_Disp.at[i].set(Ax_Disp_at_inner)
    # left bc
    Ax_Disp_at_0 = lambda_ax_mean[0] * \
        (T[1] - T[0]) / ((data["L_R"]*data["d_zeta"][0])*dz[0])
    Ax_Disp = Ax_Disp.at[0].set(Ax_Disp_at_0)
    # richt bc
    Ax_Disp_at_end = -lambda_ax_mean[data["Nz"]-2]*(T[-1]-T[-2])/(
        (data["L_R"]*data["d_zeta"][data["Nz"]-2])*(dz[-1]))
    Ax_Disp = Ax_Disp.at[-1].set(Ax_Disp_at_end)

    # Kinetics for heat transfer (heat coefficient gas-wall)
    g_qw = 4 * U * (T - T_cool) / data["D_R"]

    # Calculating effective heat capacity
    rhocp_cat = ((data["r_core"]**3 / data["r_shell"]**3) * (1 - data["epsilon_core"]) * data["density_cat_core"] * data["cp_core"]
                 + (data["r_shell"]**3 - data["r_core"]**3) / data["r_shell"]**3 * (1 - data["epsilon_shell"]) * data["density_cat_shell"] * data["cp_shell"])
    rhocp_g = density_fluid*cp_fluid*data["epsilon"] \
        + rhocp_cat*(1-data["epsilon"]) + eps
    dxdt_temp_0 = (-v_gas_in*data["density_fluid_in"]*cp_fluid[0]
                   * ((T[0]-data["T_gas_in"])/dz[0]))/rhocp_g[0] \
        - r_eff[0].T*H_r[0]/rhocp_g[0] \
        - g_qw[0]/rhocp_g[0] \
        + Ax_Disp[0]/rhocp_g[0]
    dxdt_tau_0 = dxdt_temp_0/data["T_scale"]
    dxdt = dxdt.at[1, 0].set(dxdt_tau_0)
    # ODEs inner control volumes
    dxdt_temp = (-v_gas_in*data["density_fluid_in"]*cp_fluid[1:data["Nz"]]
                 * ((T[1:data["Nz"]]-T[0:data["Nz"]-1])/dz[1:data["Nz"]])) \
        / rhocp_g[1:data["Nz"]] \
        - r_eff[1:data["Nz"]]*H_r[1:data["Nz"]]/rhocp_g[1:data["Nz"]] \
        - g_qw[1:data["Nz"]]/rhocp_g[1:data["Nz"]] \
        + Ax_Disp[1:data["Nz"]]/rhocp_g[1:data["Nz"]]
    dxdt_tau = dxdt_temp/data["T_scale"]
    dxdt = dxdt.at[1, 1:data["Nz"]].set(dxdt_tau)

    # Flatten to create a long row vector with entries for each component in each control volume
    dxdt = dxdt.flatten()
    return dxdt
