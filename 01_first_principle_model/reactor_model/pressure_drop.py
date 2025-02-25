"""
Created on Fri Mar 17 10:07:51 2023

@author: peterson
"""
import thermodynamics as thermo
import jax
from parameter import data


@jax.jit
def pressure_drop(T_in, x_in, rho_fluid_in, v_gas_in):
    """
    Calculate pressure drop according to Eisfeld and Schnitzlein.

    Parameters:
    T_in : float
        Input temperature.
    x_in : array
        Molar fraction of input components.
    rho_fluid_in : float
        Density of the input fluid.
    v_gas_in : float
        Velocity of the gas input.

    Returns:
    dpdz : float
        Pressure drop per unit length.
    """
    # Constants
    K_1 = 154
    k_1 = 1.15
    k_2 = 0.87

    # Calculate fluid viscosity
    viscosity_fluid = thermo.get_viscosity_fluid(T_in, x_in)

    # Auxiliary variables for pressure drop calculation
    A_w = 1 + 2 / (3 * (data["D_R"] / data["dp_cat"]) * (1 - data["epsilon"]))
    B_w = (k_1 * (data["dp_cat"] / data["D_R"])**2 + k_2)**2
    Re_dp = rho_fluid_in * v_gas_in * data["dp_cat"] / viscosity_fluid

    # Dimensionless pressure loss
    Psi = (K_1 * A_w**2 / Re_dp * (1 - data["epsilon"])**2 / data["epsilon"]**3
           + A_w / B_w * (1 - data["epsilon"]) / (data["epsilon"]**3))

    # Pressure loss in bar
    dpdz = Psi * rho_fluid_in * v_gas_in**2 / data["dp_cat"]
    return dpdz
