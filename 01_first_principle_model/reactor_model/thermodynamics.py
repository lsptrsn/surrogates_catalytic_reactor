import jax.numpy as np
import jax

from parameter import data
from utils import replace_out_of_bounds_values
eps = np.finfo(float).eps


@jax.jit
def pure_cp(T):
    """
    Calculate the specific heat capacity for pure components in J/molK.

    Parameters:
    - T: Temperature in Kelvin.

    Returns:
    - cp_pure: Specific heat capacity for each pure component.
    """
    T = np.atleast_1d(T)
    R = [data["R"]/data["Molar_Mass"][0],  # H2
         data["R"]/data["Molar_Mass"][1],  # CO2
         data["R"]/data["Molar_Mass"][2],  # CH4
         data["R"]/data["Molar_Mass"][3],  # H2O
         data["R"]/data["Molar_Mass"][4],  # N2
         ]

    substances = {
        "H2": [392.8422, 2.4906, -3.6262, -1.9624, 35.6197, -81.3691, 62.6668, R[0]],
        "CO2": [514.5073, 3.4923, -0.9306, -6.0861, 54.1586, -97.5157, 70.9687, R[1]],
        "CH4": [1530.8043, 4.2038, -16.615, -3.5668, 43.0563, -86.5507, 65.5986, R[2]],
        "H2O": [706.3032, 5.1703, -6.0865, -6.6011, 36.2723, -63.0965, 46.2058, R[3]],
        "N2": [0, 2.5, 2.5, 0, 0, 0, 0, R[4]]
    }

    def calculate_cp(coeffs, R, T):
        A, B, C, D, E, F, G = coeffs
        T_ratio = T / (A + T)
        return (B + (C - B) * T_ratio**2
                * (1 - A / (A + T) * (D + E * T_ratio + F * T_ratio**2 + G * T_ratio**3))) * R

    cp_pure = np.zeros((len(data["species"]), len(T)))
    for idx, (substance, coeffs) in enumerate(substances.items()):
        cp_pure = cp_pure.at[idx].set(calculate_cp(coeffs[:-1], coeffs[-1], T))
    cp_pure = replace_out_of_bounds_values(cp_pure, eps)
    return cp_pure


@jax.jit
def pure_component_heat_conductivity(T):
    """
    Calculate pure component heat conductivity based on temperature in W/(mK).

    Parameters:
    - T: Temperature in Kelvin.

    Returns:
    - lambda_pure: Heat conductivity values for each pure component.
    """
    # Coefficients based on VDI
    p = [
        [0.651*10**-3, 0.76730*10**-3, -0.687050*10**-6,
         0.506510*10**-9, -0.138540*10**-12],  # H2
        [-3.882*10**-3,  0.05283*10**-3, 0.071460*10**-6,
         -0.070310*10**-9, 0.01890*10**-12],  # CO2
        [8.154*10**-3, 0.00811*10**-3, 0.351530*10**-6,
         -0.338650*10**-9, 0.140920*10**-12],  # CH4
        [13.918*10**-3, -0.04699*10**-3, 0.258066*10**-6,
         -0.183149*10**-9, 0.055092*10**-12],  # H2O
        [4.303*10**-3, 0.04728*10**-3, 0.007780*10**-6,
         0, 0, ]
    ]
    lambda_pure = np.array([
        p[0][0]+p[0][1]*T+p[0][2]*T**2+p[0][3]*T**3+p[0][4]*T**4,
        p[1][0]+p[1][1]*T+p[1][2]*T**2+p[1][3]*T**3+p[1][4]*T**4,
        p[2][0]+p[2][1]*T+p[2][2]*T**2+p[2][3]*T**3+p[2][4]*T**4,
        p[3][0]+p[3][1]*T+p[3][2]*T**2+p[3][3]*T**3+p[3][4]*T**4,
        p[4][0]+p[4][1]*T+p[4][2]*T**2+p[4][3]*T**3+p[4][4]*T**4,
    ])
    lambda_pure = replace_out_of_bounds_values(lambda_pure, eps)
    return lambda_pure


@jax.jit
def pure_component_viscosity(T):
    """
    Calculate pure component viscosity in Pa s.

    Parameters:
    - T: Temperature in Kelvin.

    Returns:
    - viscosity_pure: Viscosity values for each pure component.
    """
    p = [
        [0.18024*10**-5, 0.27174*10**-7, -0.13395*10**-10,
         0.00585*10**-12, -0.00104*10**-15],  # H2
        [-0.18024*10**-5, 0.65989*10**-7, -0.37108*10**-10,
         0.01586*10**-12, -0.00300*10**-15],  # CO2
        [-0.07759*10**-5, 0.50484*10**-7, -0.43101*10**-10,
         0.03118*10**-12, -0.00981*10**-15],  # CH4
        [0.64966*10**-5, -0.15102*10**-7, 1.15935*10**-10,
         -0.10080*10**-12, 0.03100*10**-15],  # H2O
        [0.16196*10**-5, 0.81279*10**-7, -0.41263*10**-10,
         0.01668*10**-12, -0.00276*10**-15]  # N2
    ]
    viscosity_pure = np.array([
        p[0][0]+p[0][1]*T+p[0][2]*T**2+p[0][3]*T**3+p[0][4]*T**4,
        p[1][0]+p[1][1]*T+p[1][2]*T**2+p[1][3]*T**3+p[1][4]*T**4,
        p[2][0]+p[2][1]*T+p[2][2]*T**2+p[2][3]*T**3+p[2][4]*T**4,
        p[3][0]+p[3][1]*T+p[3][2]*T**2+p[3][3]*T**3+p[3][4]*T**4,
        p[4][0]+p[4][1]*T+p[4][2]*T**2+p[4][3]*T**3+p[4][4]*T**4,
    ])
    viscosity_pure = replace_out_of_bounds_values(viscosity_pure, eps)
    return viscosity_pure


@jax.jit
def get_cp_fluid(T, x):
    """
    Calculate the specific heat capacity of the fluid.

    Parameters:
    - T: Temperature in Kelvin.
    - x: Mass fraction of each component.

    Returns:
    - cp_fluid: Specific heat capacity of the fluid.
    """
    w = x * data["Molar_Mass"][:, None] / \
        np.sum(x * data["Molar_Mass"][:, None], axis=0)
    cp_fluid_pure = pure_cp(T).T
    cp_fluid = np.sum(w.T * cp_fluid_pure, axis=1).flatten()
    return cp_fluid


@jax.jit
def interaction_parameter_phi(T, x):
    """
    Calculate the interaction parameter phi according to Pohling et al.

    Parameters:
    - T: Temperature in Kelvin.
    - x: Mass fraction of each component.

    Returns:
    - phi_sum: Interaction parameter phi.
    """
    try:
        len_T = len(T)
        T = T[0]
        x = x[:, 0]
    except:
        len_T = 1

    viscosity_pure = pure_component_viscosity(T)
    Molar_Mass = data["Molar_Mass"]

    # Calculate phi_ij using vectorized operations
    viscosity_1 = viscosity_pure[:, np.newaxis]
    viscosity_2 = viscosity_pure[np.newaxis, :]
    Molar_Mass_1 = Molar_Mass[:, np.newaxis]
    Molar_Mass_2 = Molar_Mass[np.newaxis, :]

    phi_ij = (1 + (viscosity_1 / viscosity_2 + eps) ** 0.5 *
              (Molar_Mass_2 / Molar_Mass_1 + eps) ** 0.25) ** 2 \
        / (8 * (1 + Molar_Mass_1 / Molar_Mass_2 +eps)) ** 0.5

    # Calculate phi_sum using vectorized operations
    phi_sum = np.dot(x, phi_ij.T)
    # Duplicate the result by the length of T
    phi_sum = np.tile(phi_sum, (len_T, 1)).T
    return np.array(phi_sum)


@jax.jit
def get_lambda_fluid(T, x):
    """
    Calculate the heat conductivity of the fluid.

    Parameters:
    - T: Temperature in Kelvin.
    - x: Mass fraction of each component.

    Returns:
    - lambda_fluid: Heat conductivity of the fluid.
    """
    lambda_pure = pure_component_heat_conductivity(T).T
    phi_sum = interaction_parameter_phi(T, x)
    lambda_fluid = np.sum((x.T * lambda_pure / phi_sum.T), axis=1).flatten()
    lambda_fluid = replace_out_of_bounds_values(lambda_fluid, eps)
    return lambda_fluid


@jax.jit
def get_viscosity_fluid(T, x):
    """
    Calculate the viscosity of the fluid.

    Parameters:
    - T: Temperature in Kelvin.
    - x: Mass fraction of each component.

    Returns:
    - viscosity_fluid: Viscosity of the fluid.
    """
    viscosity_pure = pure_component_viscosity(T).T
    phi = interaction_parameter_phi(T, x)
    return np.sum((x.T * viscosity_pure / phi.T)).flatten()


@jax.jit
def get_reaction_enthalpy(T):
    """
    Calculate the reaction enthalpy.

    Parameters:
    - T: Temperature in Kelvin.

    Returns:
    - reaction_enthalpy: Reaction enthalpy.
    """
    # 5th order polynomial adapted to the reaction enthalpies from VDI Wärmeatlas
    p = np.array([-1.17506718366332e-10, 4.92506928016944e-07, -0.000808214721985996,
                  0.660624551702347, -237.268346322685, -135868.273143470])
    return np.polyval(p, T)


@jax.jit
def get_heat_transfer_coeff(T, x_prop, v_gas, rho_fluid):
    """
    Calculate the heat transfer coefficient.

    Parameters:
    - T: Temperature in Kelvin.
    - x_prop: Mass fraction property.
    - v_gas: Gas velocity.
    - rho_fluid: Fluid density.

    Returns:
    - U: Heat transfer coefficient (with respect to average temperature).
    - U_slash: Modified heat transfer coefficient (with respect to bed center).
    - LAMBDA_AX: Effective axial thermal conductivity.
    """
    # fitting parameters
    scale_alpha_wall = 2.72263664
    correction = 1.01050855

    # Constants
    C_f, sigma, emi, phi = 1.25, 5.67e-08, 0.4, 0.0077

    # Calculating lambda_cat
    lmbd_core_eff = (1 - data["epsilon_core"]) / \
        data["tau_core"] * data["conductivity_thermal_core"]
    lmbd_shell_eff = (1 - data["epsilon_shell"]) / \
        data["tau_shell"] * data["conductivity_thermal_shell"]
    r_shell_cubed = data["r_shell"] ** 3
    r_core_cubed = data["r_core"] ** 3
    a1 = lmbd_shell_eff / lmbd_core_eff
    a2 = (r_shell_cubed - r_core_cubed) / r_shell_cubed
    lambda_cat = (1 + (a1 - 1) * a2 - ((a1 - 1) ** 2 * a2 *
                  (1 - a2)) / (3 * a1 + (a1 - 1) * a2)) * lmbd_core_eff

    # Using memoized functions
    lambda_fluid = get_lambda_fluid(data["T_prop"], x_prop)
    viscosity_fluid = get_viscosity_fluid(data["T_prop"], x_prop)
    cp_fluid = get_cp_fluid(data["T_prop"], x_prop)

    # Further calculations
    B = C_f * ((1 - data["epsilon"]) / data["epsilon"]) ** (10 / 9)
    k_p = lambda_cat / lambda_fluid
    k_rad = 4 * sigma / (2 / emi - 1) * T ** 3 * data["dp_cat"] / lambda_fluid
    N = 1 + (k_rad - B) / k_p

    k_c = 2 / N * ((B * (k_p + k_rad - 1)) / (N ** 2 * k_p) * np.log((k_p + k_rad) / B) +
                   (B + 1) / (2 * B) * (k_rad - B) - (B - 1) / N)
    k_bed = (1 - (1 - data["epsilon"]) ** 0.5) * data["epsilon"] * (data["epsilon"] ** (-1) + k_rad) \
        + (1 - data["epsilon"]) ** 0.5 * (phi * k_p + (1 - phi) * k_c)
    lambda_bed = k_bed * lambda_fluid

    # Effective radial heat transfer coefficient in the fixed bed reactor is
    # calculated according to the correlation of Yagi and Kunii
    Pe_0 = v_gas * rho_fluid * cp_fluid * data["dp_cat"] / lambda_fluid
    K_r = 7 * (2 - (1 - (2 / (data["D_R"] / data["dp_cat"]))) ** 2)
    K_ax = 2
    LAMBDA_R = lambda_bed + Pe_0 / K_r * lambda_fluid
    LAMBDA_AX = lambda_bed + Pe_0 / K_ax * lambda_fluid

    Re_0 = v_gas * data["dp_cat"] * rho_fluid / viscosity_fluid
    Pr = viscosity_fluid * cp_fluid / lambda_fluid
    alpha_wall_int = lambda_fluid / data["dp_cat"] * \
        (((1.3 + 5 / (data["D_R"] / data["dp_cat"]))
          * (lambda_bed / lambda_fluid))
         + 0.19 * Re_0**0.75 * Pr**0.33)
    alpha_wall = (1 / alpha_wall_int + 1 / data["alpha_wall_out"])**(-1)
    alpha_wall = alpha_wall*scale_alpha_wall

    Bi = alpha_wall / LAMBDA_R * data["D_R"] / 2
    U = (1 / alpha_wall + (data["D_R"] / 2 * 1 /
         (3 * LAMBDA_R)) * (Bi + 3) / (Bi + 4))**(-1)
    U_slash = alpha_wall * (Bi**4 + 24 * Bi**3
                            + 240 * Bi**2 + 1152 * Bi + 2304) \
        / (16 * (Bi**2 + 6 * Bi + 12)**2)
    U_slash = U_slash + correction*T - 1*T
    return U, U_slash, LAMBDA_AX
