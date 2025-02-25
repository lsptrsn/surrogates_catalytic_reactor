"""
Solving Partial-Differential Equation 1D-PFTR for Methanantion
Author: Luisa Peterson
"""
import jax.numpy as np

# Define the data dictionary with default values
data = {
    "T_gas_in": 250+273.15,  # Gas temperature in K
    "T_cool": 260 + 273.15,  # Reactor wall temperature [K]
    "F_in_Nl_min": 10,  # Norm volume flow in Nl/min
    "p_R": 3E5,  # Reactor pressure in Pa
    "x_in": np.array([0.4, 0.1, 0, 0, 0.5]),  # Inlet molar gas composition [-]
    "t_end": 300,  # Final time point for the simulation in s
    "Nt": 1750,  # Number of time steps
    "Nz": 370,  # Number of control volumes for Finite-Volume-Method
    "Nz_all": 385, # Number of all control_volumes including inlet channel

    # Other parameters
    "R": 8.3145,  # Universal gas constant in J/(mol*K)
    "species": ['H2', 'CO2', 'CH4', 'H2O', 'Ar'],  # List with species
    "nue": np.array([-4, -1, 1, 2, 0]),  # Stoichiometric matrix
    # Molar Mass in kg/mol
    "Molar_Mass": np.array([2.01588, 44.0095, 16.0425, 18.01528, 39.948]) * 1e-3,
    "standard_enthalpy_of_formation": -164900,  # in J/mol
    "density_H2": 0.0000899,  # kg/l"
    "T_prop": None,
    "X_CO2_prop": None,

    # Reactor Parameters
    "epsilon": 0.39,  # Specific catalyst volume in (m_cat)^3/(m_ges^3)
    "D_R": 0.01,  # Inner diameter reactor in m
    "L_R": 1.9272727272727272,  # Reactor length in m
    "V_r": None,  # Reaction volume in m^3
    "alpha_wall_out": 2000, # outer wall heat transfer coefficient W/m^2/K [250, 2000]

    # Inlet Parameters
    "X_0": 0,  # Conversion at time 0 [-]
    "X_in": 0,  # Conversion at inlet [-]
    "n_in_total": None,  # Total molar fraction in mol
    "n_in": None,  # Inlet molar gas composition in mol
    "w_in": None,  # Inlet molar gas composition by weight
    "M_gas_in": None,  # Molar Mass of the gas
    "density_fluid_in": None,  # Gas density by ideal gas law in kg/m^3
    "F_in": None,  # Gas flow rate in m^3/s
    "v_gas_in": None,  # Gas velocity in m/s
    # pure component heat capacity # in J/kg/K
    "heat_capacity_pure_T_in" : np.array([14444.53, 1035.75, 3013.09, 1975.07, 520.33]),
    "heat_capacity_gas": None,  # heat capacity for gas in J/kg/K
    "ATR_0": None,  # Adiabatic tempreature rise in K

    # Reaction rate
    "k0_LHHW": 3.46e-6,  # pre-exponential factor in mol Pa^-1 s^-1 g^-1cat LHHW
    "energy_act_LHHW": 77.5e3,  # activation energy in J mol^-1 LHHW
    "k0_PL": 1e-4,  # pre-exponential factor in mol Pa^-1 s^-1 g^-1cat
    "energy_act_PL": 70e3,  # activation energy in J mol^-1
    "n_H2": 0.5,
    "n_CO2": 0.5,

    # Catalyst Parameters
    "dp_cat": None,  # Catalyst particle diameter in m
    "diffusion_volume": np.array([6.12, 26.9, 25.14, 13.1, 16.2]),
    "cat_frac": 2/3,  # catalyst particle fraction in % (= catalyst dilution)
    # core-specific
    "epsilon_core": 0.687,  # porosity core (measurement uncertainties +/- 10%)
    "tau_core": None,  # tortursity core (Bruggeman just approximation)
    "r_core": 0.00125,  # radius core in m
    "d_pore_core": 9.9e-9,  # diameter of pores in the core in m
    "conductivity_thermal_core": 0.19,  # thermal conductivity core in W/(mK)
    "density_cat_core": 3451.4,  # nonporous density solid in core in kg/m^3
    "cp_core": 1107,   # heat capacity core in J/(kgK) -> correlations from VDI
    # shell-specific
    "epsilon_shell": 0.48,  # porosity shell
    "tau_shell": None,  # tortursity shell
    "r_shell": 0.00125, # radius shell in m
    "d_pore_shell": 5e-9,  # diameter of pores in the core in m
    "conductivity_thermal_shell": 0.19,  # thermal conductivity shell in W/(mK)
    "density_cat_shell": 2718.5,  # nonporous density solid in shell in kg/m^3
    "cp_shell": 1107,   # heat capacity core in J/(kgK)
    # r_core = r_shell: fixed-bed dilution
    # r_core < r_shell: catalyst with intert shell

    # Parameters for Finite-Volume-Method
    "zeta_mid": None,
    "D_zeta": None,
    "d_zeta": None,
    "zeta": None,
    "t_span": None,
    "t_points": None,

    # Utility parameters
    "L_to_m3": 1e-3,  # Conversion factor
    "min_to_s": 60.0,  # Conversion factor
    "derivatives": None,  # build derivatives
    "T_scale_add": None,
    "T_scale": None,  # scaling factor for temperature
}


# Define a function to update dependent values based on current data
def update_dependent_values(data):
    data["T_prop"] = np.ones(data["Nz"])*650
    data["X_CO2_prop"] = np.ones(data["Nz"])*0.45
    data["V_r"] = data["L_R"] / 4 * np.pi * data["D_R"] ** 2
    data["n_in_total"] = (data["p_R"] * data["V_r"]
                          / data["R"] / data["T_gas_in"])  # ideal gas law
    data["n_in"] = data["n_in_total"] * data["x_in"]
    data["w_in"] = (data["x_in"] * data["Molar_Mass"]
                    / np.sum(data["x_in"] * data["Molar_Mass"]))
    data["M_gas_in"] = np.sum(data["x_in"] * data["Molar_Mass"])
    data["density_fluid_in"] = (data["p_R"] * data["M_gas_in"]
                                / data["R"] / data["T_gas_in"])
    data["F_in"] = (data["F_in_Nl_min"] * data["L_to_m3"] / data["min_to_s"]
                    * (1.013E5 / data["p_R"]) * (data["T_gas_in"] / 273.15))
    data["T_min"] = np.minimum(np.min(data["T_cool"]), np.min(data["T_gas_in"]))
    data["surface"] = np.pi/4 * data["D_R"] ** 2
    data["surface_free"] = data["surface"]*(1-data["epsilon"])
    data["v_gas_in"] = data["F_in"] / data["surface_free"]
    data["heat_capacity_gas"] = np.sum(data["heat_capacity_pure_T_in"]*data["w_in"])
    data["ATR_0"] = (data["w_in"][1]*-data["standard_enthalpy_of_formation"])\
        / data["Molar_Mass"][1] / data["heat_capacity_gas"]
    data["dp_cat"] = 2*data["r_shell"]
    data["tau_core"] = data["epsilon_core"]**-0.5  # Bruggeman
    data["tau_shell"] = data["epsilon_shell"]**-0.5  # Bruggemann
    data["zeta_mid"] = np.linspace(0, 1, 2 * data["Nz"] + 1)
    data["D_zeta"] = data["zeta_mid"][2:][::2] - data["zeta_mid"][0: -1][::2]
    data["d_zeta"] = data["zeta_mid"][3:][::2] - data["zeta_mid"][1: -2][::2]
    data["zeta"] = data["zeta_mid"][1:][::2]
    data["t_span"] = np.array([0, data["t_end"]])
    data["t_points"] = np.linspace(0, data["t_end"],
                                   num=data["Nt"])
    data["T_scale_add"] = data["T_gas_in"]
    data["T_scale"] = np.max(data["T_cool"]) + data["ATR_0"] - data["T_gas_in"]
    return data
# Call the update_dependent_values function to compute initial dependent values
data = update_dependent_values(data)
