import os
import jax.numpy as np
import random
import warnings

from parameter import data
import parameter
import inlet_profiles
from handle_dynamic_solve import run_reactor_model_dynamic
from utils import replace_out_of_bounds_values

warnings.simplefilter("ignore")
eps = np.finfo(float).eps


def run_PDE_CH4_wrapper(simulation_params, profile_params, numerical_params):
    """
    Parameters:
    - simulation_params (dict): Simulation-specific parameters.
    - profile_params (dict): Parameters for inlet profiles.
    - numerical_params (dict): Numerical settings and controls.
    - data (dict): Data structure to store parameters.

    Returns:
    - sol: Simulation results.
    """
    T_gas_in = simulation_params["T_gas_in"] + 273.15  # Convert to Kelvin
    T_cool_in = simulation_params["T_cool"][0] + 273.15  # Convert to Kelvin
    T_cool_out = simulation_params["T_cool"][1] + 273.15  # Convert to Kelvin
    flow_rates = np.array([simulation_params["F_H2"],
                           simulation_params["F_CO2"],
                           simulation_params["F_CH4"],
                           0,
                           simulation_params["F_Ar"]])
    flow_rates = flow_rates.astype(float)

    # Handle flow rates close to zero
    flow_rates = np.where(flow_rates < eps, eps, flow_rates)

    data.update({
        "T_gas_in": T_gas_in,
        "p_R": simulation_params["pressure"],
        "x_in": flow_rates / np.sum(flow_rates),
        "t_end": numerical_params["t_end"],
        "Nt": numerical_params["Nt"],
        "Nz": numerical_params["Nz"],
        "Nz_all": numerical_params["Nz_all"],
        "derivatives": numerical_params["calculate_derivatives"]
    })


    F_in_profile = inlet_profiles.choose_flow_rate_profile(
        profile_params["choose_F_in_profile"],
        data["t_end"],
        data["Nt"],
        np.sum(flow_rates),
        np.sum(flow_rates) * profile_params["flow_rate_change"],
        profile_params["jump_start"],
        profile_params["wind_profile_start"],
        numerical_params["enable_profile_plot"]
    )

    F_in_profile = replace_out_of_bounds_values(F_in_profile, eps)


    # Generate temperature and inlet flow profiles
    T_cool_profile = inlet_profiles.choose_cooling_temperature_profile(
        profile_params["choose_T_cool_profile"],
        data["t_end"],
        data["Nt"],
        T_cool_in,
        T_cool_out,
        profile_params["ramp_start"],
        numerical_params["enable_profile_plot"]
    )


    data.update({
        "F_in_Nl_min": F_in_profile,
        "T_cool": T_cool_profile,
    })

    # Check for zero flow rates
    if np.sum(flow_rates) < 1e-10:
        print('Information: Flow rate is zero')
        F_in_profile_eps = np.ones_like(F_in_profile) * eps
        data.update({
            'x_in': np.array([eps, eps, 0, 0, 1]),
            "F_in_Nl_min": F_in_profile_eps
        })

    # Update dependent values in data
    parameter.update_dependent_values(data)

    # Set initial values
    if simulation_params["start_up"] is True:
        iv_X_CO2 = np.ones((data["Nz"], 1)) * data["X_0"]
        x_values = np.linspace(0, 1, data["Nz"])
        iv_T = inlet_profiles.get_log_saturation(
            x_values, data["T_gas_in"], data["T_cool"][0], 10).reshape(-1, 1)
    else:
        try:
            results_folder = os.path.join(os.getcwd(), "results")
            file_name_X = 'conversion_' \
                + simulation_params["initial_values"] +  '.npy'
            iv_X_CO2 = np.load(os.path.join(results_folder,
                                            file_name_X))[1:, -1]
            file_name_T = 'temperature_' \
                + simulation_params["initial_values"] +  '.npy'
            iv_T = np.load(os.path.join(results_folder,
                                        file_name_T))[1:, -1]
        except FileNotFoundError:
            print('Data not available. Performing cold start up.')
            iv_X_CO2 = np.ones((data["Nz"], 1)) * data["X_0"]
            x_values = np.linspace(0, 1, data["Nz"])
            iv_T = inlet_profiles.get_log_saturation(x_values, data["T_gas_in"], data["T_cool"][0], 10).reshape(-1, 1)

    # Validate input ranges
    if not (500 <= T_gas_in <= 750):
        print('Warning: Inlet temperature is not between 500 and 750 K.')
    if not (1e5 <= simulation_params["pressure"] <= 10e5):
        print('Warning: Pressure is not between 1e5 and 10e5 Pa.')
    if (np.sum(flow_rates) > 20) or (np.sum(flow_rates) * profile_params["flow_rate_change"] > 20):
        print('Warning: Flow rate is greater than 20 Nl/min.')
    if not np.all((500-273.15 <= np.array(simulation_params["T_cool"])) & (np.array(simulation_params["T_cool"]) <= 750-273.15)):
        print('Warning: Cooling temperature is not between 500 and 750 K.')

    # Run the reactor model
    sol = run_reactor_model_dynamic(
        initial_values=[iv_X_CO2, iv_T],
        simulation_name=numerical_params["simulation_name"],
        dynamic_plot=numerical_params["enable_dynamic_plot"]
    )

    return sol
