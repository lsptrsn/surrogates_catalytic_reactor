import jax.numpy as np

from initialize_dynamic_solve import run_PDE_CH4_wrapper


if __name__ == "__main__":
    simulation_params = {
        "T_cool": [250, 270],  # Initial and final wall temperatures in °C (227 °C < T_cool < 476 °C)
        "T_gas_in": 250,  # Inlet gas temperature in °C (set to 250)
        "pressure": 2e5,  # Reactor pressure in Pa (1e5 Pa < p < 10e5 Pa)
        "F_H2": 10*(4/10),  # Hydrogen flow rate in Nl/min (2.40 – 120 nL/min)
        "F_CO2": 10*(1/10),  # Carbon Dioxide flow rate in Nl/min (0.12 – 25   nL/min)
        "F_CH4": 0,  # Methane flow rate in Nl/min (0.12 – 25   nL/min)
        "F_Ar": 20*(5/10),  # Argon flow rate in Nl/min (0.16 – 50   nL/min)
        "start_up": False,  # Perform a cold start up or load in data for start
        # just needed if start_up is set to False
        "initial_values": "start_up",  # Initial conditions or file name
    }

    profile_params = {
        # Input parameter: Multiplier that adjusts the total flow rate. In a
        # PtX context, this could simulate changes in the availability of
        # renewable energy that affect the supply of raw materials. The supply
        # changes without delay, i.e. a profile can be passed directly. Sudden
        # changes in the form of jumps are also possible. Predifined profiles
        # are 'log_saturation', 'solar_inlet_flow', 'wind_profile' or 'jump'.
        "choose_F_in_profile": 'log_saturation',  # Flow rate profile type
        # If your profile is jump, please choose:
        'wind_profile_start': 0,  # start time of the predefined wind profile
        "flow_rate_change": 1,  # set to 1 if not change is wanted
        # If your problem is wind profile, please choose:
        "jump_start": 0,  # Start time for flow rate jump change in s

        # Control parameter: Specifies the type of temperature control profile
        # for the reactor wall. This profile simulates how the cooling
        # temperature is adjusted over time. The control typically runs gradually
        # (2 Kelvin per minute) for both cooling and heating. This can be
        # achieved using the 'ramp_cooling_temperature' function. Alternatively,
        # one can choose 'log_saturartion' or 'random_ramp'
        "choose_T_cool_profile": 'log_saturation',  # Cooling temperature profile type
        "ramp_start": 0,  # Start time for temperature ramp
    }

    numerical_params = {
        "t_end": 60,  # End time of the simulation (seconds)
        "Nz": 371,  # Number of spatial discretization points (385 are in the reactor) w/o inlet channel
        "Nz_all": 371,  # Number of spatial discretization points (385 are in the reactor)
        "Nt": 60*100,  # Number of time steps
        "calculate_derivatives": False,  # Toggle for derivative calculations
        "enable_profile_plot": False,  # Toggle for profile plotting
        "enable_dynamic_plot": False,  # Toggle for dynamic plotting
        "simulation_name": 'temp_cool_up',  # File name for simulation results
    }


    # Run the simulation with specified parameters
    sol = run_PDE_CH4_wrapper(
        simulation_params=simulation_params,
        profile_params=profile_params,
        numerical_params=numerical_params,
    )
