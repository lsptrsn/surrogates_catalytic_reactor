#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:36:58 2024

@author: peterson
"""
from functools import partial
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import random
import pandas as pd
from scipy.interpolate import interp1d

from parameter import data
#import parameter

random.seed(3)


def choose_flow_rate_profile(profile_name, t_end, Nt, F_in_in,
                         F_in_out, jump_start=None,  wind_profile_start=None,
                         plotting=True):
    """Selects and calls the appropriate inlet profile function.

    Args:
      profile_name: The name of the inlet profile.
      custom_t_end: End time of the simulation.
      custom_Nt: Number of time steps.
      F_in_in: Initial inlet flow rate.
      F_in_out: Final inlet flow rate.
      jump_time: Time of the jump for the 'jump' profile.

    Returns:
      The calculated inlet flow profile.
    """

    profile_functions = {
        'log_saturation': log_saturation,
        'solar_inlet_flow': solar_inlet_flow,
        'jump': jump,
        'wind_profile': wind_profile
    }

    if profile_name not in profile_functions:
        raise ValueError(f"Invalid profile name: {profile_name}")

    elif profile_name == 'jump':
        return profile_functions[profile_name](t_end, Nt,
                                               F_in_in, F_in_out,
                                               jump_start,
                                               plotting)
    elif profile_name == 'wind_profile':
        return profile_functions[profile_name](wind_profile_start, t_end, Nt,
                                               plotting)
    else:
        return profile_functions[profile_name](t_end, Nt,
                                               F_in_in, F_in_out,
                                               plotting)


def choose_cooling_temperature_profile(profile_name, t_end, Nt, T_w_in,
                                       T_w_out, profile_start=0,
                                       plotting=True):
    """Selects and calls the appropriate inlet profile function.

    Args:
      profile_name: The name of the inlet profile.
      custom_t_end: End time of the simulation.
      custom_Nt: Number of time steps.
      F_in_in: Initial inlet flow rate.
      F_in_out: Final inlet flow rate.
      jump_time: Time of the jump for the 'jump' profile.

    Returns:
      The calculated inlet flow profile.
    """

    profile_functions = {
        'ramp_cooling_temperature': ramp_cooling_temperature,
        'random_ramping_temperature': random_ramping_temperature,
        'log_saturation': log_saturation,
    }

    if profile_name not in profile_functions:
        raise ValueError(f"Invalid profile name: {profile_name}")

    else:
        return profile_functions[profile_name](t_end, Nt,
                                               T_w_in, T_w_out, profile_start,
                                               plotting)


def get_log_saturation(x, start, end, steepness=1.0):
    """
    Generates a logarithmic saturation curve that starts at 'start' and ends at 'end'.
    :param x: Input array of values (similar to your 'x' array)
    :param start: Starting value (initial point)
    :param end: Ending value (endpoint)
    :param steepness: Steepness factor (default is 1.0)
    :return: Output array with logarithmic saturation values
    """
    return start + (end - start) * (1 - np.exp(-steepness * x))



@partial(jax.jit, static_argnums=1)
def log_saturation(t_end, t_num, value_in, value_out, profile_start=0, plotting=False):
    """
    Create an inlet flow profile based on a potential solar profile.
    Parameters:
    t_end (int): End time in seconds.
    t_num (int): Number of time steps.
    value_in (float): Input value.
    value_out (float): Output value.
    plotting (bool): Whether the profile should be plotted.

    Returns:
    np.ndarray: Inlet flow profile over time.
    """
    time = np.linspace(0, t_end, t_num)
    profile = get_log_saturation(time, value_in, value_out, 0.1)
    if plotting is True:
        plt.plot(np.linspace(0, t_end, t_num), profile)
        plt.xlabel('Simulation time in s')
        plt.ylabel('Input')
        plt.show()
    return profile


def solar_inlet_flow(t_end, t_num, F_in_in, F_in_out, plotting=False):
    """
    Create an inlet flow profile based on a potential solar profile.
    Parameters:
    t_end (int): End time in seconds (one day are 86400 seconds.)
    t_num (int): Number of time steps.
    F_in_in (float): Minimum flow in Nl/min.
    F_in_out (float): Maximum flow in Nl/min.
    plotting (bool): Whether the profile should be plotted.

    Returns:
    np.ndarray: Inlet flow profile over time.
    """
    # Create a time array
    time_array = np.linspace(0, t_end, t_num)

    # Convert time to hours
    time_hours = (time_array / 3600) % 24

    # Flow profile function based on the hour of the day
    def flow_profile(hour):
        if 0 <= hour < 6 or 22 <= hour <= 24:
            return F_in_in
        else:
            # Scale hour to range [0, 1] for sinusoidal function
            scaled_hour = (hour - 6) / 16
            # Use a sinusoidal function to model the solar profile
            return F_in_in+(F_in_out-F_in_in)*np.sin(np.pi*scaled_hour)
    # Apply the flow profile function to each time step
    profile = np.array([flow_profile(hour) for hour in time_hours])
    if plotting is True:
        plt.plot(np.linspace(0, t_end, t_num), profile)
        plt.xlabel('Simulation time in s')
        plt.ylabel('Flow rate in Nl/min')
        plt.show()
    return profile



def ramp_cooling_temperature(t_end, t_num, value_in, value_out,
                             ramp_start, plotting=False):
    """
    Create a temperature profile with a smooth linear cooling ramp.

    Parameters:
      t_end (int): End time in seconds.
      t_num (int): Number of time steps.
      value_in (float): Lowest temperature (starting value).
      value_out (float): Highest temperature (target value).
      ramp_start (int): Time at which the cooling ramp starts in seconds.
      plotting (bool): Whether the profile should be plotted.

    Returns:
      tuple: (profile, plot_flag)
        - profile (np.ndarray): Temperature profile over time.
        - plot_flag (bool): True if plotting is requested, False otherwise.
    """

    time = np.linspace(0, t_end, t_num)

    # Define temperature change per timestep
    if value_in < value_out:
        slope = 2 / 60  # Convert 2 K/min to K/s
    else:
        slope = - 2/ 60

    # Define the profile with linear interpolation for smooth ramp
    profile = np.full(t_num, value_in, dtype=np.float64)  # Initialize

    # Calculate the slope considering possible limited ramp time
    ramp_start_position = int(ramp_start * t_num / t_end)
    ramp = value_in + slope * (time[ramp_start_position:]
                               - time[ramp_start_position])
    # Define temperature change per timestep
    if value_in < value_out:
        values_remaining = np.minimum(ramp, value_out)
    else:
        values_remaining = np.maximum(ramp, value_out)
    # Apply the linear ramp to the profile after ramp_start
    profile = profile.at[ramp_start_position:].set(values_remaining)

    # Check if target temperature is reached at the end
    if profile[-1] != value_out:
        # Print warning message only if not reached
        print(f"Warning: Not enough time for cooling ramp to reach {value_out} K. Reached {profile[-1]:.2f} K instead.")

    if plotting is True:
        plt.plot(np.linspace(0, t_end, t_num), profile)
        plt.xlabel('Simulation time in s')
        plt.ylabel('Cooling temperature in K')
        plt.show()

    return profile


def jump(t_end, t_num, value_in, value_out, jump_time, plotting=False):
    """
    Create a profile with an instantaneous jump.

    Parameters:
        t_end (int): End time in seconds.
        t_num (int): Number of time steps.
        value_in (float): Initial flow rate.
        value_out (float): Final flow rate after the jump.
        jump_time (int): Time at which the jump occurs in seconds.
        plotting (bool): Whether the profile should be plotted.

    Returns:
        tuple: (profile, plot_flag)
            - profile (np.ndarray): Flow rate profile over time.
            - plot_flag (bool): True if plotting is requested, False otherwise.
    """
    time = np.linspace(0, t_end, t_num)

    # Initialize profile with starting value
    profile = np.full(t_num, value_in, dtype=np.float64)

    # Apply the jump at the specified time step
    jump_time_position = int(jump_time * t_num / t_end)
    profile = profile.at[jump_time_position:].set(value_out)

    if plotting is True:
        plt.plot(time, profile)
        plt.xlabel('Simulation time in s')
        plt.ylabel('Flow rate in Nl/min')
        plt.show()

    return profile


def random_ramping_temperature(t_end, t_num, value_in, value_out,
                               ramp_start, plotting=False):
    """
    Create a temperature profile with random ramps going up and down.

    Parameters:
      t_end (int): End time in seconds.
      t_num (int): Number of time steps.
      value_in (float): Lowest temperature (starting value).
      value_out (float): Highest temperature (target value).
      plotting (bool): Whether the profile should be plotted.

    Returns:
      np.ndarray: Temperature profile over time.
    """

    time = np.linspace(0, t_end, t_num)
    profile = np.full(t_num, value_in, dtype=np.float64)  # Initialize profile
    dt = t_end / t_num  # Time step in seconds
    temp = (value_in+value_out)/2
    ramp_duration = 60  # Ramp change every 60 seconds
    steps_per_ramp = int(ramp_duration / dt)  # Number of steps per ramp

    for i in range(0, t_num, steps_per_ramp):
        # Choose a random slope between 2 K/min and 10 K/min
        slope = random.uniform(2, 10) / 60  # Convert to K/s

        # Generate a random number between 0 and 1
        random_factor = random.uniform(0, 1)
        if random_factor < 0.7:
            # Randomly decide whether to ramp up or down
            if random.choice([True, False]):
                slope *= -1  # Invert slope for cooling down
        else:
            # Favor moving towards the starting temperature
            if temp > (value_in + value_out) / 2:
                slope *= -1  # Cool down if above starting temp
            # Otherwise, keep warming up

        # Create ramp for the next 60 seconds
        for j in range(steps_per_ramp):
            if i + j >= t_num:
                break  # Avoid going beyond the time range
            temp += slope * dt  # Increment temperature
            temp = np.clip(temp, value_in, value_out)  # Keep temp within limits
            profile = profile.at[i + j].set(temp)

    if plotting is True:
        plt.plot(time, profile)
        plt.xlabel('Simulation time in s')
        plt.ylabel('Cooling temperature in K')
        plt.show()

    return profile


def wind_profile(t_start, t_num, t_end,  plotting=False):
    # Define the relative path
    relative_path = '../wind_profile/FINO3_wind_power_PEM_H2_2022.csv'
    # Load the CSV file
    df_wind = pd.read_csv(relative_path, skiprows=1)
    df_wind["time"] = pd.to_datetime(df_wind["time"], format="%d-%b-%Y %H:%M:%S")
    H2_kg_10_min = np.array(df_wind["H2_kg_10min"])

    # Convert H2_kg_10_min to H2_Nl_min
    H2_kg_min = (H2_kg_10_min / 10)
    H2_L_min = H2_kg_min / data["density_H2"]
    H2_Nl_min = H2_L_min / (1.013E5 / data["p_R"]) / (data["T_gas_in"] / 273.15)
    df_wind["H2_Nl_min"] = H2_Nl_min
    df_wind['year_month_day'] = df_wind['time'].dt.to_period('D')
    df_wind['year_month_day'] = df_wind['year_month_day'].astype(str)

    # scale down to one reactor
    df_wind["H2_Nl_min_one_reactor"] = H2_Nl_min/2000
    # scale it up to total flow rate by assuming stoichometric feed and 50% dilution
    df_wind["flow_rate_Nl_min_one_reactor"] = df_wind["H2_Nl_min_one_reactor"]*2.5

    # interpolation
    # only get data for one month
    date = '2022-01-01'
    df_wind = df_wind[df_wind['year_month_day'].str.startswith(date)]
    # Create a time array in minutes, starting from 0, with intervals of 10 minutes
    time_in_seconds = np.arange(0, len(df_wind) * 10, 10)*60
    t_start = 0
    t_num = np.shape(df_wind)[0]*10*60
    t_end = np.max(time_in_seconds)
    data["Nt"] = t_num
    data["t_end"] = t_end
    # parameter.update_dependent_values(data)
    # Create interpolation function for time and H2_Nl_min_one_reactor
    interp_func = interp1d(time_in_seconds, df_wind["flow_rate_Nl_min_one_reactor"], kind='linear')
    time_new = np.linspace(t_start, t_end, t_num)
    flow_rate_interp = interp_func(time_new)

    if plotting is True:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_new-t_start, flow_rate_interp)
        # Add labels and grid
        plt.xlabel("Time in s")
        plt.ylabel("H2 Flow (Nl/min)")
        plt.title("Hydrogen Flow Rate Over Time")
        plt.grid(True)
        plt.show()
        # Show plot

    return flow_rate_interp
