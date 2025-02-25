#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:25:40 2024

@author: peterson
"""

import numpy as np
import os
from scipy.interpolate import CubicSpline

# List of filenames to load
file_names = [
    'conversion_flow_rate_down_long.npy',
    'conversion_flow_rate_up_long.npy',
    'conversion_temp_cool_down_long.npy',
    'conversion_temp_cool_up_long.npy',
    # 'conversion_temp_cool_up_inlet_channel_long.npy',
    'conversion_corrected_flow_rate_down_long.npy',
    'conversion_corrected_flow_rate_up_long.npy',
    'conversion_corrected_temp_cool_down_long.npy',
    'conversion_corrected_temp_cool_up_long.npy',
    # 'conversion_corrected_temp_cool_up_inlet_channel_long.npy',
    'temperature_flow_rate_down_long.npy',
    'temperature_flow_rate_up_long.npy',
    'temperature_temp_cool_down_long.npy',
    'temperature_temp_cool_up_long.npy',
    # 'temperature_temp_cool_up_inlet_channel_long.npy',
    'time_flow_rate_down_long.npy',
    'time_flow_rate_up_long.npy',
    'time_temp_cool_down_long.npy',
    'time_temp_cool_up_long.npy',
    # 'time_temp_cool_up_inlet_channel_long.npy',
    'z_flow_rate_down_long.npy',
    'z_flow_rate_up_long.npy',
    'z_temp_cool_down_long.npy',
    'z_temp_cool_up_long.npy',
    # 'z_temp_cool_up_inlet_channel_long.npy',
]

for file_name in file_names:
    var_name = file_name.replace('.npy', '').replace(' ', '_')
    globals()[var_name] = np.load(file_name)

# Print confirmation
print("All data loaded successfully.")

t_shape = 3000
z_shape = t_shape+1

# function to interpolate data
def interpolate_to_target_shape(data, target_shape=(1000,)):
    x_old = np.linspace(0, 1, data.shape[0])  # Original x-axis
    x_new = np.linspace(0, 1, target_shape[0])  # New x-axis for target shape

    # Interpolation function using cubic spline
    interpolator = CubicSpline(x_old, data, axis=0, bc_type='natural')  # Natural boundary conditions
    return interpolator(x_new)

# Iterate through each parameter, interpolate, reshape, and save
for param_name in file_names:
    param_name_stripped = os.path.splitext(param_name)[0]
    param_data = globals()[param_name_stripped]
    if "derivatives" in param_name:
        end_value_t = int(1 * param_data.shape[-1])
        param_data = param_data[:, ::2]
        interpolated_data = interpolate_to_target_shape(param_data, target_shape=(t_shape,))
        reshaped_data = interpolated_data.reshape(t_shape, t_shape)
    elif "conversion" in param_name or "temperature" in param_name:
        end_value_t = int(1 * param_data.shape[-1])
        param_data = param_data[:, ::2]
        interpolated_data = interpolate_to_target_shape(param_data, target_shape=(z_shape,))
        reshaped_data = interpolated_data.reshape(z_shape, t_shape)
    elif "time_" in param_name:
        end_value_t = int(1 * param_data.shape[-1])
        param_data = param_data[::2]
        reshaped_data = param_data
    else:
        interpolated_data = interpolate_to_target_shape(param_data, target_shape=(z_shape,))
        reshaped_data = interpolated_data

    # Save the reshaped data to _long.npy file with "interp_state" added to the name
    output_file_path = f"{param_name_stripped}_interp_state.npy"
    np.save(output_file_path, reshaped_data)

    # Print confirmation
    print(f"{param_name_stripped} has been interpolated, reshaped to shape {reshaped_data.shape}, and saved as {output_file_path}")
