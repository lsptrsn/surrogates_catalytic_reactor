#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:19:23 2024

@author: peterson
"""

__all__ = [
            "sampled_data",
            "train_test_split_time",
            "train_test_split_conditions"
          ]

import numpy as np
import opinf.parameters

Params = opinf.parameters.Params()  # call parameters from dataclass


def sampled_data(t, z, states, derivatives, entries):
    """
    Reduces the input data matrix by taking every xth value for the row
    (spatial coordinate) and every yth axis for the column (time coordinate).
    """
    start_value_t = 0
    end_value_t = int(1 * states.shape[-1])
    end_value_z = int(1 * states.shape[-2])
    z = z[0:end_value_z+2*Params.step_z_sampling:Params.step_z_sampling]
    t = t[start_value_t:end_value_t:Params.step_t_sampling]
    if len(states.shape) == 3:
        states = states[:, :end_value_z+2*Params.step_z_sampling:Params.step_z_sampling,
                        start_value_t:end_value_t:Params.step_t_sampling]
        derivatives = derivatives[:, :end_value_z:Params.step_z_sampling,
                                  start_value_t:end_value_t:Params.step_t_sampling]
        entries = entries[:, start_value_t:end_value_t:Params.step_t_sampling]
    else:
        states = states[:end_value_z+2*Params.step_z_sampling:Params.step_z_sampling,
                        start_value_t:end_value_t:Params.step_t_sampling]
        derivatives = derivatives[:end_value_z:Params.step_z_sampling,
                                  start_value_t:end_value_t:Params.step_t_sampling]
        entries = entries[start_value_t:end_value_t:Params.step_t_sampling]
    return t, z, states, derivatives, entries


def train_test_split_time(matrix, training_split):
    """
    Splits the input matrix into training and test sets based on the time axis.

    Parameters:
    - matrix (np.ndarray): Input array to be split.
    - training_split (float): Fraction of the data to use for training.

    Returns:
    - tuple: (train_set, test_set) split arrays.
    """
    # Determine the number of columns (for 2D) or elements (for 1D)
    num_cols = matrix.shape[-1]
    train_cols = int(num_cols * training_split)

    # Perform the split
    train_set = matrix[..., :train_cols]
    test_set = matrix[..., train_cols:]

    return train_set, test_set


def train_test_split_conditions(matrix, num_trajectories):
    """
    Randomly selects complete trajectories from the input matrix for training
    and testing. The number of trajectories is specified by the user.
    """
    num_cols = matrix.shape[1]
    traj_len = int(num_cols / num_trajectories)
    traj_indices = np.arange(num_cols).reshape(-1, traj_len)
    np.random.shuffle(traj_indices)
    train_indices = traj_indices[:int(num_trajectories * 0.8)].flatten()
    test_indices = traj_indices[int(num_trajectories * 0.8):].flatten()
    train_set = matrix[:, train_indices]
    test_set = matrix[:, test_indices]
    return train_set, test_set
