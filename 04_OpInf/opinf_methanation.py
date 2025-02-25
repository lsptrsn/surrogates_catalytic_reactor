#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:31:41 2023

@author: peterson
"""

###############################################################################
# PACKAGES
###############################################################################
import numpy as np
import time
import torch
import os
from scipy.ndimage import gaussian_filter

import opinf

# Check if GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Parameters
###############################################################################
Params = opinf.parameters.Params()  # call parameters from dataclass

###############################################################################
# LOAD DATA
###############################################################################
# Load data
z_all = np.load(f"IEEE_data/{Params.case}/z{Params.file_suffix}_interp_state.npy")
t = np.load(f"IEEE_data/{Params.case}/time{Params.file_suffix}_interp_state.npy")
X_all = np.load(f"IEEE_data/{Params.case}/conversion{Params.file_suffix}_interp_state.npy")
T_all = np.load(f"IEEE_data/{Params.case}/temperature{Params.file_suffix}_interp_state.npy")

if Params.true_derivatives is True:
    Xdot = np.load(f"IEEE_data/{Params.case}/derivatives_conversion{Params.file_suffix}_interp_state.npy")
    Tdot = np.load(f"IEEE_data/{Params.case}/derivatives_temperature{Params.file_suffix}_interp_state.npy")
else:
    Xdot = np.ones_like(X_all[1:])
    Tdot = np.ones_like(T_all[1:])

if 'B' in Params.model_structure:
    entries = np.load(f"data/wall_temperature{Params.file_suffix}_interp_state.npy")
else:
    entries = np.ones(t.size).reshape(-1, 1)

if Params.smoothing is True:
    print('Smoothing is on')
    X_all = gaussian_filter(X_all, sigma=[1, 50])
    T_all = gaussian_filter(T_all, sigma=[1, 50])
    Xdot = gaussian_filter(Xdot, sigma=[1, 50])
    Tdot = gaussian_filter(Tdot, sigma=[1, 50])


# get sample size
_, _, X_all, Xdot, _ = opinf.pre.sampled_data(t, z_all, X_all, Xdot, entries)
t, z_all, T_all, Tdot, entries = opinf.pre.sampled_data(t, z_all, T_all,
                                                        Tdot, entries)

X = X_all[1:]  # leaving out left boundary
T = T_all[1:]  # leaving out left boundary
z = z_all[1:]  # leaving out left boundary

# see shapes
print("# Shapes of all matrices:")
print("_" * 75)
print("z_all shape:", z_all.shape)
print("z shape:", z.shape)
print("t shape:", t.shape)
print("X shape:", X.shape)
print("Xdot shape:", Xdot.shape)
print("T shape:", T.shape)
print("Tdot shape:", Tdot.shape)
Params.batch_size = X.shape[1]

###############################################################################
# DEVIDE INTO TRAINING AND TEST DATA
###############################################################################
training_split = 5/6
# training_split = 1
# List of datasets to split
datasets = [X, X_all, Xdot, T, T_all, Tdot, entries.flatten(), t.flatten()]

# Apply the train-test split to each dataset
split_data = [opinf.pre.train_test_split_time(data, training_split)
              for data in datasets]

# Unpack the split datasets into corresponding variables
((X_train, X_test), (X_all_train, X_all_test), (Xdot_train, Xdot_test),
 (T_train, T_test), (T_all_train, T_all_test), (Tdot_train, Tdot_test),
 (entries_train, entries_test), (t_train, t_test)) = split_data

# Combine the results for X and T
Q_true = np.vstack((X, T))
Q_true_train = np.vstack((X_train, T_train))
Q_true_test = np.vstack((X_test, T_test))
Qdot_true = np.vstack((Xdot, Tdot))
Qdot_true_train = np.vstack((Xdot_train, Tdot_train))
Qdot_true_test = np.vstack((Xdot_test, Tdot_test))
print('Shape of training snapshots:', Q_true_train.shape)
print('Shape of testing snapshots:', Q_true_test.shape)


# see how data evolves in space and time
opinf.utils.plot_3D(z, t_train, t_test, X, name='conversion',
                    function_name='conversion X in [-]')
opinf.utils.plot_3D(z, t_train, t_test, T, name='temperature',
                    function_name='temperature T in K')
opinf.utils.plot_3D_flat(z, t, X, name='conversion',
                         function_name='conversion X in [-]')
opinf.utils.plot_3D_flat(z, t, T, name='temperature',
                         function_name='temperature T in K')

# visualize snapshots to check how the solution looks qualitatively
opinf.utils.plot_PDE_data(X, z, t,
                          function_name='conversion X in [-]')
opinf.utils.plot_PDE_data(T, z, t,
                          function_name='temperature T in K')


###############################################################################
# DATA PREPROCSESSING
###############################################################################
# shift data to have steady state at 0
shift_by_X = np.mean(X_train, axis=1).reshape(-1, 1)
shift_by_T = np.mean(T_train, axis=1).reshape(-1, 1)
X_train_shifted, X_train_ss = opinf.pre.shift(X_train, shift_by=None)
T_train_shifted, T_train_ss = opinf.pre.shift(T_train, shift_by=None)
reference_states = np.concatenate((X_train_ss, T_train_ss))

# check shifted data
opinf.utils.plot_PDE_data(X_train_shifted, z, t_train,
                          function_name='shifted conversion X in [-]')
opinf.utils.plot_PDE_data(T_train_shifted, z, t_train,
                          function_name='shifted temperature T in K')

Q_train_shifted = np.concatenate((X_train_shifted, T_train_shifted))

###############################################################################
# CHOOSE DIMENSION OF THE ROM
###############################################################################
previous_ROM_order = 0
best_model = None
best_threshold = np.inf
# Initialize a dictionary to store the results
results_r = {}
# calculate SVDs
[V_X, S_X] = opinf.basis.pod(states=X_train_shifted,
                             r="full", mode='dense')
[V_T, S_T] = opinf.basis.pod(states=T_train_shifted,
                             r="full", mode='dense')
# loop over different energy treshholds
for energy_treshhold in Params.thresholds:
    print("\n")
    print("# Getting the reduced basis")
    print("_" * 75)
    # 1. FOR CONVERSION X
    r_X_tolerance = opinf.basis.svdval_decay(S_X, tol=Params.tolerance,
                                             normalize=True, plot=True,
                                             name_tag='conversion X in [-]')
    r_X_energy = opinf.basis.cumulative_energy(S_X, energy_treshhold,
                                               plot=True,
                                               name_tag='conversion X in [-]')
    Params.r_X = r_X_energy
    r_X = Params.r_X

    # 2. FOR TEMPERATURE T
    r_T_tolerance = opinf.basis.svdval_decay(S_T, tol=Params.tolerance,
                                             normalize=True, plot=True,
                                             name_tag='temperature T in K')
    r_T_energy = opinf.basis.cumulative_energy(S_T, energy_treshhold,
                                               plot=True,
                                               name_tag='temperature T in K')
    Params.r_T = r_T_energy
    r_T = Params.r_T

    # 3. SUMMING UP
    Params.ROM_order = Params.r_X + Params.r_T
    print(f"Order of reduced model: {Params.ROM_order}")
    # If condition to check if the ROM order has not changed
    if Params.ROM_order == previous_ROM_order:
        print('Same modes as before. I am rising the tolerance.')
        continue  # Skip to the next iteration if the order is the same
    previous_ROM_order = Params.ROM_order
    if energy_treshhold == Params.thresholds[0]:
        opinf.basis.svd_results(S_X, 'conversion X in [-]')
        opinf.basis.svd_results(S_T, 'temperature T in K')

    ###########################################################################
    # CONSTRUCT A LOW-DIMENSIONAL SUBSPACE
    ###########################################################################
    # Extract the first r singular vectors and stack them
    Q_reduced, V_reduced, V_reduced_nonlin, Xi = opinf.basis.get_basis_and_reduced_data(
        V_X, V_T, Q_train_shifted, reference_states)
    Q_reduced_unscaled = Q_reduced
    # plot first modes of conversion
    opinf.utils.plot_POD_modes(z, V_reduced, 0, Params.r_X,
                               'conversion X in [-]')
    opinf.utils.plot_POD_modes(z, V_reduced, r_X, r_X+r_T,
                               'temperature T in K')

    ###########################################################################
    # ESTIMATE TIME DERIVATIVES AND SCALING
    ###########################################################################
    if Params.true_derivatives is True:
        print("\n")
        print('# Model and data set up')
        print("_" * 75)
        print('I have access to the true derivatives')
        # true reduced derivatives
        Qdot_reduced = opinf.utils.reduced_state(Qdot_true_train, V_reduced)

        # scaling
        if Params.scaling is True:
            scaling_fac_X, Q_reduced_X_scaled, Qdot_reduced_X_scaled = \
                opinf.utils.scaled_states(Q_reduced[:r_X, :],
                                          Qdot_reduced[:r_X, :])
            # temperature
            scaling_fac_T, Q_reduced_T_scaled, Qdot_reduced_T_scaled = \
                opinf.utils.scaled_states(Q_reduced[r_X:, :],
                                          Qdot_reduced[r_X:, :])
            # concatenate
            Q_reduced = np.concatenate((Q_reduced_X_scaled,
                                        Q_reduced_T_scaled))
            Qdot_reduced = np.concatenate((Qdot_reduced_X_scaled,
                                           Qdot_reduced_T_scaled))
    else:
        print('I do not have access to the true derivatives')
        # scaling
        if Params.scaling is True:
            # conversion
            scaling_fac_X, Q_reduced_X_scaled = \
                opinf.utils.scaled_states(Q_reduced[:r_X, :])
            # temperature
            scaling_fac_T, Q_reduced_T_scaled = \
                opinf.utils.scaled_states(Q_reduced[r_X:, :])
            Q_reduced = np.concatenate((Q_reduced_X_scaled,
                                        Q_reduced_T_scaled))

        # approximation of reduced derivatives (smoothing via savgol possible)
        Qdot_reduced = opinf.utils.ddt_uniform(Q_reduced, t[1] - t[0], order=4,
                                               smoothing_method=None)

    ###########################################################################
    # Plot reduced trajectories and derivatives
    ###########################################################################
    opinf.utils.plot_reduced_trajectories(
        t_train, Q_reduced, 0, r_X,
        'reduced trajectories conversion X in [-]')
    opinf.utils.plot_reduced_trajectories(
        t_train, Q_reduced, r_X, r_X+r_T,
        'reduced trajectories temperature T in K')
    opinf.utils.plot_reduced_trajectories(
        t_train, Qdot_reduced, 0, r_X,
        'reduced derivatives conversion X in [-]')
    opinf.utils.plot_reduced_trajectories(
        t_train, Qdot_reduced, r_X, r_X+r_T,
        'reduced derivatives temperature T in K')

    ###########################################################################
    # BUILD REDUCED ORDER MODEL
    ###########################################################################
    entries_train = entries_train.reshape(-1, 1)
    Params.input_dim = entries_train.shape[1]
    rom = opinf.models.create_rom()

    ###########################################################################
    # INFER REDUCED-ORDER OPERATORS
    ###########################################################################
    # Train the model and store the results
    time_training_start = time.time()

    model, loss_track = opinf.training.train_model(Q_reduced, Qdot_reduced,
                                                   t_train, entries_train,
                                                   rom, integration=False,
                                                   plotting=True)
    time_training = time.time()-time_training_start
    print('Total training time: ', time_training)
    # Access the 'A' and 'B' attributes of the wrapped model
    A_OpInf, B_OpInf, C_OpInf, H_OpInf = opinf.training.learned_model(model)


    # Define the time span for integration
    t_span = [t[0], t[-1]]

    # Initial reduced state from the training data
    y0 = opinf.utils.reduced_state(Q_train_shifted[:, 0], V_reduced)

    # If scaling is enabled, scale the initial conditions for X and T
    if Params.scaling:
        # Scale the reduced state for X and T components
        _, y0_X = opinf.utils.scaled_states(y0[:r_X],
                                            derivatives=None,
                                            scaling_fac=scaling_fac_X)
        _, y0_T = opinf.utils.scaled_states(y0[r_X:],
                                            derivatives=None,
                                            scaling_fac=scaling_fac_T)
        # Combine scaled states
        y0 = np.concatenate((y0_X, y0_T))
    else:
        # If no scaling, set scaling factors to 1
        scaling_fac_X = scaling_fac_T = 1

    # Integrate the reduced-order model (ROM) over the time span
    time_solving_start = time.time()
    sol_ROM = opinf.models.integrate(t_span, y0, t, entries, model)
    time_solving = time.time()-time_solving_start
    print('Total solving time: ', time_solving)
    sol_reduced = sol_ROM.y

    # Check if the solution has the correct shape for the time points
    if sol_reduced.shape[-1] == t.shape[0]:
        print("\n# Evaluation")
        print("_" * 75)
        print('Solving the IVP was successful')

        # Separate X and T components from the reduced solution
        X_pred_reduced = sol_reduced[:r_X, :]
        T_pred_reduced = sol_reduced[r_X:, :]

        # If scaling was applied, unscale the predicted reduced solutions
        if Params.scaling:
            X_pred_reduced = opinf.utils.unscaled_states(scaling_fac_X,
                                                         X_pred_reduced)
            T_pred_reduced = opinf.utils.unscaled_states(scaling_fac_T,
                                                         T_pred_reduced)
            sol_reduced[:r_X, :] = X_pred_reduced
            sol_reduced[r_X:, :] = T_pred_reduced

        # Reconstruct the full state from the reduced solution
        sol = opinf.utils.full_state(sol_reduced, V_reduced)
        # Unshift the solution using reference states
        sol = opinf.pre.unshift(sol, reference_states)

        # Apply nonlinear POD correction if the basis is not POD
        if Params.basis != 'POD':
            sol_linear = sol  # Store the linear reconstruction
            # Perform polynomial expansion on the reduced states (for nonlinearity)
            poly_update = np.concatenate(opinf.basis.polynomial_form(sol_reduced,
                                                                     p=3), axis=0)
            # Add the nonlinear correction term to the solution
            sol_correction_nonlin = V_reduced_nonlin @ Xi @ poly_update
            sol = sol_linear + sol_correction_nonlin

        # Compute relative Frobenius error and split data into training/testing sets
        sol_train, sol_test = opinf.pre.train_test_split_time(sol, training_split)

        # Evaluate the model on the overall dataset
        print('Results for overall dataset:')
        rel_froerr = opinf.post.run_postprocessing(Q_true, sol, Params, X,
                                                   T, z_all, t, r_X, r_T,
                                                   X_all, T_all)

        # Evaluate the model on the training dataset
        print('Results for training:')
        rel_froerr_train = opinf.post.run_postprocessing(Q_true_train,
                                                         sol_train, Params,
                                                         X_train, T_train,
                                                         z_all, t_train, r_X, r_T,
                                                         X_all_train, T_all_train)

        # Evaluate the model on the testing dataset
        if training_split != 1:
            print('Results for testing:')
            rel_froerr_test = opinf.post.run_postprocessing(Q_true_test,
                                                            sol_test, Params,
                                                            X_test, T_test,
                                                            z_all, t_test,
                                                            r_X, r_T,
                                                            X_all_test, T_all_test)

        # Compute absolute and relative L2 errors
        abs_l2err, rel_l2err = opinf.post.lp_error(t, Q_true, sol)

        # Save the best model with the current threshold
        results_r[energy_treshhold] = {
            "model": model,
            "A_OpInf": A_OpInf,
            "B_OpInf": B_OpInf,
            "C_OpInf": C_OpInf,
            "H_OpInf": H_OpInf,
            "Frobenius norm": rel_froerr
        }

        # Find the model with the lowest Frobenius norm error
        for threshold, result in results_r.items():
            if result["Frobenius norm"] < best_threshold:
                print("*" * 70)
                print('This is the best result so far. I am storing it!')
                print("*" * 70)
                best_threshold = result["Frobenius norm"]
                best_model = result["model"]
                sol_test_best = sol_test
                sol_train_best = sol_train

    else:
        # If the integration failed, print the error message
        print(sol_ROM.message)

# Save the best model if the save_results parameter is enabled
if Params.save_results and results_r.items():
    # opinf.utils.save_results(best_model, Params)
    save_dir = os.path.join('IEEE_eval', Params.case)
    os.makedirs(save_dir, exist_ok=True)

    # Dictionary of arrays to save
    arrays = {
        "Q_true_test": Q_true_test,
        "Q_true_train": Q_true_train,
        "sol_test": sol_test_best,
        "sol_train": sol_train_best,
        "t_test": t_test,
        "t_train": t_train,
        "z": z,
        "z_all": z_all,
    }
    # Save arrays as .npy files
    for name, array in arrays.items():
        filepath = os.path.join(save_dir, f"{name}.npy")
        np.save(filepath, array)
    print(f"Arrays saved in: {save_dir}")
