#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:54:50 2024

@author: peterson
"""
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
import os
import pandas as pd
import pickle as pk

# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': False,
#     'pgf.rcfonts': False,
# })

###############################################################################
# COLORS
###############################################################################
# Define your colors once
mpi_colors = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255)
}


def _find_nearest_index(array, value):
    """
    Find the index of the nearest value in an array.

    Parameters:
    array (np.ndarray): Input array.
    value (float): Target value.

    Returns:
    int: Index of the nearest value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    """
    Compute the absolute and relative errors between Qtrue and Qapprox,
    where Qapprox approximates Qtrue.

    Parameters:
    Qtrue (np.ndarray): True data.
    Qapprox (np.ndarray): Approximation to Qtrue.
    norm (function): Function to compute the norm of a matrix.

    Returns:
    float: Absolute error.
    float: Relative error.
    """
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data


def frobenius_error(Qtrue, Qapprox):
    """
    Compute the absolute and relative Frobenius-norm errors between the
    snapshot sets Qtrue and Qapprox, where Qapprox approximates Qtrue.

    Parameters:
    Qtrue (np.ndarray): "True" data.
    Qapprox (np.ndarray): An approximation to Qtrue.

    Returns:
    float: Absolute error.
    float: Relative error.
    """
    # Check dimensions
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim != 2:
        raise ValueError("Qtrue and Qapprox must be two-dimensional")

    # Compute the errors
    return _absolute_and_relative_error(Qtrue, Qapprox,
                                        lambda Z: la.norm(Z, ord="fro"))


def plot_PDE_dynamics_2D(case, gnn, z, t, X, X_pred, title_list, function_name='f'):
    """
    Plot 2D dynamics of PDE data.

    Parameters:
    case (int): The case number for evaluation.
    gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    z (np.ndarray): Spatial coordinates.
    t (np.ndarray): Time points.
    X (np.ndarray): True temperature data.
    X_pred (np.ndarray): Predicted temperature data.
    title_list (list): List of plot titles.
    function_name (str, optional): Function name. Defaults to 'f'.
    """
    case_folder = 'case_' + str(case)

    # Define colors for the plots
    colors_1 = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors_1)
    colors_2 = [mpi_colors[name] for name in ['mpi_blue', 'mpi_grey', 'mpi_red']]
    mpi_cmap_compare = LinearSegmentedColormap.from_list('Custom', colors_2)

    # Determine plot limits
    X_min, X_max = np.min(X), np.max(X)
    z_min, z_max = np.min(z), np.max(z)
    t_min, t_max = np.min(t), np.max(t)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each data and its color map
    for ax, data, title in zip(axes, [X, X_pred, (X - X_pred)], title_list[1:]):
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("length $z$ in m", fontsize=20, labelpad=20)
        ax.set_ylabel("time $t$ in s", fontsize=20, labelpad=20)
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(t_min, t_max)
        ax.tick_params(axis='both', which='major', labelsize=20, pad=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if title == title_list[-1]:
            vmin, vmax = -X_max/50, X_max/50
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(z, t, data.T, cmap=mpi_cmap_compare, shading='auto', rasterized=True)
        else:
            vmin, vmax = np.min(X), np.max(X)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(z, t, data.T, cmap=mpi_cmap, norm=norm, shading='auto', rasterized=True)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        z_grid, t_grid = np.meshgrid(z, t)
        cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    plt.tight_layout()

    # Save and display the plot
    file_name = f'fig_2D_plot_all_{title_list[0]}_{gnn}.svg'
    base_dir = os.getcwd()
    # Construct the directory path
    output_dir = os.path.join(base_dir, 'reports', 'eval', case_folder)

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the file
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, format='svg', bbox_inches='tight', transparent=True, dpi=300)
    #plt.show()
    return


def run_postprocessing(case, gnn, X_true, T_true, z, t, X_pred, T_pred, method_name):
    """
    Run postprocessing on the predicted and true data.

    This function calculates the Frobenius norm error between the true and predicted data,
    prints the relative Frobenius norm error, and plots the 2D dynamics of the conversion
    and temperature data.

    Parameters:
    case (int): The case number for evaluation.
    gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    X_true (np.ndarray): True conversion data.
    T_true (np.ndarray): True temperature data.
    z (np.ndarray): Spatial coordinates.
    t (np.ndarray): Time points.
    X_pred (np.ndarray): Predicted conversion data.
    T_pred (np.ndarray): Predicted temperature data.
    method_name (str): Name of the method used for prediction.

    Returns:
    abs_froerr (float): Absolute Frobenius norm error.
    rel_froerr (float): Relative Frobenius norm error.
    """

    abs_froerr, rel_froerr = frobenius_error(Qtrue=X_true,
                                             Qapprox=X_pred)
    print(f"Relative Frobenius-norm error for X: {rel_froerr:%}")

    abs_froerr, rel_froerr = frobenius_error(Qtrue=T_true,
                                             Qapprox=T_pred)
    print(f"Relative Frobenius-norm error for T: {rel_froerr:%}")

    Q_true = np.vstack((X_true, T_true))
    Q_pred = np.vstack((X_pred, T_pred))
    abs_froerr, rel_froerr = frobenius_error(Qtrue=Q_true,
                                             Qapprox=Q_pred)
    print(f"Absolute Frobenius-norm error: {abs_froerr}")
    print(f"Relative Frobenius-norm error: {rel_froerr:%}")

    # graphics
    plot_PDE_dynamics_2D(case, gnn, z, t, X_true, X_pred,
                          ['conversion', 'conversion in [-] - truth',
                          'conversion in [-] - '+str(method_name),
                          'conversion in [-] - residual'])
    plot_PDE_dynamics_2D(case, gnn, z, t, T_true, T_pred,
                          ['temperature', 'temperature in K - truth',
                          'temperature in K - '+str(method_name),
                          'temperature in K - residual'])
    return abs_froerr, rel_froerr


def run_IEEE_eval(case, gnn, validation_mode, POD_run=False, POD_type="SVD", POD_thresh=150, ensemble_run=False, run_num=""):
    """
    Run IEEE evaluation for the given case.

    This function loads the data and predictions, performs postprocessing,
    and generates plots and error metrics.

    Parameters:
    case (int): The case number for evaluation.
    gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    validation_mode (Boolean): True for training-validation run, False for training-testing run.
    POD_run (Boolean, optional): True for POD+GNN run, False for pure GNN.
    POD_type (str, optional): Type of POD ('SVD', 'PCA'). Defaults to 'SVD'.
    POD_thresh (int, optional): Threshold for POD.
    ensemble_run (Boolean, optional): True for ensemble run, False every other time.
    run_num (str, optional): Identifying number of ensemble run, defaults to empty string ("") for ensemble_run = False.

    Returns:
    rel_err (float): Relative Frobenius norm error.
    """
    base_dir = os.getcwd()
    case_folder = 'case_' + str(case)
    data_folder = os.path.join(base_dir, 'data', 'interim', case_folder)
    pred_folder = os.path.join(base_dir, 'reports', 'predictions', case_folder)
    z = pd.read_csv(f'{data_folder}/z.csv', header=None)
    t = pd.read_csv(f'{data_folder}/test_time.csv', header=None)
    X_pred = pd.read_csv(f'{pred_folder}/c_{gnn}{run_num}.csv', header=0)
    T_pred = pd.read_csv(f'{pred_folder}/T_{gnn}{run_num}.csv', header=0)

    if ensemble_run:
        X_pred = X_pred.iloc[:, 1:]
        T_pred = T_pred.iloc[:, 1:]
    else:
        X_pred = X_pred.iloc[:, 1:]
        T_pred = T_pred.iloc[:, 1:]

    if validation_mode:
        X_true = pd.read_csv(f'{data_folder}/validation_conversion.csv', header=None)
        T_true = pd.read_csv(f'{data_folder}/validation_temperature.csv', header=None)
    else:
        X_true = pd.read_csv(f'{data_folder}/test_conversion.csv', header=None)
        T_true = pd.read_csv(f'{data_folder}/test_temperature.csv', header=None)

    if POD_run:
        POD_folder = os.path.join(base_dir, 'POD', case_folder)

        if POD_type == "SVD":

            U_X_train = pd.read_csv(f'{POD_folder}/U_train_conversion.csv', header=None).iloc[:, :POD_thresh]
            X_pred = np.array(U_X_train).dot(np.array(X_pred))

            U_T_train = pd.read_csv(f'{POD_folder}/U_train_temperature.csv', header=None).iloc[:, :POD_thresh]
            T_pred = np.array(U_T_train).dot(np.array(T_pred))

        elif POD_type == "PCA":

            pca_X = pk.load(open(f'{POD_folder}/pca_train_conversion.pkl',"rb"))
            X_pred = np.transpose(pca_X.inverse_transform(np.transpose(X_pred)))

            pca_T = pk.load(open(f'{POD_folder}/pca_train_temperature.pkl',"rb"))
            T_pred = np.transpose(pca_T.inverse_transform(np.transpose(T_pred)))

        elif POD_type == "manual":
            X_true = X_true.iloc[0:-1:2,:]
            T_true = T_true.iloc[0:-1:2,:]


    # unnormalize temperature and conversion
    T_mu = 0.4533159615958609
    T_sigma = 0.13944152723688819
    T_true = (T_true * T_sigma) + T_mu
    T_pred = (T_pred * T_sigma) + T_mu
    T_min = 400  # should expected lowest T
    T_max = 800  # should be the high risk T
    T_true = T_true * (T_max - T_min) + T_min
    T_pred = T_pred * (T_max - T_min) + T_min

    X_mu = 0.8487244223653506
    X_sigma = 0.2238620396750383
    X_true = (X_true * X_sigma) + X_mu
    X_pred = (X_pred * X_sigma) + X_mu


    abs_err, rel_err = run_postprocessing(case,
                    gnn,
                    np.array(X_true),
                    np.array(T_true),
                    np.array(z).flatten(),
                    np.array(t).flatten(),
                    np.array(X_pred),
                    np.array(T_pred),
                    'GNN')

    return abs_err, rel_err
