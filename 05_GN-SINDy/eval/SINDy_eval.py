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

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
})

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


def plot_PDE_dynamics_2D(z, t, X, X_pred, title_list, function_name='f'):
    """
    Plot 2D dynamics of PDE data.

    Parameters:
    z (np.ndarray): Spatial coordinates.
    t (np.ndarray): Time points.
    X (np.ndarray): True temperature data.
    X_pred (np.ndarray): Predicted temperature data.
    title_list (list): List of plot titles.
    function_name (str, optional): Function name. Defaults to 'f'.
    """
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
            vmin, vmax = -X_max/5, X_max/5
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
    file_name = f'fig_2D_plot_all_{title_list[0]}_SINDy.svg'
    plt.savefig(f'{file_name}', format='svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

    return


def run_postprocessing(X_true, T_true, z, t, X_pred, T_pred, method_name):
    """
    Run postprocessing on the predicted and true data.

    This function calculates the Frobenius norm error between the true and predicted data,
    prints the relative Frobenius norm error, and plots the 2D dynamics of the conversion
    and temperature data.

    Parameters:
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
    plot_PDE_dynamics_2D(z, t, X_true, X_pred,
                         ['conversion', 'conversion in [-] - truth',
                          'conversion in [-] - '+str(method_name),
                          'conversion in [-] - deviation'])
    plot_PDE_dynamics_2D(z, t, T_true, T_pred,
                         ['temperature', 'temperature in K - truth',
                          'temperature in K - '+str(method_name),
                          'temperature in K - deviation'])

    return rel_froerr


if __name__ == "__main__" :
    data_path = 'case_1/3/'
    z = np.load(data_path+"z.npy")[:-1].flatten()
    t_train = np.load(data_path+"t_train.npy")
    t_test = np.load(data_path+"t_test.npy")
    X_true = np.load(data_path+"conversion_train_true.npy")
    T_train_true = np.load(data_path+"temperature_train_true.npy")
    X_train_pred = np.load(data_path+"conversion_train_SINDy.npy")
    T_train_pred = np.load(data_path+"temperature_train_SINDy.npy")
    X_test_true = np.load(data_path+"conversion_test_true.npy")
    T_test_true = np.load(data_path+"temperature_test_true.npy")
    X_test_pred = np.load(data_path+"conversion_test_SINDy.npy")
    T_test_pred = np.load(data_path+"temperature_test_SINDy.npy")
    run_postprocessing(np.array(X_test_true),
                        np.array(T_test_true),
                        z,
                        t_test,
                        np.array(X_test_pred),
                        np.array(T_test_pred),
                        'GN-SINDy')
