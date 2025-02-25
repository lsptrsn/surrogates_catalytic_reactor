__all__ = [
    "plot_3D",
    "plot_3D_flat",
    "plot_entries",
    "plot_PDE_data",
    "plot_compare_PDE_data",
    "plot_PDE_dynamics_2D",
    "plot_PDE_dynamics",
    "plot_POD_modes",
    "plot_reduced_trajectories"
]


import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import opinf.parameters

Params = opinf.parameters.Params()  # Load parameters from dataclass

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

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': False,
#     'pgf.rcfonts': False,
# })

def plot_3D(z, t_train, t_test, X, name='', function_name='Function f'):
    """
    This function plots a 3D surface with color changing over time and adds a black line
    marking the transition between t_train and t_test.

    Parameters:
    z (numpy.ndarray): The length in meters.
    t_train (numpy.ndarray): The training time in seconds.
    t_test (numpy.ndarray): The testing time in seconds.
    X (numpy.ndarray): The function values to be plotted.
    name (str, optional): The name of the plot. Defaults to ''.
    function_name (str, optional): The name of the function. Defaults to 'Function f'.
    """
    fig = plt.figure(figsize=(10, 8))
    t = np.hstack((t_train, t_test))
    ax = fig.add_subplot(111, projection='3d')

    # Separate training and test data
    x_train, y_train = np.meshgrid(z, t_train)
    x_test, y_test = np.meshgrid(z, t_test)

    # Plot training data
    surf_train = ax.plot_surface(x_train, y_train, X[:, 0:len(t_train)].T,
                                 color=mpi_colors['mpi_grey'],
                                 linewidth=0, antialiased=True, shade=False,
                                 alpha=0.9)

    # Plot test data
    surf_test = ax.plot_surface(x_test, y_test, X[:, len(t_train):].T,
                                color=mpi_colors['mpi_green'],
                                linewidth=0, antialiased=True, shade=False,
                                alpha=0.9)

    # Set axis limits
    ax.set_xlim([np.min(z), np.max(z)])
    ax.set_ylim([np.min(t), np.max(t)])
    ax.set_zlim([np.min(X), np.max(X)])

    # Formatting
    ax.set_xlabel("length $z$ in m", fontsize=20, labelpad=20)
    ax.set_ylabel("time $t$ in s", fontsize=20, labelpad=20)
    ax.set_zlabel(f"{function_name}", fontsize=20, labelpad=20, rotation=90)
    ax.zaxis.set_rotate_label(False)  # Disable automatic rotation
    ax.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.labelpad = 20  # Increase padding to prevent label cut-off
    ax.set_box_aspect(aspect=None, zoom=0.8)

    plt.tight_layout()

    # Save and show the figure
    file_name = f'3D_plot_{name}.svg' if name else '3D_plot.svg'
    plt.savefig(f'./results/{file_name}', format='svg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    return



def plot_3D_flat(z, t, X, name='', function_name='Function f'):
    """
    This function plots a 2D colormap with color changing over time.

    Parameters:
    z (numpy.ndarray): The length in meters.
    t (numpy.ndarray): The time in seconds.
    X (numpy.ndarray): The function values to be plotted.
    name (str, optional): The name of the plot. Defaults to ''.
    function_name (str, optional): The name of the function. Defaults to 'Function f'.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a custom colormap that blends mpi_blue and mpi_red
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)

    # Calculate vmin and vmax for the colormap
    vmin, vmax = np.min(X), np.max(X)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot the colormap
    pcm = ax.pcolormesh(z, t, X.T, cmap=mpi_cmap, norm=norm,
                        shading='auto', rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cbar.set_label(f"{function_name}", fontsize=30,
                   labelpad=20, rotation = 270)  # Increase fontsize here
    cbar.ax.tick_params(labelsize=20)

    # Formatting
    ax.set_xlabel("length $z$ in m", fontsize=20, labelpad=20)  # Increase fontsize here
    ax.set_ylabel("time $t$ in s", fontsize=20, labelpad=20)  # Increase fontsize here
    ax.tick_params(axis='both', which='major', labelsize=20, pad=8)  # Increase fontsize here
    plt.tight_layout()

    # Save and show the figure
    file_name = f'2D_plot_{name}.svg' if name else '2D_plot.svg'
    plt.savefig(f'./results/{file_name}', format='svg',
                bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    return


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


def plot_entries(time_values, entries_train, entries_test):
    # Create a custom colormap with MPI colors
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)

    # Create a figure and axis for conversion plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size
    # Plot first modes of conversion
    color = iter(mpi_cmap(np.linspace(.05, 1, len(entries_test))))
    for j in range(len(entries_train)):
        if j == 0:
            ax.plot(time_values, entries_train[j],
                    color='grey', linewidth=3,
                    label="$T_\mathrm{cool, train}$")
        else:
            ax.plot(time_values, entries_train[j],
                    color='grey', linewidth=3)
    for j in range(len(entries_test)):
        c = next(color)
        indice=f"cool, {j+1}"
        ax.plot(time_values, entries_test[j],
                color=c, linewidth=3,
                label=f"$T_{{\mathrm{{cool, {j+1}}}}}$")

    # Set axis labels and title
    ax.set_xlabel('time $t$ in s', fontsize=20)  # Increase x-axis label font size
    ax.set_ylabel('$T_\mathrm{cool}$ in K', fontsize=20)  # Increase y-axis label font size
    ax.set_title('input over time', fontsize=20)  # Increase title font size
    ax.legend(loc='best', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_ylim(bottom=np.min(entries_train), top=np.max(entries_train))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # Save the conversion plot
    plt.savefig('./results/entires.svg',
                format='svg', bbox_inches='tight', transparent=True)
    plt.show()
    return

def plot_PDE_data(Z, z_all, t, function_name='Function f'):
    """
    Visualize temperature data in space and time.

    Parameters:
    Z (np.ndarray): Temperature data.
    z_all (np.ndarray): Spatial coordinates.
    t (np.ndarray): Time points.
    function_name (str, optional): Function name. Defaults to 'Function f'.
    """

    # Set colormap using MPI colors
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red', 'mpi_grey']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)

    # Create a new figure with specified size and resolution
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.subplot(111)  # Create a subplot

    # Define specific time points to sample and visualize
    sample_t = np.array([1, 20, 100, 200, 300, 320, 600])
    # Find the nearest indices in the time array for the specified sample times
    sample_columns = [_find_nearest_index(t, time) for time in sample_t]

    # Generate colors for each sample time
    color = iter(mpi_cmap(np.linspace(.05, 1, len(sample_columns))))

    # Loop through the selected sample columns and plot each
    for i, j in enumerate(sample_columns):
        q_all = Z[:, j]  # Extract temperature data for the specific time column
        c = next(color)  # Get the next color from the colormap iterator
        ax.plot(z_all, q_all, color=c, linewidth=3, label=f"$t = {float(t[i]):.2f} s$")  # Plot the data

    # Set y-axis limits based on the data range with some margin
    if np.min(Z) > 0:
        ax.set_ylim(np.min(Z) * 0.9, np.max(Z) * 1.1)
    else:
        ax.set_ylim(np.min(Z) * 1.1, np.max(Z) * 1.1)

    # Set x-axis limits based on spatial coordinates
    ax.set_xlim(z_all[0], z_all[-5])

    # Set axis labels and formatting
    ax.set_xlabel("spatial coordinate $z$ in m", fontsize=24)
    ax.set_ylabel(f"{function_name}", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set plot title
    ax.set_title("snapshot data", fontsize=20)
    ax.grid(True)  # Enable grid

    # Hide top and right plot spines for aesthetic reasons
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set major tick locators for both axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Add legend outside of the plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0, frameon=False, fontsize=14)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Define the directory and filename for saving the plot
    directory = './results/'
    file_name = f'snapshot_data_' + function_name.split()[1][0] + '.svg'

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot as an SVG file with a transparent background
    plt.savefig(f'{directory}{file_name}', bbox_inches='tight', transparent=True)

    # Display the plot
    plt.show()

def plot_compare_PDE_data(Z_true, Z_pred, z, t, title, function_name='Function f'):
    """
    Visualize temperature data in space and time and compare predicted and true values.

    Parameters:
    Z_true (np.ndarray): True temperature data.
    Z_pred (np.ndarray): Predicted temperature data.
    z_all (np.ndarray): Spatial coordinates.
    t (np.ndarray): Time points.
    title (str): Plot title.
    function_name (str, optional): Function name. Defaults to 'Function f'.
    """
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.subplot(111)
    sample_t = np.array([1, 20, 100, 200, 300, 450, 800])
    sample_columns = [_find_nearest_index(t, time) for time in sample_t]
    color = iter(mpi_cmap(np.linspace(.05, 1, len(sample_columns))))
    for i, j in enumerate(sample_columns):
        c = next(color)
        q_all_true = Z_true[:, j]
        ax.plot(z, q_all_true, color=mpi_colors['mpi_grey'], linestyle='-', linewidth=3)
        q_all_pred = Z_pred[:, j]
        ax.plot(z, q_all_pred, color=c, linestyle='dashed', linewidth=3, label=f"$t = {sample_t[i]} s$")
    ax.set_ylim(np.min(Z_true) * 0.9, np.max(Z_true) * 1.1)
    ax.set_xlim(z[0], z[-5])
    ax.set_xlabel("spatial coordinate $z$ in m", fontsize=16)
    ax.set_ylabel(f"{function_name}", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    file_name = f'{title.lower().replace(" ", "_")}.svg'
    plt.savefig(f'./results/{file_name}', bbox_inches='tight', transparent=True)
    plt.show()


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
    colors_1 = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors_1)
    colors_2 = [mpi_colors[name] for name in ['mpi_blue', 'mpi_grey', 'mpi_red']]
    mpi_cmap_compare = LinearSegmentedColormap.from_list('Custom', colors_2)
    X_min, X_max = min(np.min(X), np.min(X_pred)), max(np.max(X), np.max(X_pred))
    z_min, z_max = np.min(z), np.max(z)
    t_min, t_max = np.min(t), np.max(t)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Plot each data and its color map
    for ax, data, title in zip(axes, [X, X_pred, np.abs(X - X_pred)], title_list[1:]):
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("length $z$ in m", fontsize=20, labelpad=20)
        ax.set_ylabel("time $t$ in s", fontsize=20, labelpad=20)
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(t_min, t_max)
        ax.tick_params(axis='both', which='major', labelsize=20, pad=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        if title == title_list[-1]:
            vmin, vmax = -X_max/10, X_max/10
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(z, t, data.T, cmap=mpi_cmap_compare, norm=norm, shading='auto', rasterized=True)
        else:
            vmin, vmax = np.min(X), np.max(X)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(z, t, data.T, cmap=mpi_cmap, norm=norm, shading='auto', rasterized=True)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        z_grid, t_grid = np.meshgrid(z, t)
        cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    plt.tight_layout()
    file_name = f'2D_plot_all_{title_list[0]}.svg'
    plt.savefig(f'./results/{file_name}', format='svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()
    return


def plot_PDE_dynamics(z, t, X, X_pred, title_list, function_name='f'):
    """
    Plots the dynamics of a partial differential equation (PDE) using a 3D plot.

    Parameters:
    - z: ndarray, spatial grid
    - t: ndarray, time grid
    - X: ndarray, original PDE solution
    - X_pred: ndarray, predicted PDE solution
    - title_list: list, list of titles for the plots
    - function_name: str, name of the PDE function (default is 'f')

    Returns:
    None
    """

    # Create a custom colormap that blends mpi_blue and mpi_red
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)

    # Define the limits for the axes
    z_min, z_max = np.min(z), np.max(z)
    t_min, t_max = np.min(t), np.max(t)
    X_min, X_max = min(np.min(X), np.min(X_pred)), max(np.max(X), np.max(X_pred))

    # Create the figure and axes
    fig = plt.figure(figsize=(18, 8))
    plt.figtext(0, 0.5, '', fontsize=16, ha='center')
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    # Plot the surfaces
    x, y = np.meshgrid(z, t)
    normalized_t = (t - t_min) / (t_max - t_min)
    color_array = np.array([mpi_cmap(val) for val in normalized_t])
    color_array = np.repeat(color_array[:, np.newaxis, :], x.shape[1], axis=1)

    for ax, data, title in zip(axes, [X, X_pred, (X - X_pred)],
                               title_list[1:]):
        surf = ax.plot_surface(x, y, data.T, facecolors=color_array,
                               linewidth=0,
                               antialiased=True, shade=False, alpha=0.9)
        ax.set_xlabel(r"length $z$", fontsize=20, labelpad=20)
        ax.set_ylabel(r"time $t$", fontsize=20, labelpad=20)
        ax.set_zlabel(function_name, fontsize=20, labelpad=20)
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(t_min, t_max)
        if title == title_list[-1]:
            ax.set_zlim(-X_max/2, X_max/2)
            ax.set_zlim(-X_max/10, X_max/10)
        else:
            ax.set_zlim(X_min, X_max)
        ax.set_title(title, fontsize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis='both', which='major', labelsize=20, pad=8)

    # Save and display the plot
    file_name = f'fig_3D_plot_all_{title_list[0]}_OpInf.svg'
    plt.savefig(f'./results/{file_name}', format='svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()
    return


def plot_POD_modes(z, basis, min_idx, max_idx, title):
    """
    Plots the Proper Orthogonal Decomposition (POD) modes.

    Parameters:
    - z_all: ndarray, spatial grid
    - basis: ndarray, POD basis functions
    - min_idx: int, minimum index of modes to plot
    - max_idx: int, maximum index of modes to plot
    - title: str, title of the plot

    Returns:
    None
    """

    # Create a custom colormap with MPI colors
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red', 'mpi_grey']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)

    # Create a figure and axis for conversion plot
    fig, ax_conv = plt.subplots(figsize=(8, 6))  # Increase figure size
    # Plot first modes of conversion
    color = iter(mpi_cmap(np.linspace(.05, 1, len(range(max_idx-min_idx)))))
    for j in range(min_idx, max_idx):
        c = next(color)
        indices = np.nonzero(basis[:, j])[0]
        submatrix = basis[indices[0]:indices[-1]+1, j]
        ax_conv.plot(z, submatrix,
                     color=c, linewidth=3,
                     label=f"PC {j+1}")

    # Set axis labels and title
    ax_conv.set_xlabel('spatial coordinate $z$ in m', fontsize=20)  # Increase x-axis label font size
    ax_conv.set_ylabel('principal components', fontsize=20)  # Increase y-axis label font size
    ax_conv.set_title(title, fontsize=20)  # Increase title font size
    ax_conv.legend(loc="best", fontsize=18)  # Increase legend font size

    ax_conv.tick_params(axis='both', which='major', labelsize=20)
    ax_conv.set_xlim(left=0, right=2)
    ax_conv.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax_conv.yaxis.set_major_locator(plt.MaxNLocator(3))

    # Save the conversion plot
    plt.savefig(f'./results/reduced_states_'+title+'.svg',
                format='svg', bbox_inches='tight', transparent=True)
    plt.show()
    return



def plot_reduced_trajectories(t, reduced_data, min_idx, max_idx, title):
    """
    Plot the reduced data over time for each dimension.

    Parameters:
    - t (np.ndarray): Time points.
    - reduced_data (np.ndarray): Reduced data to be plotted.

    Returns:
    - None
    """
    # Create a custom colormap with MPI colors
    colors = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red', 'mpi_grey']]
    mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors)
    color = iter(mpi_cmap(np.linspace(.05, 1, len(range(Params.ROM_order)))))
    i = 1
    for idx in range(min_idx, max_idx):
        c = next(color)
        plt.plot(t, reduced_data[idx], color=c, label=f'No. {i}')
        i = i+1
    if title is None:
        plt.title('reduced trajectories', fontsize=16)
    else:
        plt.title(title, fontsize=16)
    plt.xlabel('time in s', fontsize=14)
    plt.ylabel('shifted reduced data', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.show()
    return
