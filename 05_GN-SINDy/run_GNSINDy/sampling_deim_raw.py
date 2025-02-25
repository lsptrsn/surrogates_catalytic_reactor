#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:19:23 2024

@author: forootani
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set up paths
up1 = os.path.abspath('..')
sys.path.insert(0, up1)

from deepymod.data import Dataset
from deepymod.data.samples import Subsample_random
from deepymod.utils import plot_config_file
from deepymod.data.DEIM_class import DEIM

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


def create_data():
    """
    Create data for analysis.

    Returns:
    coords_main_unique (torch.Tensor): Unique coordinates.
    data (torch.Tensor): Processed data.
    """
    # load in data
    data_path = "../data/IEEE_data/case_"+case+"/"
    if case == "1":
        case_name = '_flow_rate_down_interp_state.npy'
    elif case == "2":
        case_name = '_flow_rate_up_interp_state.npy'
    elif case == "3":
        case_name = '_temp_cool_down_interp_state.npy'
    elif case == "4":
        case_name = '_temp_cool_up_interp_state.npy'
    data_conv_all = np.load(data_path+"conversion"+case_name)
    data_temp_all = np.load(data_path+"temperature"+case_name)
    t_o_all = np.load(data_path+"time"+case_name).reshape(-1, 1)
    x_o = np.load(data_path+"z"+case_name).reshape(-1, 1)

    # Define the split index
    split_idx = int(t_o_all.shape[0] * split_ratio)

    # Sequentially split the data
    data_conv_train = data_conv_all[:, :split_idx]
    data_conv_test = data_conv_all[:, split_idx:]

    data_temp_train = data_temp_all[:, :split_idx]
    data_temp_test = data_temp_all[:, split_idx:]

    t_train = t_o_all[:split_idx]
    t_test = t_o_all[split_idx:]

    # get training data
    data_conv_org = data_conv_train
    data_temp_org = data_temp_train
    t_o = t_train

    # Normalize time vector
    t_o = t_o / np.max(t_o)

    # Remove last element from x_o
    x_o = x_o[:-1, :]

    # Perform DEIM for conversion data
    deim_instance_conv = DEIM(data_conv_org, 4, t_o.squeeze(), x_o.squeeze(),
                              tolerance=DEIM_tolerance, num_basis=1)
    S_s_conv, T_s_conv, U_s_conv = deim_instance_conv.execute()

    # Perform DEIM for temperature data
    deim_instance_temp = DEIM(data_temp_org, 4, t_o.squeeze(), x_o.squeeze(),
                              tolerance=DEIM_tolerance, num_basis=1)
    S_s_temp, T_s_temp, U_s_temp = deim_instance_temp.execute()

    # Create coordinates tensors
    coords_conv = torch.from_numpy(np.stack((T_s_conv, S_s_conv),
                                            axis=-1)).reshape(-1, 2)
    coords_temp = torch.from_numpy(np.stack((T_s_temp, S_s_temp),
                                            axis=-1)).reshape(-1, 2)

    # Combine coordinates
    coords_main = np.vstack((coords_conv, coords_temp))
    coords_main_unique = np.unique(coords_main, axis=0)

    # Extract processed data
    data_conv_list = []
    data_temp_list = []
    for i in range(coords_main_unique.shape[0]):
        t_ind = np.where(coords_main_unique[i, 0] == t_o)[0][0]
        x_ind = np.where(coords_main_unique[i, 1] == x_o)[0][0]
        data_conv_list.append(data_conv_org.T[t_ind - 1, x_ind - 1])
        data_temp_list.append(data_temp_org.T[t_ind - 1, x_ind - 1])

    # Convert lists to arrays
    data_conv = np.array(data_conv_list)
    normalized_data_temp = (data_temp_list - np.min(data_temp_list)) / (np.max(data_temp_list) - np.min(data_temp_list))
    data_temp = np.array(normalized_data_temp)

    # Stack data arrays
    data = torch.from_numpy(np.stack((data_conv, data_temp), axis=-1)).float()

    return torch.from_numpy(coords_main_unique).float(), data

# Define the split index
case = "1"
split_ratio = 5/6
DEIM_tolerance = 1e-7

# load in data
data_path = "../data/IEEE_data/case_"+case+"/"
if case == "1":
    case_name = '_flow_rate_down_interp_state.npy'
elif case == "2":
    case_name = '_flow_rate_up_interp_state.npy'
elif case == "3":
    case_name = '_temp_cool_down_interp_state.npy'
elif case == "4":
    case_name = '_temp_cool_up_interp_state.npy'
data_conv_all = np.load(data_path+"conversion"+case_name)
data_temp_all = np.load(data_path+"temperature"+case_name)
t_o_all = np.load(data_path+"time"+case_name).reshape(-1, 1)
x_o = np.load(data_path+"z"+case_name).reshape(-1, 1)

split_idx = int(t_o_all.shape[0] * split_ratio)

# Sequentially split the data
data_conv_train = data_conv_all[:, :split_idx]
data_conv_test = data_conv_all[:, split_idx:]

data_temp_train = data_temp_all[:, :split_idx]
data_temp_test = data_temp_all[:, split_idx:]

t_train = t_o_all[:split_idx]
t_test = t_o_all[split_idx:]

# get training data
data_conv = data_conv_train
data_temp = data_temp_train
t_o = t_train

# Perform DEIM for conversion data
deim_instance_conv = DEIM(data_conv, 4, t_o.squeeze(), x_o.squeeze(),
                          tolerance=DEIM_tolerance, num_basis=1)
S_s_conv, T_s_conv, U_s_conv = deim_instance_conv.execute()

# Perform DEIM for temperature data
deim_instance_temp = DEIM(data_temp, 4, t_o.squeeze(), x_o.squeeze(),
                          tolerance=DEIM_tolerance, num_basis=1)
S_s_temp, T_s_temp, U_s_temp = deim_instance_temp.execute()

t_o_mesh, x_o_mesh = np.meshgrid(t_o, x_o)

# Reshape to 1D arrays for scatter plot
t_o_mesh_1d = t_o_mesh.flatten()
x_o_mesh_1d = x_o_mesh.flatten()
data_conv_1d = data_conv.flatten()
data_temp_1d = data_temp.flatten()/np.max(data_temp)
data_temp_1d = data_temp.flatten()

# Create dataset
x_t, u = create_data()
num_of_samples = 10000000
dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "random_state": 0,
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,
)

# Get coordinates and data
coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()

# Set up colormap
mpi_colors = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255)
}
# Define colors for the plots
colors_1 = [mpi_colors[name] for name in ['mpi_blue', 'mpi_green', 'mpi_red']]
mpi_cmap = LinearSegmentedColormap.from_list('Custom', colors_1)

# Define the path where you want to save the file
save_dir = os.path.join('results', 'DEIM_results')
os.makedirs(save_dir, exist_ok=True)

# Figure size
fig_size = (8, 4)
# reduce for managable figure size
x_o_mesh_1d = x_o_mesh_1d[::50]
t_o_mesh_1d = t_o_mesh_1d[::50]
data_conv_1d = data_conv_1d[::50]
data_temp_1d = data_temp_1d[::50]

# Define custom subplot positions
left_plot_pos = [0.1, 0.15, 0.35, 0.7]  # [left, bottom, width, height]
middle_plot_pos = [0.55, 0.15, 0.35, 0.7]
colorbar_pos = [0.93, 0.15, 0.02, 0.7]  # For colorbar

# Figure 1 - Conversion
fig_1 = plt.figure(figsize=fig_size)
# left plot
ax1 = fig_1.add_axes(left_plot_pos)
im1 = ax1.scatter(S_s_conv, T_s_conv, c=U_s_conv, cmap=mpi_cmap, marker="x", s=20)
ax1.set_title(r'Greedy samples: \texttt{Q-DEIM}', fontsize=20)
ax1.set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax1.set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_aspect('auto')
# right plot
ax2 = fig_1.add_axes(middle_plot_pos)
sc1 = ax2.scatter(x_o_mesh_1d, t_o_mesh_1d, c=data_conv_1d, cmap=mpi_cmap, marker='.')
ax2.set_title(r'conversion in [-]', fontsize=20)
ax2.set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax2.set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.set_aspect('auto')
cbar_ax1 = fig_1.add_axes(colorbar_pos)
cbar1 = fig_1.colorbar(sc1, cax=cbar_ax1)
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.1  # Adjust threshold as needed
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Q_DEIM_conversion.svg'), format='svg', bbox_inches='tight', dpi=50, transparent=True)
plt.show()

# Figure 2 - Temperature
fig_2 = plt.figure(figsize=fig_size)
# left plot
ax3 = fig_2.add_axes(left_plot_pos)
im2 = ax3.scatter(S_s_temp, T_s_temp, c=U_s_temp, cmap=mpi_cmap, marker="x", s=20)
ax3.set_title(r'Greedy samples: \texttt{Q-DEIM}', fontsize=20)
ax3.set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax3.set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
ax3.set_aspect('auto')
# right plot
ax4 = fig_2.add_axes(middle_plot_pos)
sc2 = ax4.scatter(x_o_mesh_1d, t_o_mesh_1d, c=data_temp_1d, cmap=mpi_cmap, marker='.')
ax4.set_title(r'temperature in K', fontsize=20)
ax4.set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax4.set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
ax4.yaxis.set_major_locator(plt.MaxNLocator(5))
ax4.set_aspect('auto')
cbar_ax2 = fig_2.add_axes(colorbar_pos)
cbar2 = fig_2.colorbar(sc2, cax=cbar_ax2)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Q_DEIM_temperature.svg'), format='svg', bbox_inches='tight', dpi=72, transparent=True)
plt.show()

# Figure 3, combined flattened 2d plots
data_temp_1d = data_temp.flatten()/np.max(data_temp)
data_temp_1d = data_temp_1d[::50]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sc1 = axs[0].scatter(x_o_mesh_1d, t_o_mesh_1d, c=data_conv_1d, cmap=mpi_cmap, marker='.')
axs[0].set_title(r'conversion in [-]', fontsize=20)
axs[0].set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
axs[0].set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
cbar1 = fig.colorbar(sc1, ax=axs[0])
sc2 = axs[1].scatter(x_o_mesh_1d, t_o_mesh_1d, c=data_temp_1d, cmap=mpi_cmap, marker='.')
axs[1].set_title(r'temperature in K', fontsize=20)
axs[1].set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
axs[1].set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
cbar2 = fig.colorbar(sc2, ax=axs[1])
plt.tight_layout()
file_name = os.path.join(save_dir, 'Q_DEIM_combined_2D.svg')
plt.savefig(file_name, format='svg', bbox_inches='tight', transparent=True,
            dpi=100)
plt.show()

# # Figure 4 - Conversion + Temperature
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im1 = ax[0].scatter(S_s_conv, T_s_conv, c=U_s_conv, cmap=mpi_cmap, marker="x", s=20)
ax[0].set_title(r'\texttt{Q-DEIM} - Conversion in [-]', fontsize=20)
ax[0].set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax[0].set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax[0].set_xlim(left=0, right=2)
ax[0].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(5))
im2 = ax[1].scatter(S_s_temp, T_s_temp, c=U_s_temp, cmap=mpi_cmap, marker="x", s=20)
ax[1].set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax[1].set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax[1].set_title(r'\texttt{Q-DEIM} - Temperatature in K', fontsize=20)
ax[1].set_xlim(left=0, right=2)
ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(5))
fig.colorbar(mappable=im1, ax=ax[0])
fig.colorbar(mappable=im2, ax=ax[1])
plt.tight_layout()
file_name = os.path.join(save_dir, 'Q_DEIM_conv_temp.svg')
plt.savefig(file_name, format='svg', bbox_inches='tight', transparent=True,
            dpi=300)
plt.show()

# Figure 5, combined Q-Deim Samples
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im3 = ax[0].scatter(coords[:, 1], coords[:, 0], c=data[:, 1],
                    cmap=mpi_cmap, marker="x", s=20)
ax[0].set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax[0].set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax[0].set_xlim(left=0, right=2)
ax[0].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].set_title(r'\texttt{Q-DEIM} Union - Conversion in [-]', fontsize=20)
im4 = ax[1].scatter(coords[:, 1], coords[:, 0], c=data[:, 0],
                    cmap=mpi_cmap, marker="x", s=20)
ax[1].set_ylabel("time $t$ in s", fontsize=20, labelpad=2)
ax[1].set_xlabel("length $z$ in m", fontsize=20, labelpad=2)
ax[1].set_xlim(left=0, right=2)
ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(5))
ax[1].set_title(r'\texttt{Q-DEIM} Union - Temperature in K', fontsize=20)
fig.colorbar(mappable=im3, ax=ax[0])
fig.colorbar(mappable=im4, ax=ax[1])
plt.tight_layout()
file_name = os.path.join(save_dir, 'Q_DEIM_conv_temp_combined.svg')
plt.savefig(file_name, format='svg', bbox_inches='tight', transparent=True,
            dpi=300)
plt.show()
