from torch_geometric.data import Data, Dataset
import torch
import pandas as pd
import numpy as np
import ipdb
import random

import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm

seed = 1
random.seed(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def plot_train_trajectories(train_trajectory, test_trajectory, path):
    """
    Plot training and test trajectories and save the plot as an image.

    Parameters:
    - train_trajectory (list or array): Training trajectory data.
    - test_trajectory (list or array): Test trajectory data.
    - path (str): File path to save the plot image.
    """
    # Check if trajectories are of the same length; if not, truncate to the shortest
    min_length = min(len(train_trajectory), len(test_trajectory))
    train_trajectory = train_trajectory[:min_length]
    test_trajectory = test_trajectory[:min_length]

    # Create a range for the x-axis based on the trajectory length
    epochs = range(1, min_length + 1)

    # Plotting the training and test trajectories
    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_trajectory, 'o-', label='Training')
    plt.plot(epochs, test_trajectory, 'x-', label='Test')

    # Adding some plot decorations
    plt.xlabel('Epoch')
    plt.ylabel('Root mean squared error (RMSE)')
    plt.legend()
    plt.grid(True)
    fig.savefig(path, dpi=150)


def dataframe_to_gif(df_true, df_pred, output_filepath='trajectory.gif', snapshots_per_second=5):
    """
    Creates a GIF from two DataFrames, showing the evolution of properties over time.

    Parameters:
    - df_true (pd.DataFrame): DataFrame containing true property values.
    - df_pred (pd.DataFrame): DataFrame containing predicted property values.
    - output_filepath (str): File path for the output GIF.
    - snapshots_per_second (int): Number of snapshots to display per second in the GIF.
    """
    # Temporary directory to save individual frames
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)

    # Figure for plotting
    fig, ax = plt.subplots(figsize=(14, 5))

    # Generate and save frames
    filenames = []
    for i, (col1, col2) in enumerate(tqdm(zip(df_true.columns, df_pred.columns), total=len(df_true.columns), desc='Creating gif')):
        ax.clear()
        ax.plot(df_true.index, df_true[col1], label='True value', lw=5, alpha=0.5)
        ax.plot(df_pred.index, df_pred[col2], label='GNN predicted', lw=1)
        ax.set(title=f'Snapshot: {i}', xlabel='Spatial Point', ylabel='Property Value')
        plt.legend()
        plt.tight_layout()

        # Filename for the current frame
        filename = f'{temp_dir}/frame_{i:04d}.png'
        plt.savefig(filename)
        filenames.append(filename)

    # Create GIF
    with imageio.get_writer(output_filepath, mode='I', fps=snapshots_per_second) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove temporary frames
    for filename in filenames:
        os.remove(filename)
    os.rmdir(temp_dir)

    print(f'GIF saved as {output_filepath}')


class ReactorDataset_simple(Dataset):
    def __init__(self, df_z, df_c, df_T):
        """
        Reactor dataset class for simple prediction.

        Parameters:
        - df_z (pd.DataFrame): DataFrame of spatial points.
        - df_c (pd.DataFrame): DataFrame of conversion values.
        - df_T (pd.DataFrame): DataFrame of temperature values.
        """
        super(ReactorDataset_simple, self).__init__()
        self.df_z = df_z
        self.df_c = df_c
        self.df_T = df_T
        self.num_graphs = df_c.shape[1]  # Number of time snapshots / graphs

    def len(self):
        """
        Get the number of graphs in the dataset.

        Returns:
        - int: Number of graphs.
        """
        return self.num_graphs

    def get(self, idx):
        """
        Get a single graph and its target from the dataset.

        Parameters:
        - idx (int): Index of the graph.

        Returns:
        - data (torch_geometric.data.Data): PyTorch Geometric Data object containing node features, edge indices, and edge attributes.
        """
        # Node features: conversion and temperature at each spatial point for a given time snapshot
        node_features = torch.tensor(pd.concat([self.df_c.iloc[:, idx], self.df_T.iloc[:, idx]], axis=1).values, dtype=torch.float)

        # Edge indices: connect each node to its immediate next node to form a chain
        edges_forward = torch.tensor([[i, i + 1] for i in range(self.df_c.shape[0] - 1)], dtype=torch.long)
        edges_backward = edges_forward[:, [1, 0]]  # Reverse the direction of edges
        edge_index = torch.cat([edges_forward, edges_backward], dim=0).t().contiguous()

        # Edge attributes: distances between spatial points (z_j - z_i)
        diff_forward = self.df_z.iloc[1:].values - self.df_z.iloc[:-1].values
        diff_backward = -diff_forward  # Reverse the direction of differences
        edge_attr = torch.tensor(np.concatenate([diff_forward, diff_backward]), dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return data


class ReactorDataset_window(Dataset):
    def __init__(self, df_z, df_c, df_T, window_size, POD_run=False, POD_thresh=150):
        """
        Reactor dataset class for windowed prediction.

        Parameters:
        - df_z (pd.DataFrame): DataFrame of spatial points.
        - df_c (pd.DataFrame): DataFrame of conversion values.
        - df_T (pd.DataFrame): DataFrame of temperature values.
        - window_size (int): Size of the prediction window.
        - POD_run (Boolean, optional): True for POD+GNN run, False for pure GNN.
        - POD_thresh (int, optional): Number of reduced features. Defaults to 150.
        """
        super(ReactorDataset_window, self).__init__()
        self.df_z = df_z
        self.df_c = df_c
        self.df_T = df_T
        self.window_size = window_size
        if POD_run:
            self.spatial_points = POD_thresh
        else:
            self.spatial_points = df_z.shape[0]
        self.num_graphs = df_c.shape[1] - window_size

    def len(self):
        """
        Get the number of graphs in the dataset.

        Returns:
        - int: Number of graphs.
        """
        return self.num_graphs

    def get(self, idx):
        """
        Get a single graph and its target from the dataset.

        Parameters:
        - idx (int): Index of the graph.

        Returns:
        - data_in (torch_geometric.data.Data): PyTorch Geometric Data object
        containing node features and edge indices.
        - x_out (torch.Tensor): Target node features for the next time snapshot.
        """
        # Extract data for the current window
        df_c_window = pd.melt(self.df_c.iloc[:, idx:idx+self.window_size])['value']
        df_T_window = pd.melt(self.df_T.iloc[:, idx:idx+self.window_size])['value']

        # Node features: conversion and temperature at each spatial point for a given time snapshot
        node_features = torch.tensor(pd.concat([df_c_window, df_T_window],
                                               axis=1).values,
                                     dtype=torch.float)

        # Create undirected edges within each time snapshot
        num_nodes = self.spatial_points * self.window_size
        undirected_edges = []
        for i in range(0, num_nodes, self.spatial_points):
            edges_forward = torch.tensor([[j, j + 1]
                                          for j in range(i, i+self.spatial_points-1)],
                                         dtype=torch.long)
            edges_backward = edges_forward[:, [1, 0]]
            edges_within_snapshot = torch.cat([edges_forward, edges_backward], dim=0).t()
            undirected_edges.append(edges_within_snapshot)
        undirected_edges_tensor = torch.cat(undirected_edges, dim=1).contiguous()

        # Create directed edges between corresponding nodes of consecutive time snapshots
        directed_edges_lst = []
        for i in range(0, num_nodes-self.spatial_points, self.spatial_points):
            directed_edges = torch.tensor([[j, j+self.spatial_points]
                                           for j in range(i, i+self.spatial_points)],
                                          dtype=torch.long).t()
            directed_edges_lst.append(directed_edges)
        directed_edges_tensor = torch.cat(directed_edges_lst, dim=1).contiguous()

        # Combine undirected and directed edges
        edge_index = torch.cat([undirected_edges_tensor, directed_edges_tensor],
                               dim=1)

        # Create PyTorch Geometric Data object for input
        data_in = Data(x=node_features, edge_index=edge_index)

        # Target node features for the next time snapshot
        x_out = torch.tensor(pd.concat([self.df_c.iloc[:, idx+self.window_size],
                                        self.df_T.iloc[:, idx+self.window_size]],
                                       axis=1).values, dtype=torch.float)
        
        return data_in, x_out
