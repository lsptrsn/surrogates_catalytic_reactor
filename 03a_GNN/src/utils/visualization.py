import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
import numpy as np


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
    fig = plt.figure(figsize=(4,3))
    plt.plot(epochs, train_trajectory, 'o-', label='Training')
    plt.plot(epochs, test_trajectory, 'x-', label='Test')

    # Adding some plot decorations
    plt.xlabel('Epoch')
    plt.ylabel('Root mean squared error (RMSE)')
    plt.legend()
    plt.grid(True)
    fig.savefig(path, dpi=150)


def dataframe_to_gif(df_true, df_pred, gnn, case, output_filepath='trajectory.gif', snapshots_per_second=5):
    """
    Creates a GIF from two DataFrames, showing the evolution of properties over time.

    Parameters:
    - df_true (pd.DataFrame): DataFrame containing true property values.
    - df_pred (pd.DataFrame): DataFrame containing predicted property values.
    - output_filepath (str): File path for the output GIF.
    - snapshots_per_second (int): Number of snapshots to display per second in the GIF.
    """
    # Temporary directory to save individual frames
    temp_dir = 'reports/predictions/case_'+str(case)+'/GIF'
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
        filename = f'{temp_dir}/frame_{i:04d}_{gnn}.png'
        plt.savefig(filename)
        filenames.append(filename)

    snapshots_per_second = int(np.ceil(1000/snapshots_per_second))
    # Create GIF
    with imageio.get_writer(output_filepath, mode='I', duration=snapshots_per_second) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove temporary frames
    for filename in filenames:
        os.remove(filename)
    os.rmdir(temp_dir)

    print(f'GIF saved as {output_filepath}')
