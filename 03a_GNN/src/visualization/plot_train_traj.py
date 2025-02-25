from src.utils.visualization import plot_train_trajectories
import pandas as pd


def plot_train_traj(case, gnn):
    """
    Plot training trajectories.

    Parameters:
    - case (int): The case number.
    - gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').

    Returns:
    None
    """
    # Define the folder for the specific case
    case_folder = 'case_' + str(case)

    # Load the training trajectory data
    train_traj_folder = f'reports/training_traj/{case_folder}'
    df_training_traj = pd.read_csv(f'{train_traj_folder}/train_traj_{gnn}.csv')

    # Define the path to save the plot
    plot_path = f'{train_traj_folder}/train_traj_{gnn}.png'

    # Plot the training trajectories and save the plot
    plot_train_trajectories(df_training_traj['Train_rmse'].to_numpy(),
                            df_training_traj['Test_rmse'].to_numpy(),
                            plot_path)
    return
