from src.utils.gnn_architectures import GCN, GAT, GGNN, RGGNN
from src.utils.train_test_predict import predict_future_snapshots, predict_future_snapshots_window, shift_time_snapshot
import os
import torch
import time


def predict(case, hidden_dim, gnn, window_size, num_future_snapshots, run_num="", seed=1):
    """
    Predict future snapshots using the trained GNN model.

    Parameters:
    - case (int): The case number.
    - hidden_dim (int): Dimension of the hidden layers in the GNN.
    - gnn (str): Type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    - window_size (int or None): Size of the window for window-based dataset, or None for simple dataset.
    - num_future_snapshots (int): Number of future snapshots to predict.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """
    case_folder = 'case_' + str(case)

    # Set seeds for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Load the training dataset
    train_dataset = torch.load(f'data/processed/{case_folder}/train_{gnn}.pt')

    # Get the last training graph and its initial state
    if window_size is None:
        last_training_graph = train_dataset.get(len(train_dataset)-1)
        c_0 = last_training_graph.x[:, 0].cpu().numpy()
        T_0 = last_training_graph.x[:, 1].cpu().numpy()
    else:
        window_size = int(window_size)
        last_training_graph, last_target = train_dataset.get(len(train_dataset)-1)
        # Update input graph to get the very last information from the training set
        shift_time_snapshot(last_training_graph.x, last_target)
        c_0 = last_target[:, 0].cpu().numpy()
        T_0 = last_target[:, 1].cpu().numpy()

    # Initialize the GNN model based on the chosen architecture
    if gnn == 'GCN':
        model = GCN(node_feature_dim=2, hidden_dim=hidden_dim,
                    window_size=window_size)
    elif gnn == 'GAT':
        model = GAT(node_feature_dim=2, hidden_dim=hidden_dim,
                    window_size=window_size)
    elif gnn == 'GGNN':
        model = GGNN(node_feature_dim=2, hidden_dim=hidden_dim,
                    window_size=window_size)
    elif gnn == 'RGGNN':
        model = RGGNN(node_feature_dim=2, hidden_dim=hidden_dim,
                    window_size=window_size)

    # Load the trained model
    train_model_folder = f'models/{case_folder}'
    model.load_state_dict(torch.load(f'{train_model_folder}/model_{gnn}.pt'))

    # Predict future snapshots using the trained model
    time_solving_start = time.time()
    if window_size is None:
        df_c_pred, df_T_pred = predict_future_snapshots(model,
                                                        last_training_graph,
                                                        num_future_snapshots)
    else:
        df_c_pred, df_T_pred = predict_future_snapshots_window(model,
                                                               last_training_graph,
                                                               num_future_snapshots)
    time_solving = time.time()-time_solving_start
    print('time_solving: ', time_solving)

    # Append initial graph in case of including training
    df_c_pred.insert(loc=0, column='initial', value=c_0)
    df_T_pred.insert(loc=0, column='initial', value=T_0)

    # Create folder to store predictions
    pred_folder = f'reports/predictions/{case_folder}'
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    # Save predicted data
    df_c_pred.to_csv(f'{pred_folder}/c_{gnn}{run_num}.csv', index=False)
    df_T_pred.to_csv(f'{pred_folder}/T_{gnn}{run_num}.csv', index=False)
    return
