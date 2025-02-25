import torch
import time
import os
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import pandas as pd
import random
from torch.optim.lr_scheduler import StepLR
from src.utils.gnn_architectures import GCN, GAT, GGNN, RGGNN
from src.utils.train_test_predict import train_simple, test_simple, train_multistep, train_window, test_window

def train(case, hidden_dim, gnn, train_typ, num_epochs, lr, window_size,
          steps_ahead, noise_std, smoothness_weight, seed=0):
    """
    Train the specified graph neural network (GNN) model.

    Parameters:
    - case (int): The case number.
    - hidden_dim (int): The dimension of the hidden layers.
    - gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    - train_typ (str): The type of training ('simple', 'multistep', 'window').
    - num_epochs (int): The number of training epochs.
    - lr (float): The learning rate.
    - window_size (int): The size of the window for window-based training.
    - steps_ahead (int): Number of steps ahead for multistep training.
    - noise_std (float): Standard deviation of noise for multistep training.
    - smoothness_weight (float): Weight for smoothness term in multistep training.
    - seed (int, optional): Seed for reproducibility. Defaults to 0.

    Returns:
    None
    """
    # Set seed for reproducibility if provided
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))

    case_folder = 'case_' + str(case)

    train_dataset = torch.load(f'data/processed/{case_folder}/train_{gnn}.pt')
    test_dataset = torch.load(f'data/processed/{case_folder}/test_{gnn}.pt')

    # Initialize model based on specified GNN architecture
    if gnn == 'GCN':
        model = GCN(node_feature_dim=2, hidden_dim=hidden_dim, window_size=window_size)
    elif gnn == 'GAT':
        model = GAT(node_feature_dim=2, hidden_dim=hidden_dim, window_size=window_size)
    elif gnn == 'GGNN':
        model = GGNN(node_feature_dim=2, hidden_dim=hidden_dim, window_size=window_size)
    elif gnn == 'RGGNN':
        model = RGGNN(node_feature_dim=2, hidden_dim=hidden_dim, window_size=window_size)

    # Print model parameters
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    train_traj = np.zeros(num_epochs)
    test_traj = np.zeros(num_epochs)

    # Define the learning rate scheduler, criterion, and optimizer
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    time_training_start = time.time()
    for epoch in range(num_epochs):
        print('-'*110)

        if train_typ == 'simple':
            train_mse = train_simple(train_dataset, model, lr,
                              criterion, device, optimizer)
            test_mse = test_simple(test_dataset, model, criterion, device)
        elif train_typ == 'multistep':
            train_mse = train_multistep(train_dataset,
                                        model, lr,
                                        steps_ahead=steps_ahead,
                                        noise_std=noise_std,
                                        smoothness_weight=smoothness_weight)
            test_mse = test_simple(test_dataset, model, criterion, device)
        elif train_typ == 'window':
            train_mse = train_window(train_dataset, model, criterion,
                                     device, optimizer)
            test_mse = test_window(test_dataset, model, criterion, device)

        # save performance
        train_traj[epoch] = train_mse
        test_traj[epoch] = test_mse

        print(f'Epoch {epoch + 1} completed   Train_MSE: {train_mse} / Test_MSE: {test_mse}')
    time_training = time.time()-time_training_start
    print('Total training time: ', time_training)
    # Save model
    train_model_folder = f'models/{case_folder}'
    if not os.path.exists(train_model_folder):
        os.makedirs(train_model_folder)
    torch.save(model.state_dict(), f'{train_model_folder}/model_{gnn}.pt')

    df_training_traj = pd.DataFrame({
        'Train_rmse': np.sqrt(train_traj),
        'Test_rmse': np.sqrt(test_traj)
    })

    # Save training trajectory
    train_traj_folder = f'reports/training_traj/{case_folder}'
    if not os.path.exists(train_traj_folder):
        os.makedirs(train_traj_folder)

    df_training_traj.to_csv(f'{train_traj_folder}/train_traj_{gnn}.csv', index=False)
    return
