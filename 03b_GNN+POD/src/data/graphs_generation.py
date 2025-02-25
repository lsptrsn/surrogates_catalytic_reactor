# Train and test split
# author: Edgar Sanchez

import pandas as pd
import numpy as np
import random
from numpy.random import RandomState, SeedSequence, MT19937
import os
import torch
from src.utils.datasets import ReactorDataset_simple, ReactorDataset_window
import ipdb

def graphs_generation(case, gnn, window_size, validation_mode, POD_run=False, POD_thresh=150, seed=0):
    """
    Generate graphs for spatial segments.

    Parameters:
    - case (int): The case number.
    - gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').
    - window_size (int): Size of the window for window-based dataset.
    - validation_mode (Boolean): True for training-validation run, False for training-testing run.
    - POD_run (Boolean, optional): True for POD+GNN run, False for pure GNN.
    - POD_thresh (int, optional): Number of reduced features. Defaults to 150.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """
    # Set seeds for reproducibility if provided
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))

    base_dir = os.getcwd()
    case_folder = 'case_' + str(case)
    data_folder = os.path.join(base_dir, 'data', 'interim', case_folder)
    dataset_folder = os.path.join(base_dir, 'data', 'processed', case_folder)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Read spatial coordinates
    df_z = pd.read_csv(f'{data_folder}/z.csv', header=None)

    # Read conversion and temperature data
    split = 'train'
    df_c = pd.read_csv(f'{data_folder}/{split}_conversion.csv',
                       header=None)
    df_T = pd.read_csv(f'{data_folder}/{split}_temperature.csv',
                       header=None)

    # Create reactor dataset
    if window_size == 1:
        reactor_dataset = ReactorDataset_simple(df_z=df_z,
                                                df_c=df_c,
                                                df_T=df_T)
    else:
        reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                df_c=df_c,
                                                df_T=df_T,
                                                window_size=window_size,
                                                POD_run=POD_run)

    # Save reactor dataset
    torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}.pt')

    # Quick check of correct construction of datasets
    train_dataset = torch.load(f'data/processed/{case_folder}/train_{gnn}.pt')


    if validation_mode:
        try:
            # Read test conversion and temperature data if available
            split = 'validation'
            df_c = pd.read_csv(f'{data_folder}/{split}_conversion.csv',
                            header=None)
            df_T = pd.read_csv(f'{data_folder}/{split}_temperature.csv',
                            header=None)

            # Create test reactor dataset
            if window_size == 1:
                reactor_dataset = ReactorDataset_simple(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T)
            else:
                reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T,
                                                        window_size=window_size)

            # Save test reactor dataset
            torch.save(reactor_dataset, f'{dataset_folder}/test_{gnn}.pt')
            test_dataset = torch.load(f'data/processed/{case_folder}/test_{gnn}.pt')
        except:
            print('No test data. Using the training dataset instead.')
            test_dataset = train_dataset

        print('-'*110)
        print(f'Graphs in training dataset: {len(train_dataset)}')
        print(f'Graphs in validation dataset: {len(test_dataset)}')
        print('-'*110)

    else:
        try:
            # Read test conversion and temperature data if available
            split = 'test'
            df_c = pd.read_csv(f'{data_folder}/{split}_conversion.csv',
                            header=None)
            df_T = pd.read_csv(f'{data_folder}/{split}_temperature.csv',
                            header=None)

            # Create test reactor dataset
            if window_size == 1:
                reactor_dataset = ReactorDataset_simple(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T)
            else:
                reactor_dataset = ReactorDataset_window(df_z=df_z,
                                                        df_c=df_c,
                                                        df_T=df_T,
                                                        window_size=window_size)

            # Save test reactor dataset
            torch.save(reactor_dataset, f'{dataset_folder}/{split}_{gnn}.pt')
            test_dataset = torch.load(f'data/processed/{case_folder}/test_{gnn}.pt')
        except:
            print('No test data. Using the training dataset instead.')
            test_dataset = train_dataset

        print('-'*110)
        print(f'Graphs in training dataset: {len(train_dataset)}')
        print(f'Graphs in test dataset: {len(test_dataset)}')
        print('-'*110)
    return
