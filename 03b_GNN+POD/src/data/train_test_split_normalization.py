# Train and test split
# author: Edgar Sanchez

import numpy as np
import os
from tqdm import tqdm
import scipy as sp
import pickle as pk
from scipy.linalg import svd as SVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ipdb
# Set seed for reproducibility
seed = 1
np.random.seed(seed)

def train_test_split_normalization(case, train_time, validation_mode, n_points_per_second, validation_time=0):
    """
    Split the data into train and test sets and perform normalization.

    Parameters:
    - case (int): The case number.
    - train_time (int): Duration of training data in seconds.
    - validation_mode (Boolean): True for training-validation run, False for training-testing run.
    - n_points_per_second (int): Number of data points per second.
    - validation_time (int, optional): Duration of validation data in seconds. Defaults to 0.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """

    # Define file names and case dictionary
    file_names = [
        'conversion',
        # 'conversion_corrected'
        'temperature',
        'time',
        'z'
    ]
    cases_dict = {
        1: 'flow_rate_down_interp_state',
        2: 'flow_rate_up_interp_state',
        3: 'temp_cool_down_interp_state',
        4: 'temp_cool_up_interp_state'
    }

    # Construct folder and case name based on the provided case number
    case_folder = 'case_' + str(case)
    case_name = cases_dict[case]
    base_dir = os.getcwd()

    # Create folder to store split data if it doesn't exist
    split_data_folder = os.path.join(base_dir, 'data', 'interim', case_folder)
    if not os.path.exists(split_data_folder):
        os.makedirs(split_data_folder)

    # Define train/test split parameters
    train_last_idx = train_time * n_points_per_second

    # Loop through each file
    for name in tqdm(file_names):
        file_path = os.path.join(base_dir, 'data', 'raw',
                                 case_folder, f'{name}_{case_name}.npy')
        data = np.load(file_path)

        # Split data into train and test sets
        if name == 'z':
            np.savetxt(f'{split_data_folder}/{name}.csv',
                       data, delimiter=',', fmt='%f')
        else:
            # Normalize Temperature and Conversion
            if name in ['temperature']:
                T_min = 400  # expected lowest temperature
                T_max = 800  # high risk temperature
                data = (data - T_min) / (T_max - T_min)
                # T_mu = 0.4533159615958609
                # T_sigma = 0.13944152723688819
                # data = (data - T_mu)/T_sigma

            # if name in ['conversion']:
            #     X_mu = 0.8487244223653506
            #     X_sigma = 0.2238620396750383
            #     data = (data - X_mu)/X_sigma

            if validation_mode:

                validation_last_idx = (train_time + validation_time) * n_points_per_second
                try:
                    train_data = data[:, :train_last_idx]
                    validation_data = data[:, train_last_idx:validation_last_idx]

                except:
                    train_data = data[:train_last_idx]
                    validation_data = data[train_last_idx:validation_last_idx]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/validation_{name}.csv',
                        validation_data, delimiter=',', fmt='%f')

            elif not validation_mode:
                try:
                    train_data = data[:, :train_last_idx]
                    test_data = data[:, train_last_idx:]

                except:
                    train_data = data[:train_last_idx]
                    test_data = data[train_last_idx:]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/test_{name}.csv',
                        test_data, delimiter=',', fmt='%f')
    return

def POD_norm_split_reduce(case, train_time, validation_mode, n_points_per_second, POD_type="SVD", POD_thresh=150, validation_time=0):
    """
    All pre-processing for POD+GNN, includes normalization, reducing using SVD or PCA, split of the data into train and test sets and stacking into a single matrix.

    Parameters:
    - case (int): The case number.
    - train_time (int): Duration of training data in seconds.
    - validation_mode (Boolean): True for training-validation run, False for training-testing run.
    - n_points_per_second (int): Number of data points per second.
    - POD_type (str, optional): Type of POD ('SVD', 'PCA'). Defaults to 'SVD'.
    - POD_thresh (int, optional): Number of reduced features. Defaults to 150.
    - validation_time (int, optional): Duration of validation data in seconds. Defaults to 0.
    - seed (int, optional): Seed for reproducibility. Defaults to 1.

    Returns:
    None
    """

    # Define file names and case dictionary
    file_names = [
        'conversion',
        # 'conversion_corrected'
        'temperature',
        'time',
        'z'
    ]
    cases_dict = {
        1: 'flow_rate_down_interp_state',
        2: 'flow_rate_up_interp_state',
        3: 'temp_cool_down_interp_state',
        4: 'temp_cool_up_interp_state'
    }

    # Construct folder and case name based on the provided case number
    case_folder = 'case_' + str(case)
    case_name = cases_dict[case]
    base_dir = os.getcwd()

    # Create folder to store split data if it doesn't exist
    split_data_folder = os.path.join(base_dir, 'data', 'interim', case_folder)
    if not os.path.exists(split_data_folder):
        os.makedirs(split_data_folder)

    # Create folder to store POD files if it doesn't exist
    POD_folder = os.path.join(base_dir, 'POD', case_folder)
    if not os.path.exists(POD_folder):
        os.makedirs(POD_folder)

    # POD function
    def POD_internal(data, POD_folder, name, split, POD_type, POD_thresh):

        if POD_type == "SVD":

            U, s, V = SVD(data)

            np.savetxt(f'{POD_folder}/U_{split}_{name}.csv',
                    U, delimiter=',', fmt='%f')
            # np.savetxt(f'{POD_folder}/s_{split}_{name}.csv',
            #         s, delimiter=',', fmt='%f')
            # np.savetxt(f'{POD_folder}/V_{split}_{name}.csv',
            #         V, delimiter=',', fmt='%f')

            data_PC = np.transpose(U[:, :POD_thresh]).dot(data)
            #np.transpose(np.transpose(U[:, :POD_thresh]).dot(np.transpose(data)))

        elif POD_type == "PCA":
            pca = PCA(n_components=POD_thresh).fit(np.transpose(data))
            data_PC = np.transpose(pca.transform(np.transpose(data)))
            pk.dump(pca, open(f'{POD_folder}/pca_{split}_{name}.pkl',"wb"))

        elif POD_type == "manual":
            data_PC = data[0:-1:2,:]


        return data_PC

    # Loop through each file
    for name in tqdm(file_names):
        file_path = os.path.join(base_dir, 'data', 'raw',
                                 case_folder, f'{name}_{case_name}.npy')
        data = np.load(file_path)

        if name == 'z':
            np.savetxt(f'{split_data_folder}/{name}.csv',
                       data, delimiter=',', fmt='%f')

        # Normalize Temperature and Conversion
        if name in ['temperature']:
            T_min = 400  # expected lowest temperature
            T_max = 800  # high risk temperature
            data = (data - T_min) / (T_max - T_min)
            T_mu = 0.4533159615958609
            T_sigma = 0.13944152723688819
            data = (data - T_mu)/T_sigma

        if name in ['conversion']:
            X_mu = 0.8487244223653506
            X_sigma = 0.2238620396750383
            data = (data - X_mu)/X_sigma

        #data = StandardScaler().fit_transform(data)

        # Define train/test split parameters
        train_last_idx = train_time * n_points_per_second

        # split and POD only for Conversion and Temperature
        if name in ['conversion', 'temperature']:


            # Split data into train and test sets
            if validation_mode:
                validation_last_idx = (train_time + validation_time) * n_points_per_second

                train_data = data[:, :train_last_idx]
                train_data = POD_internal(train_data, POD_folder, name, "train", POD_type, POD_thresh)

                validation_data = data[:, train_last_idx:validation_last_idx]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/validation_{name}.csv',
                        validation_data, delimiter=',', fmt='%f')

            else:
                train_data = data[:, :train_last_idx]
                train_data = POD_internal(train_data, POD_folder, name, "train", POD_type, POD_thresh)

                test_data = data[:, train_last_idx:]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/test_{name}.csv',
                        test_data, delimiter=',', fmt='%f')

        else:
            if validation_mode:
                validation_last_idx = (train_time + validation_time) * n_points_per_second
                try:
                    train_data = data[:, :train_last_idx]
                    validation_data = data[:, train_last_idx:validation_last_idx]

                except:
                    train_data = data[:train_last_idx]
                    validation_data = data[train_last_idx:validation_last_idx]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/validation_{name}.csv',
                        validation_data, delimiter=',', fmt='%f')

            else:
                try:
                    train_data = data[:, :train_last_idx]
                    test_data = data[:, train_last_idx:]

                except:
                    train_data = data[:train_last_idx]
                    test_data = data[train_last_idx:]

                # Save train and test data
                np.savetxt(f'{split_data_folder}/train_{name}.csv',
                        train_data, delimiter=',', fmt='%f')
                np.savetxt(f'{split_data_folder}/test_{name}.csv',
                        test_data, delimiter=',', fmt='%f')

    return
