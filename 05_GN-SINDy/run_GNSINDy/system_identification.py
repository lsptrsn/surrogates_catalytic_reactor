#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:58:51 2024

@author: forootani (adapted by peterson)
"""

import matplotlib.pylab as plt
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import os
import random
import shutil
import sys
import time
import torch

# Set up paths
up1 = os.path.abspath('..')
sys.path.insert(0, up1)

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTestPeriodic
from deepymod.data.DEIM_class import DEIM


case = '1'
seed = 0
if torch.cuda.is_available():
    device = "cuda"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.set_per_process_memory_fraction(0.8, device=torch.device('cuda:0'))
    # torch.cuda.set_per_process_memory_growth(torch.device('cuda:0'), True)
    torch.cuda.manual_seed_all(seed)
else:
    device = "cpu"
# Settings for reproducibility
torch.cuda.empty_cache()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
rs = RandomState(MT19937(SeedSequence(seed)))


def create_data_DEIM(get_scaling=False):
    """
    Create preprocessed data for analysis.

    Args:
    DEIM_tolerance (float): Tolerance for creating the samples via DEIM.
    Default is 1e-7.
    get_scaling (bool): Wether to output the temperature scaling or not.
    Default is False.
    data_folder (str): Path to the folder containing the data files.
    Default is "/data/IEEE_data/".

    Returns:
    tuple: Tuple containing coordinates and processed data.
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
    T_min = np.min(data_temp_list)
    T_max = np.max(data_temp_list)
    normalized_data_temp = (data_temp_list - T_min) \
        / (T_max - T_min)
    data_temp = np.array(normalized_data_temp)

    # Stack data arrays
    coords = torch.from_numpy(coords_main_unique).float()
    data = torch.from_numpy(np.stack((data_conv, data_temp), axis=-1)).float()
    if get_scaling is True:
        return coords, data, T_min, T_max
    else:
        return coords, data



def load_data():
    # Load original data
    data_path = "../data/IEEE_data/case_"+case+"/"
    if case == "1":
        case_name = '_flow_rate_down_interp_state.npy'
    elif case == "2":
        case_name = '_flow_rate_up_interp_state.npy'
    elif case == "3":
        case_name = '_temp_cool_down_interp_state.npy'
    elif case == "4":
        case_name = '_temp_cool_up_interp_state.npy'
    data_conv = np.load(data_path+"conversion"+case_name)[::x, ::x]
    data_temp = np.load(data_path+"temperature"+case_name)[::x, ::x]
    t_o = np.load(data_path+"time"+case_name).reshape(-1, 1)[::x]
    x_o = np.load(data_path+"z"+case_name).reshape(-1, 1)[::x]

    # Normalize time vector
    t_o = t_o / np.max(t_o)
    # Remove last element from x_o
    x_o = x_o[:-1, :]
    # make grid
    t_o_tile = np.tile(t_o, np.shape(x_o)[0]).T
    x_o_tile = np.tile(x_o, np.shape(t_o)[0])
    # get coord and data
    coords = torch.from_numpy(np.stack((t_o_tile, x_o_tile), axis=-1)).float()
    data = torch.from_numpy(np.stack((data_conv[:-1, :], data_temp[:-1, :]), axis=-1)).float()
    return coords, data


def create_or_reset_directory(directory_path):
    """
    Create or reset a directory.

    Args:
    directory_path (str): Path to the directory.

    Returns:
    None
    """
    # Check if the directory exists
    if os.path.exists(directory_path):
        # If it exists, remove it
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' already exists so it is removed.")
        except OSError as e:
            print(f"Error removing directory '{directory_path}': {e}")
            return

    # Create the directory
    try:
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")


def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    """Compute the absolute and relative errors between Qtrue and Qapprox,
    where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||,
        relative_error = ||Qtrue - Qapprox|| / ||Qtrue||
                       = absolute_error / ||Qtrue||,

    with ||Q|| defined by norm(Q).
    """
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data


from scipy import linalg as la
def frobenius_error(Qtrue, Qapprox):
    """Compute the absolute and relative Frobenius-norm errors between the
    snapshot sets Qtrue and Qapprox, where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||_F,
        relative_error = ||Qtrue - Qapprox||_F / ||Qtrue||_F.

    Parameters
    ----------
    Qtrue : (n, k)
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j].
    Qapprox : (n, k)
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j].

    Returns
    -------
    abs_err : float
        Absolute error ||Qtrue - Qapprox||_F.
    rel_err : float
        Relative error ||Qtrue - Qapprox||_F / ||Qtrue||_F.
    """
    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim != 2:
        raise ValueError("Qtrue and Qapprox must be two-dimensional")

    # Compute the errors.
    return _absolute_and_relative_error(Qtrue, Qapprox,
                                        lambda Z: la.norm(Z, ord="fro"))

###############################################################################
# Get data
###############################################################################
# Create dataset
x = 1  # how many values to keep from the original dataset
split_ratio = 5/6  # split data into train an testset
DEIM_tolerance = 1e-7  # tolerance for Q-DEIM
dataset = Dataset(
    create_data_DEIM,
    apply_normalize=None,
    apply_noise=None,
    apply_shuffle=None,
    shuffle=True,
    preprocess_kwargs={
        "random_state": 0,
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 10000000},
    device=device,
)

fig, ax = plt.subplots()
ax.scatter(dataset.get_coords()[:, 1].cpu(), dataset.get_data()[:, 0].cpu(),
           label="X", s=3)
ax.scatter(dataset.get_coords()[:, 1].cpu(), dataset.get_data()[:, 1].cpu(),
           label="T", s=3)
ax.set_xlabel("spatial coordinate")
ax.legend()
ax.set_title("Dataset")
plt.show()

# get numpy variables for checking variables
coords_unsampled, data_unsampled, T_min, T_max = create_data_DEIM(
    get_scaling=True)
coords_unsampled = np.array(coords_unsampled.detach().cpu())
data_unsampled = np.array(data_unsampled.detach().cpu())
coords_sampled = np.array(dataset.get_coords().detach().cpu())
data_sampled = np.array(dataset.get_data().detach().cpu())

# Split dataset into train and test
train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split=1)
test_dataloader = train_dataloader

###############################################################################
# Train network
###############################################################################
# orders of polynomials and derivatives in library
poly_order = 2
deriv_order = 1
library = Library1D(poly_order, deriv_order)

# Neural netowrk architecture
network = NN(2, [64, 64, 64, 64], 2)
lam = 0.0005
tol = 1
estimator = STRidge(lam=lam,  # regularization parameter for ridge regression. Higher values promote smaller coefficients and more sparsity.
                    tol=tol)  # Threshold for setting small coefficients to zero. Higher values will zero out more coefficients.
# estimator = Threshold(threshold=1)  # Value of the threshold above which the terms are selected
sparsity_scheduler = TrainTestPeriodic(periodicity=50,  # apply sparsity mask per periodicity epochs
                                       patience=200,  # wait patience epochs before checking TrainTest
                                       delta=1e-4) # desired accuracy

constraint = LeastSquares()
model = DeepMoD(network, library, estimator, constraint).to(device)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99),
                             amsgrad=True, lr=1e-3)

# Training configuration
foldername = "./results/SINDy_pde"+str(seed)+"/"
create_or_reset_directory(foldername)

# Train model
time_train_start = time.time()
# Train model
train(
    model,
    train_dataloader,  # training data as dataloader
    test_dataloader,  # testing data as dataloader
    optimizer,  # Pytorch optimizer
    sparsity_scheduler,  # Decides when to update the sparsity mask.
    log_dir=foldername,  # Directory where tensorboard file is written
    exp_ID="Test",  # Unique ID to identify tensorboard file
    write_iterations=25,  # Sets how often data is written to tensorboard and checks train loss
    max_iterations=25000,  # Max number of epochs
    delta=1e-3,  # convergence_kwargs, desired accuracy
    patience=200,  # convergence_kwargs, how often to check for convergence
)
time_train = time.time()-time_train_start
###############################################################################
# Results
###############################################################################

# Print model sparsity mask and model coefficients
print('Model sparsity mask for Conversion')
print(model.sparsity_masks[0])
print('Number of entries', model.sparsity_masks[0].sum().item())
print('Model coefficients for Conversion')
print(model.constraint.coeff_vectors[0])
print('Model sparsity mask for Temperature')
print(model.sparsity_masks[1])
print('Number of entries', model.sparsity_masks[1].sum().item())
print('Model coefficients for Temperature')
print(model.constraint.coeff_vectors[1])
if deriv_order == 1:
    library = ["1", "T", "T**2", "X", "XT", "XT**2", "X**2", "X**2T", "X**2T**2",
               "T_z", "X_z", "T_zX_z",
               "X*X_z", "X**2*X_z", "T*T_z", "T**2*T_z",
               "X*T_z", "X**2*T_z", "T*X_z", "T**2*X_z"]
    ders = ["X_t", "T_t"]
    for sparse, coeff_vector, der in zip(
        model.sparsity_masks, model.constraint_coeffs(), ders
    ):
        expression = ""
        coeffs = [
            "%.5f" % number for number in (coeff_vector.detach().cpu().numpy().squeeze())
        ]
        monomials = [str(a) + "*" + str(b) for a, b in zip(coeffs, library)]
        sparse_array = sparse.detach().cpu().numpy()
        print(der, "=", np.extract(sparse_array, monomials))

###############################################################################
# Results
###############################################################################

dataset_all = Dataset(
    load_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    shuffle = False,
    subsampler=False,
    subsampler_kwargs={"number_of_samples": 10000000},
    device=device,
)

data_path = "../data/IEEE_data/case_"+case+"/"
if case == "1":
    case_name = '_flow_rate_down_interp_state.npy'
elif case == "2":
    case_name = '_flow_rate_up_interp_state.npy'
elif case == "3":
    case_name = '_temp_cool_down_interp_state.npy'
elif case == "4":
    case_name = '_temp_cool_up_interp_state.npy'
z =  np.load(data_path+"z"+case_name).reshape(-1, 1)[::x]
t = np.load(data_path+"time"+case_name).reshape(-1, 1)[::x]

# Set up mpi color
mpi_colors = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255)
}
prediction = True
if prediction == True:
    # get data from model
    time_pred_start = time.time()
    # predict
    data_pred = model(dataset_all.coords)
    time_pred = time.time()-time_pred_start
    print('Training time is: ', time_train, 's')
    print('Prediction time is: ', time_pred, ' s')


    X_pred = data_pred[0][:, 0]
    X_pred = X_pred.detach().cpu().numpy()
    T_pred_unscaled = data_pred[0][:,1]
    T_pred_unscaled = T_pred_unscaled.detach().cpu().numpy()
    T_pred = T_pred_unscaled*(T_max-T_min)+T_min

    # get original data
    X_true =  dataset_all.data[:, 0].cpu().numpy()
    T_true = dataset_all.data[:, 1].cpu().numpy()

    # reshape
    desired_size = int(np.sqrt(np.shape(X_true)[0]))
    X_pred = X_pred.reshape((desired_size, desired_size))
    T_pred = T_pred.reshape((desired_size, desired_size))
    X_true = X_true.reshape((desired_size, desired_size))
    T_true = T_true.reshape((desired_size, desired_size))
    t_train_end = int(np.shape(X_pred)[1]*split_ratio)
    X_pred_train = X_pred[:, 0:t_train_end]
    T_pred_train = T_pred[:, 0:t_train_end]
    X_true_train = X_true[:, 0:t_train_end]
    T_true_train = T_true[:, 0:t_train_end]
    t_train = t[0:t_train_end]
    X_pred_test = X_pred[:, t_train_end:]
    T_pred_test = T_pred[:, t_train_end:]
    X_true_test = X_true[:, t_train_end:]
    T_true_test = T_true[:, t_train_end:]
    t_test = t[t_train_end:]

    # store results
    parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    folder_name = 'eval/case_'+case+"/" + str(seed)
    results_folder = os.path.join(parent_folder, folder_name)
    # Ensure the directory exists
    os.makedirs(results_folder, exist_ok=True)

    # Dictionary of arrays and their file names
    arrays_to_save = {
        'conversion_train_SINDy.npy': X_pred_train,
        'temperature_train_SINDy.npy': T_pred_train,
        'conversion_train_true.npy': X_true_train,
        'temperature_train_true.npy': T_true_train,
        'conversion_test_SINDy.npy': X_pred_test,
        'temperature_test_SINDy.npy': T_pred_test,
        'conversion_test_true.npy': X_true_test,
        'temperature_test_true.npy': T_true_test,
        't_train.npy': t_train,
        't_test.npy': t_test,
        'z.npy': z,
        'training_time.npy': time_train,
        'predicting_time.npy': time_pred,
    }

    # Save arrays as .npy files
    for filename, array in arrays_to_save.items():
        np.save(os.path.join(results_folder, filename), array)

    # Get Frobenius norm
    Q_true_train = np.vstack((X_true_train, T_true_train))
    Q_pred_train = np.vstack((X_pred_train, T_pred_train))
    abs_froerr, rel_froerr = frobenius_error(Qtrue=Q_true_train,
                                             Qapprox=Q_pred_train)
    print(f"Relative Frobenius-norm error train data: {rel_froerr:%}")
    Q_true_test = np.vstack((X_true_test, T_true_test))
    Q_pred_test = np.vstack((X_pred_test, T_pred_test))
    abs_froerr, rel_froerr = frobenius_error(Qtrue=Q_true_test,
                                             Qapprox=Q_pred_test)
    print(f"Relative Frobenius-norm error test data: {rel_froerr:%}")
