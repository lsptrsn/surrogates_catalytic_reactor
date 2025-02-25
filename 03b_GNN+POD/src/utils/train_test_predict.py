import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
import ipdb
import random
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

# Set seed for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
rs = RandomState(MT19937(SeedSequence(seed)))

def train_simple(dataset, model, lr, criterion, device, optimizer):
    """
    Train the GNN model.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - lr (float): Learning rate.
    - criterion: Loss function.
    - device: Device for training (CPU or GPU).
    - optimizer: Optimizer for updating model parameters.

    Returns:
    - float: Average training loss.
    """
    model.train()
    total_loss = 0
    for i in range(len(dataset) - 1):
        current_graph = dataset[i].to(device)
        next_graph = dataset[i + 1].to(device)

        optimizer.zero_grad()
        out = model(current_graph)
        loss = criterion(out, next_graph.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataset)

def test_simple(dataset, model, criterion, device):
    """
    Evaluate the GNN model on the test dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The test dataset.
    - model: The trained GNN model.
    - criterion: Loss function.
    - device: Device for evaluation (CPU or GPU).

    Returns:
    - float: Average test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(len(dataset) - 1):
            current_graph = dataset[i].to(device)
            next_graph = dataset[i + 1].to(device)

            out = model(current_graph)
            loss = criterion(out, next_graph.x)
            total_loss += loss.item()

    avg_loss = total_loss / (len(dataset) - 1)
    return avg_loss

def predict_future_snapshots(model, last_training_graph, num_future_snapshots):
    """
    Predict future snapshots based on the last training graph.

    Parameters:
    - model: The trained GNN model.
    - last_training_graph: The last graph from the training set (PyTorch Geometric Data object).
    - num_future_snapshots (int): The number of future time snapshots to predict.

    Returns:
    - df_conversion: DataFrame with predicted conversion values.
    - df_temperature: DataFrame with predicted temperature values.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    current_graph = last_training_graph.to(device)

    conversion_predictions = []
    temperature_predictions = []

    with torch.no_grad():
        for _ in tqdm(range(num_future_snapshots), desc="Predicting the future"):
            out_features = model(current_graph)
            conversion, temperature = out_features[:, 0], out_features[:, 1]

            conversion_predictions.append(conversion.cpu().numpy())
            temperature_predictions.append(temperature.cpu().numpy())

            current_graph.x = out_features

    df_conversion = pd.DataFrame(conversion_predictions).transpose()
    df_temperature = pd.DataFrame(temperature_predictions).transpose()

    return df_conversion, df_temperature

# Define smoothness regularization function
def smoothness_regularization(node_attr):
    derivatives = torch.gradient(node_attr, dim=0)[0]
    diff_of_derivatives = derivatives[1:] - derivatives[:-1]
    squared_diff = torch.sum(diff_of_derivatives ** 2, dim=0)
    return squared_diff

def train_multistep(dataset, model, lr, steps_ahead=10, noise_std=0.0001, smoothness_weight=0.001):
    """
    Train the GNN model for multi-step prediction.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - lr (float): Learning rate.
    - steps_ahead (int): Number of steps ahead for multi-step prediction.
    - noise_std (float): Standard deviation of Gaussian noise to inject into input features.
    - smoothness_weight (float): Weight of smoothness regularization in the loss function.

    Returns:
    - float: Average training loss.
    """
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    total_loss = 0
    num_snapshots = len(dataset)
    for i in tqdm(range(num_snapshots - 1)):
        optimizer.zero_grad()
        multi_step_loss = 0

        current_graph = dataset[i].to(device)
        x_copy = current_graph.x.clone()
        for step in range(1, min(steps_ahead, num_snapshots - i)):

            noise = torch.randn_like(x_copy) * noise_std
            noisy_x = x_copy + noise
            noisy_graph = Data(noisy_x, current_graph.edge_index)

            out = model(noisy_graph)
            target_graph = dataset[i + step].to(device)

            loss = criterion(out, target_graph.x)
            smoothness_loss = smoothness_regularization(noisy_x)
            loss += torch.sum(smoothness_weight * smoothness_loss)
            multi_step_loss += loss

            x_copy = out

        multi_step_loss /= min(steps_ahead, num_snapshots - i)

        multi_step_loss.backward()
        optimizer.step()

        total_loss += multi_step_loss.item()

    average_loss = total_loss / (len(dataset) - steps_ahead)
    return average_loss

def shift_time_snapshot(tensor, new_snapshot):
    """
    Shift time snapshot by updating the input tensor.

    Parameters:
    - tensor (torch.Tensor): Input tensor containing previous snapshots.
    - new_snapshot (torch.Tensor): New snapshot to insert into the tensor.

    Returns:
    None
    """
    num_nodes = tensor.size(0)
    z = new_snapshot.size(0)

    num_chunks = num_nodes // z

    for i in range(1, num_chunks):
        start = (i-1) * z
        end = i * z
        tensor[start:end] = tensor[start + z:end + z]

    tensor[(num_chunks-1) * z:] = new_snapshot

def train_multistep_window(dataset, model, lr, steps_ahead=10, noise_std=0.0001, smoothness_weight=0.001):
    """
    Train the GNN model for multi-step prediction with window-based dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - lr (float): Learning rate.
    - steps_ahead (int): Number of steps ahead for multi-step prediction.
    - noise_std (float): Standard deviation of Gaussian noise to inject into input features.
    - smoothness_weight (float): Weight of smoothness regularization in the loss function.

    Returns:
    - float: Average training loss.
    """
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    total_loss = 0
    num_snapshots = len(dataset)
    for i in tqdm(range(num_snapshots - 1)):
        optimizer.zero_grad()
        multi_step_loss = 0

        current_graph, _ = dataset[i]
        current_graph = current_graph.to(device)
        x_copy = current_graph.x.clone()
        for step in range(1, min(steps_ahead, num_snapshots - i)):

            noise = torch.randn_like(x_copy) * noise_std
            noisy_x = x_copy + noise
            noisy_graph = Data(noisy_x, current_graph.edge_index)

            out = model(noisy_graph)

            _, target_x = dataset[i + step - 1]
            target_x = target_x.to(device)

            loss = criterion(out, target_x)
            smoothness_loss = smoothness_regularization(noisy_x)
            loss += torch.sum(smoothness_weight * smoothness_loss)
            multi_step_loss += loss

            shift_time_snapshot(x_copy, out)

        multi_step_loss /= min(steps_ahead, num_snapshots - i)

        multi_step_loss.backward()
        optimizer.step()

        total_loss += multi_step_loss.item()

    average_loss = total_loss / (len(dataset) - steps_ahead)
    return average_loss

def train_window(dataset, model, criterion, device, optimizer):
    """
    Train the GNN model with window-based dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The training dataset.
    - model: The GNN model.
    - criterion: Loss function.
    - device: Device for training (CPU or GPU).
    - optimizer: Optimizer for updating model parameters.

    Returns:
    - float: Average training loss.
    """
    model.train()
    total_loss = 0
    for i in tqdm(range(len(dataset))):
        current_graph, target_x = dataset[i]
        current_graph = current_graph.to(device)
        target_x = target_x.to(device)
        optimizer.zero_grad()
        out = model(current_graph)
        loss = criterion(out, target_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataset)

def test_window(dataset, model, criterion, device):
    """
    Evaluate the GNN model on the test dataset with window-based dataset.

    Parameters:
    - dataset (torch_geometric.data.Dataset): The test dataset.
    - model: The trained GNN model.
    - criterion: Loss function.
    - device: Device for evaluation (CPU or GPU).

    Returns:
    - float: Average test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            current_graph, target_x = dataset[i]
            current_graph = current_graph.to(device)
            target_x = target_x.to(device)

            out = model(current_graph)
            loss = criterion(out, target_x)
            total_loss += loss.item()

    avg_loss = total_loss / (len(dataset))
    return avg_loss

def predict_future_snapshots_window(model, last_training_graph, num_future_snapshots):
    """
    Predict future snapshots based on the last training graph with window-based dataset.

    Parameters:
    - model: The trained GNN model.
    - last_training_graph: The last graph from the training set (PyTorch Geometric Data object).
    - num_future_snapshots (int): The number of future time snapshots to predict.

    Returns:
    - df_conversion: DataFrame with predicted conversion values.
    - df_temperature: DataFrame with predicted temperature values.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    current_graph = last_training_graph.to(device)

    conversion_predictions = []
    temperature_predictions = []

    with torch.no_grad():
        for _ in tqdm(range(num_future_snapshots), desc="Predicting the future"):
            out_features = model(current_graph)
            conversion, temperature = out_features[:, 0], out_features[:, 1]

            conversion_predictions.append(conversion.cpu().numpy())
            temperature_predictions.append(temperature.cpu().numpy())

            shift_time_snapshot(current_graph.x, out_features)

    df_conversion = pd.DataFrame(conversion_predictions).transpose()
    df_temperature = pd.DataFrame(temperature_predictions).transpose()

    return df_conversion, df_temperature
