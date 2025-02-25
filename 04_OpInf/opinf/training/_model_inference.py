__all__ = [
    "train_model",
    "learned_model"
]
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data

import opinf.parameters
Params = opinf.parameters.Params()  # call parameters from dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
seed = 2  # Choose any seed value 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def train_model(states, derivatives, time, entries, rom,
                integration=False, plotting=False):
    """
    Trains a model using the operator inference method.

    Parameters:
    - states (np.ndarray): The reduced states.
    - derivatives (np.ndarray): The reduced derivatives.
    - time (np.ndarray): The training times.
    - rom (object): The reduced order model to be trained.

    Returns:
    - model (object): The trained model.
    - loss_track (list): The track of losses during training.
    """

    # Convert states, time, and derivatives to torch tensors
    states_torch = torch.tensor(states, device=device).T.double()
    time_torch = (torch.arange(0, derivatives.shape[-1], device=device)
                  * (time[1] - time[0])).reshape(-1, 1)
    derivatives_torch = torch.tensor(derivatives, device=device).T.double()
    entries_torch = torch.tensor(entries, device=device).double()

    # Create a dataset from the tensors
    train_dataset = torch.utils.data.TensorDataset(
        states_torch, time_torch, derivatives_torch, entries_torch
    )

    # Define dataloaders (put data into batches)
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=Params.batch_size, shuffle=False
    )
    dataloaders = {"train": train_dl}

    # Choose optimizer
    optimizer = torch.optim.NAdam(
        rom.parameters(),
        Params.adam_lr,
        Params.adam_betas,
        Params.adam_eps,
        Params.adam_weight_decay
    )

    # Learning rate according to cyclical learning rate policy
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        step_size_up=Params.lr_schedule_step_factor * len(dataloaders["train"]),
        step_size_down=Params.lr_schedule_step_factor * len(dataloaders["train"]),
        mode=Params.lr_schedule_mode,
        cycle_momentum=Params.lr_schedule_cycle_momentum,
        base_lr=Params.lr_schedule_base_lr,
        max_lr=Params.lr_schedule_max_lr,
    )
    torch.nn.utils.clip_grad_norm_(rom.parameters(), max_norm=1.0)
    # Train the model and store the results
    model, loss_track = _fit(
        rom,
        dataloaders,
        optimizer,
        scheduler=scheduler,
        integration=integration,
        plotting=plotting
    )

    return model, loss_track


def learned_model(model):
    """
    Extracts learned model parameters.

    Parameters:
    - model (object): Trained model.

    Returns:
    - A (np.ndarray): 'A' attribute of the model if applicable, else zeros.
    - B (np.ndarray): 'B' attribute of the model if applicable, else zeros.
    - C (np.ndarray): 'B' attribute of the model if applicable, else zeros.
    - H (np.ndarray): 'H' attribute of the model if applicable, else zeros.
    """

    if 'A' in Params.model_structure:
        A = model.module.A.detach().cpu().numpy()
    else:
        A = np.zeros((Params.ROM_order, Params.ROM_order))

    if 'B' in Params.model_structure:
        B = model.module.B.detach().cpu().numpy().reshape(-1, 1)
    else:
        B = np.zeros((Params.ROM_order, Params.input_dim))

    if 'C' in Params.model_structure:
        C = model.module.C.detach().cpu().numpy().reshape(-1,)
    else:
        C = np.zeros((Params.ROM_order))

    if 'H' in Params.model_structure:
        H = model.module.H.detach().cpu().numpy()
    else:
        H = np.zeros((Params.ROM_order, Params.ROM_order ** 2))
    return A, B, C, H


def _fit(model, dataloaders, optimizer, scheduler=None, integration=False,
         plotting=False):
    """
    Trains a model using the provided parameters.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - dataloaders (DataLoader): The data loaders for the training data.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler. Defaults to None.
    - integration (bool, optional): Whether to use integration. Defaults to False.
    - plotting (bool, optional): Whether to plot the loss track after training. Defaults to False.

    Returns:
    - model (torch.nn.Module): The trained model.
    - loss_track (list): The track of losses during training.
    """
    print("\n")
    print("# Optimizing the matrices")
    print("_" * 75)
    loss_track = []
    best_loss = float('inf')
    best_model_state_dict = model.state_dict()
    no_improvement_counter = 0
    early_stopping_threshold = 1000
    low_improvement_threshold = 1e-10
    criterion = torch.nn.MSELoss()

    for epoch in range(Params.num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch in dataloaders["train"]:
            optimizer.zero_grad()
            state, time, ddt, entries = batch
            y_pred = model(x=state, t=time, u=entries)
            y_pred = y_pred.to(device)
            if integration:
                timestep = torch.diff(time, axis=0)
                state_pred = _rk4th_onestep(model, state[:-1],
                                            timestep=timestep)
                loss = criterion(state_pred, state[1:]) / batch[0].shape[0]
            else:
                loss = criterion(y_pred, ddt) / batch[0].shape[0]

            loss_track.append(loss.item())

            if 'H' in Params.model_structure:
                loss += Params.regularization_H \
                    * model.module.H.abs().mean().to(device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / len(dataloaders["train"])

            if scheduler:
                scheduler.step()

        avg_loss = total_loss / len(dataloaders["train"])

        if (epoch + 1) % (Params.num_epochs/10) == 0:
            print(f"[Epoch {epoch + 1}/{Params.num_epochs}] [Avg. Loss: {avg_loss:.2e}][Learning Rate: {optimizer.param_groups[0]['lr']:.2e}]")

            if (best_loss - avg_loss) < low_improvement_threshold:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state_dict = model.state_dict()

            if no_improvement_counter >= early_stopping_threshold:
                print(f"Early stopping at epoch {epoch + 1} due to low improvement for {early_stopping_threshold} consecutive 100-epoch intervals.")
                break

    # Load the best model
    model.load_state_dict(best_model_state_dict)

    print("\nTraining completed.")
    print(f"Best loss is: {best_loss}")

    # Plot loss track if plotting is enabled
    if plotting:
        plt.figure(figsize=(12, 8))
        # Plotting the loss track with line style and color
        plt.plot(loss_track[5000:], color='royalblue', linestyle='-',
                 linewidth=5, label='Loss Track')
        # Set x and y axis labels
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Loss (log scale)', fontsize=14)
        # Set title and adjust font size
        plt.title('Training Loss Over Time', fontsize=16, fontweight='bold')
        # Set the y-axis to a logarithmic scale
        plt.yscale('log')
        # Add grid with specific style
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        # Add legend
        plt.legend(fontsize=12)
        # Show the plot
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.show()
    return model, loss_track



def _rk4th_onestep(model, x, t=0, timestep=1e-2):
    """
    Performs one step of the Runge-Kutta 4th order integration.

    Parameters:
    - model (torch.nn.Module): The model for which to integrate.
    - x (torch.Tensor): Current state.
    - t (torch.Tensor): Current time.
    - timestep (float): Integration timestep.

    Returns:
    - torch.Tensor: New state after one integration step.
    """
    k1 = model(x, t)
    k2 = model(x + 0.5 * timestep * k1, t + 0.5 * timestep)
    k3 = model(x + 0.5 * timestep * k2, t + 0.5 * timestep)
    k4 = model(x + 1.0 * timestep * k3, t + 1.0 * timestep)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep
