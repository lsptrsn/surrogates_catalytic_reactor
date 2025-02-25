#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:02:24 2024

@author: peterson
"""
# Importing necessary functions and modules
import time
from src.data.x_exploration_data import explore_data
from src.data.train_test_split_normalization import train_test_split_normalization
from src.data.graphs_generation import graphs_generation
from src.models.train import train
from src.models.predict import predict
from src.visualization.plot_train_traj import plot_train_traj
from src.visualization.plot_future_prediction import plot_future_prediction
from src.utils.IEEE_eval import run_IEEE_eval


if __name__ == "__main__":
    # Setting up parameters for the experiment
    case = 1 # Experiment case number
    gnn = 'GAT'  # Type of Graph Neural Network: GCN, GAT, GGNN, RGGNN
    train_typ = 'window'  # Type of training: simple, multistep, window
    window_size = 20  # Size of the sliding window for data processing
    hidden_dim = 252  # Dimensionality of hidden layers in the neural network
    n_points_per_second = 25*2  # Number of data points per second
    total_time = 60  # Duration of total data in seconds
    validation_mode = False  # True for training-validation run, False for training-testing run
    if validation_mode:
        train_time = 90  # Duration of training data in seconds
        validation_time = 10  # Duration of validation data in seconds
        num_future_snapshots = validation_time * n_points_per_second  # Number of future snapshots to predict
    else:
        train_time = 50  # Duration of training data in seconds
        test_time =  total_time - train_time # Duration of testing data in seconds
        num_future_snapshots = test_time * n_points_per_second  # Number of future snapshots to predict
    lr = 5.168398752277867e-05  # Learning rate
    num_epochs = 15  # Number of training epochs
    steps_ahead = 5  # Number of steps ahead for multistep training
    noise_std = 0.001  # Standard deviation of noise for multistep training
    smoothness_weight = 5  # Weight for smoothnessT term in multistep training

    # Explore the dataset
    explore_data(case)

    # Split the dataset into train and test sets, and normalize it
    if validation_mode:
        train_test_split_normalization(case,
                                        train_time=train_time,
                                        validation_mode=validation_mode,
                                        n_points_per_second=n_points_per_second,
                                        validation_time=validation_time)
    else:
        train_test_split_normalization(case,
                                        train_time=train_time,
                                        validation_mode=validation_mode,
                                        n_points_per_second=n_points_per_second)

    # Generate graphs for spatial segments
    graphs_generation(case,
                      gnn,
                      window_size,
                      validation_mode=validation_mode)

    # Train Graph Neural Networks (GNNs)
    train(case,
          hidden_dim,
          gnn=gnn,
          train_typ=train_typ,
          num_epochs=num_epochs,
          lr=lr,
          window_size=window_size,
          steps_ahead=steps_ahead,
          noise_std=noise_std,
          smoothness_weight=smoothness_weight,
          seed=0
          )

    # Use trained GNNs to predict future snapshots
    predict(case,
            hidden_dim,
            gnn,
            window_size,
            num_future_snapshots)

    # Evaluate the results
    plot_train_traj(case, gnn)  # Plot the training trajectories
    plot_future_prediction(case, gnn)  # Plot the future predictions
    run_IEEE_eval(case, gnn, validation_mode=validation_mode)  # Run evaluation based on IEEE standards
