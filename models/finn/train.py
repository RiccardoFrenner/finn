#! env/bin/python3

"""
Main file for training a model with FINN
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import time
from threading import Thread
import sys

sys.path.append("..")
from utils.configuration import Configuration
import utils.helper_functions as helpers
from finn import *


def run_training(print_progress=True, model_number=None):

    # Load the user configurations
    config = Configuration("config.json")
    
    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":  # TODO: Bug because uppercase?
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    root_path = os.path.abspath("../../data")
    data_path = os.path.join(root_path, config.data.type, config.data.name)
    
    # Set device on GPU if specified in the configuration file, else CPU
    # device = helpers.determine_device()
    device = th.device(config.general.device)
    print("="*100)
    print("Using device:", config.general.device, device)
    print("="*100)
    

    # Load samples, together with x, y, and t series
    t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                    dtype=th.float).to(device=device)
    x = np.load(os.path.join(data_path, "x_series.npy"))
    sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                            dtype=th.float).to(device=device)
    sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                            dtype=th.float).to(device=device)
    
    dx = x[1]-x[0]
    u = th.stack((sample_c, sample_ct), dim=len(sample_c.shape))
    
    # Add noise to everything except the initial condition
    u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
    
    # Initialize and set up the model
    model = FINN_DiffSorp(
        u = u,
        D = np.array([0.5, 0.1]),
        BC = np.array([[1.0, 1.0], [0.0, 0.0]]),
        dx = dx,
        layer_sizes = config.model.layer_sizes,
        device = device,
        mode="train",
        learn_coeff=True
    ).to(device=device)


    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if print_progress:
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
        if print_progress: 
            print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.LBFGS(model.parameters(),
                                lr=config.training.learning_rate)

    criterion = nn.MSELoss(reduction="mean")

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty

    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):

        epoch_start_time = time.time()
        
        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # It is necessary for LBFGS optimizer, because it requires multiple
        # function evaluations
        def closure():
            # Set the model to train mode
            model.train()
                
            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()

            # Forward propagate and calculate loss function
            u_hat = model(t=t, u=u)

            mse = criterion(u_hat, u)

            c = th.linspace(0,2,501).unsqueeze(-1).to(device)
            ret_inv = model.func_nn(c)
            # We punish ret_inv[i] - ret_inv[i+1] > 0  <==> ret_inv[i] > ret_inv[i+1] 
            # aka. monotonically decreasing ret_inv is punished
            # thus monotonically increasing ret is punished
            # thus monotonically decreasing ret will yield a smaller loss
            mse += 100 * th.mean(th.relu(ret_inv[:-1] - ret_inv[1:]))
            
            mse.backward()
            
            # print(mse.item())
            # print(model.D)
                
            return mse
        
        optimizer.step(closure)
            
        # Extract the MSE value from the closure function
        mse = closure()
        
        epoch_errors_train.append(mse.item())

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]
            # Save the model to file (if desired)
            if config.training.save_model:
                # Start a separate thread to save the model
                thread = Thread(target=helpers.save_model_to_file(
                    model_src_path=os.path.abspath(""),
                    config=config,
                    epoch=epoch,
                    epoch_errors_train=epoch_errors_train,
                    epoch_errors_valid=epoch_errors_train,
                    net=model))
                thread.start()


        
        #
        # Print progress to the console
        if print_progress:
            print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}")

    b = time.time()
    if print_progress:
        print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')
    

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")