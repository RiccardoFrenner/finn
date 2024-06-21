import sys

sys.path.append("../../../../finn/UQnet")
sys.path.append("../../oldfinn/python/diffusion_sorption/synthetic_data/")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import syn00_config as params
import torch
import torch.nn as nn
import torchpinn as tp3
from torchdiffeq import odeint

print(tp3)


class AnalyticRetardation:
    @staticmethod
    def linear(u, por, rho_s, Kd):
        factor = 1 + (1 - por) / por * rho_s * Kd
        ones_like_u = u * 0 + 1
        return ones_like_u * factor

    @staticmethod
    def freundlich(u, por, rho_s, Kf, nf):
        return 1 + (1 - por) / por * rho_s * Kf * nf * (u + 1e-6) ** (nf - 1)

    @staticmethod
    def langmuir(u, por, rho_s, smax, Kl):
        return 1 + (1 - por) / por * rho_s * smax * Kl / ((u + Kl) ** 2)


class Initialize:
    def __init__(self):
        """
        Constructor

        """
        self.model_name: str = params.model_name
        self.main_path = Path().cwd()
        self.model_path = self.main_path / self.model_name
        self.model_path.mkdir(exist_ok=True, parents=True)

        self.device_name = params.device_name
        self.device = self.determine_device()

        # NETWORK HYPER-PARAMETERS
        self.flux_layers = params.flux_layers
        self.state_layers = params.state_layers
        self.flux_nodes = params.flux_nodes
        self.state_nodes = params.state_nodes
        self.error_mult = params.error_mult
        self.breakthrough_mult = params.breakthrough_mult
        self.profile_mult = params.profile_mult
        self.phys_mult = params.phys_mult
        self.epochs = params.epochs
        self.lbfgs_optim = params.lbfgs_optim
        self.train_breakthrough = params.train_breakthrough
        self.linear = params.linear
        self.freundlich = params.freundlich
        self.langmuir = params.langmuir

        # SIMULATION-RELATED INPUTS
        self.num_vars = params.num_vars

        # Soil Parameters
        self.D = params.D
        self.por = params.por
        self.rho_s = params.rho_s
        self.Kf = params.Kf
        self.nf = params.nf
        self.smax = params.smax
        self.Kl = params.Kl
        self.Kd = params.Kd
        self.solubility = params.solubility

        # Simulation Domain
        self.X = params.X
        self.dx = params.dx
        self.Nx = int(self.X / self.dx + 1)
        self.T = params.T
        self.dt = params.dt
        self.Nt = int(self.T / self.dt + 1)
        self.cauchy_val = self.dx

        # Inputs for Flux Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_flux = params.num_layers_flux
        self.num_nodes_flux = params.num_nodes_flux
        ## Set numerical stencil to be learnable or not
        self.learn_stencil = params.learn_stencil
        ## Effective diffusion coefficient for each variable
        self.D_eff = params.D_eff
        ## Set diffusion coefficient to be learnable or not
        self.learn_coeff = params.learn_coeff
        ## Set if diffusion coefficient to be approximated as a function
        self.coeff_func = params.coeff_func
        ## Normalizer for functions that are approximated with a NN
        self.p_exp_flux = params.p_exp_flux
        ## Set the variable index to be used when calculating the fluxes
        self.flux_calc_idx = params.flux_calc_idx
        ## Set the variable indices necessary to calculate the diffusion
        ## coefficient function
        self.flux_couple_idx = params.flux_couple_idx
        ## Set boundary condition types
        self.dirichlet_bool = params.dirichlet_bool
        self.neumann_bool = params.neumann_bool
        self.cauchy_bool = params.cauchy_bool
        ## Set the Dirichlet and Neumann boundary values if necessary,
        ## otherwise set = 0
        self.dirichlet_val = params.dirichlet_val
        self.neumann_val = params.neumann_val
        ## Set multiplier for the Cauchy boundary condition if necessary
        ## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
        self.cauchy_mult = params.cauchy_mult

        # Inputs for State Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_state = params.num_layers_state
        self.num_nodes_state = params.num_nodes_state
        ## Normalizer for the reaction functions that are approximated with a NN
        self.p_exp_state = params.p_exp_state
        ## Set the variable indices necessary to calculate the reaction function
        self.state_couple_idx = params.state_couple_idx

    def determine_device(self):
        """
        This function evaluates whether a GPU is accessible at the system and
        returns it as device to calculate on, otherwise it returns the CPU.
        :return: The device where tensor calculations shall be made on
        """

        self.device = torch.device(self.device_name)
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        print("Using device:", self.device)
        print()

        # Additional Info when using cuda
        if self.device.type == "cuda" and torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print(
                "\tAllocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB"
            )
            print(
                "\tCached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB"
            )
            print()

        return self.device


class Evaluate:
    def __init__(self, x, t, data):
        """
        Constructor

        Inputs:
            x       : the spatial coordinates of the data and model prediction
            t       : time (a torch.tensor array containing all values of time steps
                        in which the output of the model will be calculated and
                        recorded)
            data    : test data

        """

        self.x = x
        self.t = t
        self.data = data

    def evaluate(self, cfg, model, u0, unseen):
        """
        This function is the main function for postprocessing. It calculates
        the test prediction, compare it with the test data, and plot the results

        Inputs:
            params  : the configuration object containing the model settings
            model   : the trained model
            u0      : initial condition values, dim: [num_features, Nx, Ny]
            unseen  : a Boolean value to determine whether this object is the
                        extrapolated training case or the unseen test case

        """

        # Set the model to evaluation mode
        model.eval()

        # Calculate prediction using the trained model
        self.ode_pred = odeint(
            model, u0.to(cfg.device), self.t.to(cfg.device), rtol=1e-5, atol=1e-6
        )

        # Calculate MSE of the FINN prediction
        self.mse_test = torch.mean(
            (self.data.to(cfg.device) - self.ode_pred) ** 2
        ).item()

        # Extract the breakthrough curve prediction
        self.pred_breakthrough = self.ode_pred[:, 0, -1].squeeze()

        # Plot breakthrough curve if this object is the extrapolated training case
        if not unseen:
            self.plot_breakthrough_sep(cfg)

        # Plot the full field solution of both dissolved and total concentration
        self.plot_full_field(
            cfg, self.ode_pred[:, 0].squeeze(), self.data[:, 0].squeeze(), True
        )
        self.plot_full_field(
            cfg, self.ode_pred[:, 1].squeeze(), self.data[:, 1].squeeze(), False
        )

    def plot_breakthrough_sep(self, cfg):
        """
        This function plots the predicted breakthrough curve in comparison
        to the data

        Input:
            cfg : the configuration object containing the model settings

        """

        plt.figure()
        plt.plot(self.t, self.data[:, 0, -1].squeeze(), label="Data")

        # Determine index of the prediction to be plotted (so that the scatter
        # plot marker is sparse and visualization is better)
        train_plot_idx = torch.arange(1, 501, 50)
        test_plot_idx = torch.arange(501, 2001, 50)

        # Plot the predicted breakthrough curve, with different color for
        # training and the extrapolation
        plt.scatter(
            self.t[train_plot_idx],
            self.pred_breakthrough[train_plot_idx].cpu().detach(),
            label="Training",
            color="green",
            marker="x",
        )
        plt.scatter(
            self.t[test_plot_idx],
            self.pred_breakthrough[test_plot_idx].cpu().detach(),
            label="Testing",
            color="red",
            marker="x",
        )
        plt.legend(fontsize=16)

        # Plot a black vertical line as separator between training and extrapolation
        sep_t = torch.cat(2 * [torch.tensor(self.t[501]).unsqueeze(0)])
        sep_y = torch.tensor([0.0, 1.1 * torch.max(self.data[:, 0, -1])])
        plt.plot(sep_t, sep_y, color="black")

        # Determine caption depending on which isotherm is being used
        if cfg.linear:
            caption = "Linear"
        elif cfg.freundlich:
            caption = "Freundlich"
        elif cfg.langmuir:
            caption = "Langmuir"
        plt.title("Breakthrough Curve (" + caption + " Sorption)", fontsize=16)
        plt.xlabel("time [days]", fontsize=16)
        plt.ylabel("Tailwater concentration [mg/L]", fontsize=16)
        plt.tight_layout()
        plt.savefig(cfg.model_path / f"{cfg.model_name}_breakthrough_curve.png")

    def plot_full_field(self, cfg, pred, data, diss):
        """
        This function plots the full field solution of the model prediction in
        comparison to the test data

        Inputs:
            cfg     : the configuration object containing the model settings
            pred    : the model prediction
            data    : test data
            diss    : a Boolean value to determine whether this is a plot for
                        the dissolved or total concentration

        """

        plt.figure(figsize=(10.0, 5.0))
        plt.subplot(121)
        plt.pcolormesh(self.x, self.t, data)
        if diss:
            caption = "Dissolved"
            save_name = "diss"
        else:
            caption = "Total"
            save_name = "tot"
        plt.title(caption + " Concentration Data", fontsize=16)
        plt.xlabel("Depth [m]", fontsize=16)
        plt.ylabel("time [days]", fontsize=16)
        plt.colorbar()
        plt.clim([0, torch.max(data)])

        plt.subplot(122)
        plt.pcolormesh(self.x, self.t, pred.cpu().detach())
        plt.title(caption + " Concentration Prediction", fontsize=16)
        plt.xlabel("Depth [m]", fontsize=16)
        plt.ylabel("time [days]", fontsize=16)
        plt.colorbar()
        plt.clim([0, torch.max(data)])
        plt.tight_layout()
        plt.savefig(cfg.model_path / f"{cfg.model_name}_c_{save_name}.png")


def train(model: nn.Module, u0, t, x_train):
    """_summary_

    Args:
        model (nn.Module): model to be trained on training data
        u0 (tensor): initial condition, dim: [num_features, Nx]
        t (tensor): time steps for integration, dim: [Nt,]
        x_train (tensor): full field solution at each time step, dim: [Nt, num_features, Nx]
    """
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
    # TODO: Just a reminder
    # u = torch.linspace(0.01, 1.00, 100)


    # Define the closure function that consists of resetting the
    # gradient buffer, loss function calculation, and backpropagation
    # The closure function is necessary for LBFGS optimizer, because
    # it requires multiple function evaluations
    # The closure function returns the loss value
    def closure():
        model.train()
        optimizer.zero_grad()
        ode_pred = odeint(
            model,
            u0.to(cfg.device),
            t.to(cfg.device),
            rtol=1e-5,
            atol=1e-6,
        )

        # If trained only using breakthrough curve data calculate the
        # predicted breakthrough, else use the whole ode_pred values
        if cfg.train_breakthrough:
            # Extract the breakthrough curve prediction and data
            cauchy_mult = (
                model.flux_modules[0].cauchy_mult
                * model.flux_modules[0].D_eff
            )
            pred_breakthrough = (
                (ode_pred[:, 0, -2] - ode_pred[:, 0, -1]) * cauchy_mult
            ).squeeze()
            data_breakthrough = data[:, 0, -1].squeeze()

            # Calculate loss based on the breakthrough curve prediction and data
            loss = cfg.breakthrough_mult * torch.sum(
                (data_breakthrough.to(cfg.device) - pred_breakthrough) ** 2
            )

            # Extract the total concentration profile at t_end
            pred_profile = ode_pred[-1, 1].squeeze()
            data_profile = data[-1, 1].squeeze()

            # Calculate the loss based on the concentration profile prediction and data
            loss += cfg.profile_mult * torch.sum(
                (data_profile.to(cfg.device) - pred_profile) ** 2
            )

        else:
            # Calculate loss using the sum squared error metric
            loss = cfg.error_mult * torch.sum(
                (data.to(cfg.device) - ode_pred) ** 2
            )

        # Extract the predicted retardation factor function for physical
        # regularization
        u = torch.linspace(0.0, 1.0, 100).view(-1, 1).to(cfg.device)
        ret_temp = model.flux_modules[0].coeff_nn(u)

        # Physical regularization: value of the retardation factor should
        # decrease with increasing concentration
        loss += cfg.phys_mult * torch.sum(
            torch.relu(ret_temp[:-1] - ret_temp[1:])
        )

        # Backpropagate to obtain gradient of model parameters
        loss.backward()

        return loss

    # Plot the predicted retardation factor as a function of dissolved
    # concentration and update at each training epoch
    fig, ax = plt.subplots()
    u = torch.linspace(0.01, 1.00, 100).view(-1, 1).to(cfg.device)
    plt.plot(u.cpu(), retardation_linear, linestyle="--", label="Linear")
    plt.plot(
        u.cpu(), retardation_freundlich, linestyle="--", label="Freundlich"
    )
    plt.plot(u.cpu(), retardation_langmuir, linestyle="--", label="Langmuir")
    ret_pred = (
        1
        / model.flux_modules[0].coeff_nn(u)
        / 10 ** model.flux_modules[0].p_exp
    )
    (ax_pred,) = ax.plot(u.cpu(), ret_pred.cpu().detach(), label="FINN")
    plt.title("Predicted Retardation Factor", fontsize=16)
    plt.xlabel(r"$c_{diss}$ [mg/L]", fontsize=16)
    plt.ylabel(r"$R$", fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()

    # Iterate until maximum epoch number is reached
    for epoch in range(start_epoch, cfg.epochs):
        a = time.time()

        # Update the model parameters and record the loss value
        optimizer.step(closure)
        loss = closure()
        train_loss.append(loss.item())

        b = time.time()

        print(
            f"Training: Epoch [{epoch + 1}/{cfg.epochs}], "
            f"Training Loss: {train_loss[-1]:.4f}, Runtime: {b - a:.4f} secs"
        )

        # Update the retardation factor plot
        ret_pred = (
            1
            / model.flux_modules[0].coeff_nn(u)
            / 10 ** model.flux_modules[0].p_exp
        )
        ax_pred.set_ydata(ret_pred.cpu().detach())
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.0001)
        plt.savefig(
            cfg.model_path / f"{cfg.model_name}_retardation_{epoch}.png"
        )

    # Plot the retardation factor and save if required
    ret_pred = (
        1
        / model.flux_modules[0].coeff_nn(u)
        / 10 ** model.flux_modules[0].p_exp
    )
    ax_pred.set_ydata(ret_pred.cpu().detach())
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.0001)
    plt.savefig(cfg.model_path / f"{cfg.model_name}_retardation.png")


# Initialization of configurations and set modules for different core samples
cfg = Initialize()


class Flux_Kernels(nn.Module):
    def __init__(self, u0, cfg, var_idx):
        """
        Constructor
        Inputs:
            u0      : initial condition, dim: [Nx, Ny]
            cfg     : configuration object of the model setup, containing boundary
                        condition types, values, learnable parameter settings, etc.
            var_idx : index of the calculated variable (could be > 1 for coupled
                        systems)
        """

        super(Flux_Kernels, self).__init__()

        self.Nx = u0.size(0)
        self.Ny = u0.size(1)
        self.u0 = u0

        self.device = cfg.device

        # Variables that act as switch to use different types of boundary
        # condition
        # Each variable consists of boolean values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to
        # be no-flux boundaries (zero neumann_val)
        self.dirichlet_bool = cfg.dirichlet_bool[var_idx]
        self.neumann_bool = cfg.neumann_bool[var_idx]
        self.cauchy_bool = cfg.cauchy_bool[var_idx]

        # Variables that store the values of the boundary condition of each type
        # Values = 0 if not used, otherwise specify in the configuration file
        # Each variable consists of real values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to
        # be no-flux boundaries
        if torch.is_tensor(cfg.dirichlet_val[var_idx]):
            self.dirichlet_val = cfg.dirichlet_val[var_idx].to(cfg.device)
        else:
            self.dirichlet_val = torch.tensor(cfg.dirichlet_val[var_idx]).to(cfg.device)

        if torch.is_tensor(cfg.neumann_val[var_idx]):
            self.neumann_val = cfg.neumann_val[var_idx].to(cfg.device)
        else:
            self.neumann_val = torch.tensor(cfg.neumann_val[var_idx]).to(cfg.device)

        # For Cauchy BC, the initial Cauchy value is set to be the initial
        # condition at each corresponding domain boundary, and will be updated
        # through time
        self.cauchy_val = []
        self.cauchy_val.append(u0[0, :].to(cfg.device))
        self.cauchy_val.append(u0[-1, :].to(cfg.device))
        self.cauchy_val.append(u0[:, 0].to(cfg.device))
        self.cauchy_val.append(u0[:, -1].to(cfg.device))

        # Set the Cauchy BC multiplier (to be multiplied with the gradient of
        # the unknown variable and the diffusion coefficient)
        if torch.is_tensor(cfg.cauchy_mult[var_idx]):
            self.cauchy_mult = cfg.cauchy_mult[var_idx].to(cfg.device)
        else:
            self.cauchy_mult = torch.tensor(cfg.cauchy_mult[var_idx]).to(cfg.device)

        # If numerical stencil is to be learned, initialize to +1 and -1 with
        # a standard deviation of 0.1 each, otherwise set it to fixed values
        self.learn_stencil = cfg.learn_stencil[var_idx]
        if self.learn_stencil:
            self.stencil = torch.tensor(
                [
                    torch.normal(torch.tensor([1.0]), torch.tensor([0.1])),
                    torch.normal(torch.tensor([-1.0]), torch.tensor([0.1])),
                ],
                dtype=torch.float,
            )
            self.stencil = nn.Parameter(self.stencil)
        else:
            self.stencil = torch.tensor([1.0, -1.0])

        # Extract the diffusion coefficient scalar value and set to be learnable
        # if desired
        if torch.is_tensor(cfg.D_eff[var_idx]):
            self.D_eff = cfg.D_eff[var_idx].to(cfg.device)
        else:
            self.D_eff = torch.tensor(cfg.D_eff[var_idx]).to(cfg.device)
        if cfg.learn_coeff[var_idx]:
            self.D_eff = nn.Parameter(torch.tensor([self.D_eff], dtype=torch.float))

        # Extract the boolean value to determine whether the diffusion coefficient
        # is a function of the unknown variable
        self.coeff_func = cfg.coeff_func[var_idx]

        # Extract value of the normalizing constant to be applied to the output
        # of the NN that predicts the diffusion coefficient function
        if torch.is_tensor(cfg.p_exp_flux[var_idx]):
            self.p_exp = cfg.p_exp_flux[var_idx].to(cfg.device)
        else:
            self.p_exp = torch.tensor(cfg.p_exp_flux[var_idx]).to(cfg.device)

        # Initialize a NN to predict diffusion coefficient as a function of
        # the unknown variable if necessary
        if self.coeff_func:
            self.coeff_nn = Coeff_NN(
                cfg.num_layers_flux[var_idx],
                cfg.num_nodes_flux[var_idx],
                len(cfg.flux_couple_idx[var_idx]),
            ).to(cfg.device)
            self.p_exp = nn.Parameter(torch.tensor([self.p_exp], dtype=torch.float))

    def forward(self, u_main, u_coupled, t):
        """
        The forward function calculates the integrated flux between each control
        volume and its neighbors

        Inputs:
            u_main      : the unknown variable to be used to calculate the flux
                            indexed by flux_calc_idx[var_idx]
                            dim: [1, Nx, Ny]

            u_coupled   : all necessary unknown variables required to calculate
                          the diffusion coeffient as a function, indexed by
                          flux_couple_idx[var_idx]
                          dim: [num_features, Nx, Ny]

            t           : time (scalar value, taken from the ODE solver)

        Output:
            flux        : the integrated flux for all control volumes
                            dim: [Nx, Ny]

        """

        # Reshape the input dimension for the coeff_nn model into [Nx, Ny, num_features]
        u_coupled = u_coupled.permute(1, 2, 0)

        # Calculate the flux multiplier (diffusion coefficient function) if set
        # to be a function, otherwise set as tensor of ones
        if self.coeff_func:
            flux_mult = self.coeff_nn(u_coupled).squeeze(2) * 10**self.p_exp
        else:
            flux_mult = torch.ones(self.Nx, self.Ny)

        flux_mult = flux_mult.to(self.device)

        # Squeeze the u_main dimension into [Nx, Ny]
        u_main = u_main.squeeze(0)

        # Left Boundary Condition
        if self.dirichlet_bool[0]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            left_bound_flux = (
                (
                    self.stencil[0] * self.dirichlet_val[0]
                    + self.stencil[1] * u_main[0, :]
                ).unsqueeze(0)
                * self.D_eff
                * flux_mult[0, :]
            )

        elif self.neumann_bool[0]:
            # If Neumann, set the Neumann value as the flux at the boundary
            left_bound_flux = torch.cat(
                self.Ny * [self.neumann_val[0].unsqueeze(0)]
            ).unsqueeze(0)

        elif self.cauchy_bool[0]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[0] = self.u0[0, :].to(self.device)
            else:
                self.cauchy_val[0] = (
                    (u_main[0, :] - self.cauchy_val[0]) * self.cauchy_mult * self.D_eff
                )
            # Calculate the flux at the boundary using the updated Cauchy value
            left_bound_flux = (
                (
                    self.stencil[0] * self.cauchy_val[0]
                    + self.stencil[1] * u_main[0, :]
                ).unsqueeze(0)
                * self.D_eff
                * flux_mult[0, :]
            )

        # Calculate the fluxes of each control volume with its left neighboring cell
        left_neighbors = (
            (self.stencil[0] * u_main[:-1, :] + self.stencil[1] * u_main[1:, :])
            * self.D_eff
            * flux_mult[1:, :]
        )
        # Concatenate the left boundary fluxes with the left neighbors fluxes
        left_flux = torch.cat((left_bound_flux, left_neighbors))

        # Right Boundary Condition
        if self.dirichlet_bool[1]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            right_bound_flux = (
                (
                    self.stencil[0] * self.dirichlet_val[1]
                    + self.stencil[1] * u_main[-1, :]
                ).unsqueeze(0)
                * self.D_eff
                * flux_mult[-1, :]
            )

        elif self.neumann_bool[1]:
            # If Neumann, set the Neumann value as the flux at the boundary
            right_bound_flux = torch.cat(
                self.Ny * [self.neumann_val[1].unsqueeze(0)]
            ).unsqueeze(0)

        elif self.cauchy_bool[1]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[1] = self.u0[-1, :].to(self.device)
            else:
                self.cauchy_val[1] = (
                    (u_main[-1, :] - self.cauchy_val[1]) * self.cauchy_mult * self.D_eff
                )
            # Calculate the flux at the boundary using the updated Cauchy value
            right_bound_flux = (
                (
                    self.stencil[0] * self.cauchy_val[1]
                    + self.stencil[1] * u_main[-1, :]
                ).unsqueeze(0)
                * self.D_eff
                * flux_mult[-1, :]
            )

        # Calculate the fluxes of each control volume with its right neighboring cell
        right_neighbors = (
            (self.stencil[0] * u_main[1:, :] + self.stencil[1] * u_main[:-1, :])
            * self.D_eff
            * flux_mult[:-1, :]
        )
        # Concatenate the right neighbors fluxes with the right boundary fluxes
        right_flux = torch.cat((right_neighbors, right_bound_flux))

        # Top Boundary Condition
        if self.dirichlet_bool[2]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            top_bound_flux = (
                (
                    self.stencil[0] * self.dirichlet_val[2]
                    + self.stencil[1] * u_main[:, 0]
                ).unsqueeze(1)
                * self.D_eff
                * flux_mult[:, 0]
            )

        elif self.neumann_bool[2]:
            # If Neumann, set the Neumann value as the flux at the boundary
            top_bound_flux = torch.cat(
                self.Nx * [self.neumann_val[2].unsqueeze(0)]
            ).unsqueeze(1)

        elif self.cauchy_bool[2]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[2] = self.u0[:, 0].to(self.device)
            else:
                self.cauchy_val[2] = (
                    (u_main[:, 0] - self.cauchy_val[2]) * self.cauchy_mult * self.D_eff
                )
            # Calculate the flux at the boundary using the updated Cauchy value
            top_bound_flux = (
                (
                    self.stencil[0] * self.cauchy_val[2]
                    + self.stencil[1] * u_main[:, 0]
                ).unsqueeze(1)
                * self.D_eff
                * flux_mult[:, 0]
            )

        # Calculate the fluxes of each control volume with its top neighboring cell
        top_neighbors = (
            (self.stencil[0] * u_main[:, :-1] + self.stencil[1] * u_main[:, 1:])
            * self.D_eff
            * flux_mult[:, 1:]
        )
        # Concatenate the top boundary fluxes with the top neighbors fluxes
        top_flux = torch.cat((top_bound_flux, top_neighbors), dim=1)

        # Bottom Boundary Condition
        if self.dirichlet_bool[3]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            bottom_bound_flux = (
                (
                    self.stencil[0] * self.dirichlet_val[3]
                    + self.stencil[1] * u_main[:, -1]
                ).unsqueeze(1)
                * self.D_eff
                * flux_mult[:, -1]
            )

        elif self.neumann_bool[3]:
            # If Neumann, set the Neumann value as the flux at the boundary
            bottom_bound_flux = torch.cat(
                self.Nx * [self.neumann_val[3].unsqueeze(0)]
            ).unsqueeze(1)

        elif self.cauchy_bool[3]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[3] = self.u0[:, -1].to(self.device)
            else:
                self.cauchy_val[3] = (
                    (u_main[:, -1] - self.cauchy_val[3]) * self.cauchy_mult * self.D_eff
                )
            # Calculate the flux at the boundary using the updated Cauchy value
            bottom_bound_flux = (
                (
                    self.stencil[0] * self.cauchy_val[3]
                    + self.stencil[1] * u_main[:, -1]
                ).unsqueeze(1)
                * self.D_eff
                * flux_mult[:, -1]
            )

        # Calculate the fluxes of each control volume with its bottom neighboring cell
        bottom_neighbors = (
            (self.stencil[0] * u_main[:, 1:] + self.stencil[1] * u_main[:, :-1])
            * self.D_eff
            * flux_mult[:, :-1]
        )
        # Concatenate the bottom neighbors fluxes with the bottom boundary fluxes
        bottom_flux = torch.cat((bottom_neighbors, bottom_bound_flux), dim=1)

        # Integrate all fluxes at all control volume boundaries
        flux = left_flux + right_flux + top_flux + bottom_flux

        return flux


class Coeff_NN(torch.nn.Module):
    """
    The class Coeff_NN constructs a feedforward NN required for calculation
    of diffusion coefficient as a function of u
    It will be called in the Flux_Kernels constructor if the cfg.coeff_func is
    set to be True
    """

    def __init__(self, num_layers, num_nodes, num_vars):
        """
        Constructor

        Inputs:
            num_layers  : number of hidden layers (excluding output layer)
            num_nodes   : number of hidden nodes in each hidden layer
            num_vars    : number of features used as inputs

        """

        super(Coeff_NN, self).__init__()

        # Initialize the layer as an empty list
        layer = []

        # Add sequential layers as many as the specified num_layers, append
        # to the layer list, including the output layer (hence the +1 in the
        # iteration range)
        for i in range(num_layers + 1):
            # Specify number of input and output features for each layer
            in_features = num_nodes
            out_features = num_nodes

            # If it is the first hidden layer, set the number of input features
            # to be = num_vars
            if i == 0:
                in_features = num_vars
            # If it is the output layer, set the number of output features to be = 1
            elif i == num_layers:
                out_features = 1

            # Create sequential layer, if output layer use sigmoid activation function
            if i < num_layers:
                layer.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features, out_features), torch.nn.Tanh()
                    )
                )
            else:
                layer.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features, out_features), torch.nn.Sigmoid()
                    )
                )

        # Convert the list into a sequential module
        self.layers = torch.nn.Sequential(*layer)

    def forward(self, input):
        """
        The forward function calculates the approximation of diffusion coefficient
        function using the specified input values

        Input:
            input   : input for the function, all u that is required to calculate
                        the diffusion coefficient function (could be coupled with
                        other variables), dim: [Nx, Ny, num_features]

        Output:
            output      : the approximation of the diffusion coefficient function

        """

        output = self.layers(input)

        return output


class Net_Model(nn.Module):
    def __init__(self, u0, cfg):
        """
        Constructor
        Inputs:
            u0      : initial condition, dim: [num_features, Nx, Ny]
            cfg     : configuration object of the model setup, containing boundary
                        condition types, values, learnable parameter settings, etc.
        """

        super(Net_Model, self).__init__()

        self.flux_modules = nn.ModuleList()
        self.num_vars = u0.size(0)
        self.cfg = cfg

        # Create flux kernel for each variable to be calculated
        for var_idx in range(self.num_vars):
            self.flux_modules.append(Flux_Kernels(u0[var_idx], self.cfg, var_idx))

    def forward(self, t, u):
        """
        The forward function calculates du/dt to be put into the ODE solver

        Inputs:
            t   : time (scalar value, taken from the ODE solver)
            u   : the unknown variables to be calculated taken from the previous
                    time step, dim: [num_features, Nx, Ny]

        Output:
            du  : the time derivative of u (du/dt), dim: [num_features, Nx, Ny]

        """
        flux = []

        # Use flux and state kernels to calculate du/dt for all unknown variables
        for var_idx in range(self.num_vars):
            flux.append(
                self.flux_modules[var_idx](
                    u[self.cfg.flux_calc_idx[var_idx]],
                    u[self.cfg.flux_couple_idx[var_idx]],
                    t,
                )
            )

        du = torch.stack(flux)

        return du


def load_data(is_testing: bool = False):
    # Load training data with the original dimension of [Nx, Nt]
    # Reshape into [Nt, 1, Nx, Ny], with Ny = 1
    folder = "../../oldfinn/python/diffusion_sorption/synthetic_data/"
    if cfg.linear:
        folder += "data_linear"
    elif cfg.freundlich:
        folder += "data_freundlich"
    elif cfg.langmuir:
        folder += "data_langmuir"
    else:
        raise ValueError("Invalid config entry")

    c_diss = pd.read_csv(
        folder + f'/c_diss{"_test" if is_testing else ""}.csv', sep="\t", header=None
    )
    c_diss = torch.tensor(np.array(c_diss)).unsqueeze(1)
    c_diss = c_diss.permute(2, 0, 1).unsqueeze(1)

    c_tot = pd.read_csv(
        folder + f'/c_tot{"_test" if is_testing else ""}.csv', sep="\t", header=None
    )
    c_tot = torch.tensor(np.array(c_tot)).unsqueeze(1)
    c_tot = c_tot.permute(2, 0, 1).unsqueeze(1)

    # Concatenate dissolved and total concentration data along the second dimension
    # (dim=1 in Python)
    return torch.cat((c_diss, c_tot), dim=1)


def main():
    u0 = torch.zeros(cfg.num_vars, cfg.Nx, 1)
    model = Net_Model(u0, cfg)

    # Train the model
    train_data = load_data()
    x = torch.linspace(0.0, cfg.X, cfg.Nx)
    t = torch.linspace(0.0, cfg.T, cfg.Nt)
    trainer = Training(model, cfg)
    trainer.model_train(u0, t[:51], train_data[:51])

    # Evaluate the trained model with extrapolation of the train dataset
    extrapolate_train = Evaluate(x, t, train_data)
    extrapolate_train.evaluate(cfg, model, u0, False)

    # Evaluate the trained model with unseen test dataset
    model_test = model
    model_test.flux_modules[0].dirichlet_val[0] = 0.7
    model_test.flux_modules[1].dirichlet_val[0] = 0.7
    test_data = load_data(is_testing=True)
    unseen_test = test.Evaluate(x, t, test_data)
    unseen_test.evaluate(cfg, model_test, u0, True)


if __name__ == "__main__":
    main()
