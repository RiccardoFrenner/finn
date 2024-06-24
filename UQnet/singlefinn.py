import sys

sys.path.append("../../../../finn/UQnet")
sys.path.append("../../oldfinn/python/diffusion_sorption/synthetic_data/")

import time
from pathlib import Path

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
        self.device = torch.device("cpu")

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
        ## Set retardation factor to be approximated as a function
        self.is_retardation_a_func = params.is_retardation_a_func
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

        if torch.is_tensor(cfg.D_eff[var_idx]):
            self.D_eff = cfg.D_eff[var_idx].to(cfg.device)
        else:
            self.D_eff = torch.tensor(cfg.D_eff[var_idx]).to(cfg.device)
        if cfg.learn_coeff[var_idx]:
            self.D_eff = nn.Parameter(torch.tensor([self.D_eff], dtype=torch.float))

        self.is_retardation_a_func = cfg.is_retardation_a_func[var_idx]

        # Extract value of the normalizing constant to be applied to the output
        # of the NN that predicts the diffusion coefficient function
        if torch.is_tensor(cfg.p_exp_flux[var_idx]):
            self.p_exp = cfg.p_exp_flux[var_idx].to(cfg.device)
        else:
            self.p_exp = torch.tensor(cfg.p_exp_flux[var_idx]).to(cfg.device)

        # Initialize a NN to predict retardation factor as a function of
        # the unknown variable if necessary
        if self.is_retardation_a_func:
            self.ret_inv_fun = RetardationInverse(
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
                          the retardation factor as a function, indexed by
                          flux_couple_idx[var_idx]
                          dim: [num_features, Nx, Ny]

            t           : time (scalar value, taken from the ODE solver)

        Output:
            flux        : the integrated flux for all control volumes
                            dim: [Nx, Ny]

        """

        # Reshape the input dimension for the retardation model into [Nx, Ny, num_features]
        u_coupled = u_coupled.permute(1, 2, 0)

        # Calculate the flux multiplier (retardation function) if set
        # to be a function, otherwise set as tensor of ones
        if self.is_retardation_a_func:
            ret_inv = self.ret_inv_fun(u_coupled).squeeze(2) * 10**self.p_exp
        else:
            ret_inv = torch.ones(self.Nx, self.Ny)

        ret_inv = ret_inv.to(self.device)

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
                * ret_inv[0, :]
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
                * ret_inv[0, :]
            )

        # Calculate the fluxes of each control volume with its left neighboring cell
        left_neighbors = (
            (self.stencil[0] * u_main[:-1, :] + self.stencil[1] * u_main[1:, :])
            * self.D_eff
            * ret_inv[1:, :]
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
                * ret_inv[-1, :]
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
                * ret_inv[-1, :]
            )

        # Calculate the fluxes of each control volume with its right neighboring cell
        right_neighbors = (
            (self.stencil[0] * u_main[1:, :] + self.stencil[1] * u_main[:-1, :])
            * self.D_eff
            * ret_inv[:-1, :]
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
                * ret_inv[:, 0]
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
                * ret_inv[:, 0]
            )

        # Calculate the fluxes of each control volume with its top neighboring cell
        top_neighbors = (
            (self.stencil[0] * u_main[:, :-1] + self.stencil[1] * u_main[:, 1:])
            * self.D_eff
            * ret_inv[:, 1:]
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
                * ret_inv[:, -1]
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
                * ret_inv[:, -1]
            )

        # Calculate the fluxes of each control volume with its bottom neighboring cell
        bottom_neighbors = (
            (self.stencil[0] * u_main[:, 1:] + self.stencil[1] * u_main[:, :-1])
            * self.D_eff
            * ret_inv[:, :-1]
        )
        # Concatenate the bottom neighbors fluxes with the bottom boundary fluxes
        bottom_flux = torch.cat((bottom_neighbors, bottom_bound_flux), dim=1)

        # Integrate all fluxes at all control volume boundaries
        flux = left_flux + right_flux + top_flux + bottom_flux

        return flux


class RetardationInverse(torch.nn.Module):
    """
    The class RetardationInverse constructs a feedforward NN for calculating the retardation factor as a function of u.
    It will be called in the Flux_Kernels constructor if the cfg.is_retardation_a_func is set to be True.
    """

    def __init__(self, num_layers, num_nodes, num_vars):
        """
        Constructor

        Inputs:
            num_layers  : number of hidden layers (excluding output layer)
            num_nodes   : number of hidden nodes in each hidden layer
            num_vars    : number of features used as inputs

        """

        super(RetardationInverse, self).__init__()

        layer = []
        for i in range(num_layers + 1):
            in_features = num_nodes
            out_features = num_nodes

            if i == 0:
                in_features = num_vars
            elif i == num_layers:
                out_features = 1

            layer.append(torch.nn.Linear(in_features, out_features))
            if i < num_layers:
                layer.append(torch.nn.Tanh())
            else:
                layer.append(torch.nn.Sigmoid())

        self.layers = torch.nn.Sequential(*layer)

    def forward(self, input):
        """
        The forward function calculates the approximation of retardation factor
        function using the specified input values

        Input:
            input   : input for the function, all u that is required to calculate
                        the retardation factor function (could be coupled with
                        other variables), dim: [Nx, Ny, num_features]

        Output:
            output      : the approximation of the retardation factor function

        """

        output = self.layers(input)

        return output


class ConcentrationPredictor(nn.Module):
    def __init__(self, u0: torch.Tensor, cfg: Initialize):
        """TODO: Docstring

        Args:
            u0 (tensor): initial condition, dim: [num_features, Nx]
            cfg (_type_): _description_
        """
        super(ConcentrationPredictor, self).__init__()

        self.cfg = cfg
        self.u0 = u0
        self.dudt_fun = ConcentrationChangeRatePredictor(u0, cfg)

    def forward(self, t):
        """Predict the concentration profile at given time steps from an initial condition using the FINN method.

        Args:
            t (tensor): time steps

        Returns:
            tensor: Full field solution of concentration at given time steps.
        """

        return odeint(self.dudt_fun, self.u0, t, rtol=1e-5, atol=1e-6)

    def run_training(self, t: torch.Tensor, u_full_train: torch.Tensor):
        """Train to predict the concentration from the given full field training data.

        Args:

            t (tensor): time steps for integration, dim: [Nt,]
            x_train (tensor): full field solution at each time step, dim: [Nt, num_features, Nx]
        """
        out_dir = Path("data_out")
        out_dir.mkdir(exist_ok=True, parents=True)

        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1)

        u_ret = torch.linspace(0.0, 1.0, 100).view(-1, 1).to(self.cfg.device)
        # TODO: Should not be here
        ret_linear = AnalyticRetardation.linear(
            u_ret, por=self.cfg.por, rho_s=self.cfg.rho_s, Kd=self.cfg.Kd
        )
        ret_freundlich = AnalyticRetardation.freundlich(
            u_ret,
            por=self.cfg.por,
            rho_s=self.cfg.rho_s,
            Kf=self.cfg.Kf,
            nf=self.cfg.nf,
        )
        ret_langmuir = AnalyticRetardation.langmuir(
            u_ret,
            por=self.cfg.por,
            rho_s=self.cfg.rho_s,
            smax=self.cfg.smax,
            Kl=self.cfg.Kl,
        )
        np.save(out_dir / "u_ret.npy", u_ret)
        np.save(out_dir / "retardation_linear.npy", ret_linear)
        np.save(out_dir / "retardation_freundlich.npy", ret_freundlich)
        np.save(out_dir / "retardation_langmuir.npy", ret_langmuir)

        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # The closure function is necessary for LBFGS optimizer, because
        # it requires multiple function evaluations
        # The closure function returns the loss value
        def closure():
            self.train()
            optimizer.zero_grad()
            ode_pred = self.forward(t)  # aka. y_pred
            # TODO: mean instead of sum?
            loss = self.cfg.error_mult * torch.sum((u_full_train - ode_pred) ** 2)

            # Physical regularization: value of the retardation factor should decrease with increasing concentration
            ret_inv_pred = self.retardation_inv_scaled(u_ret)
            loss += self.cfg.phys_mult * torch.sum(
                torch.relu(ret_inv_pred[:-1] - ret_inv_pred[1:])
            )  # TODO: mean instead of sum?

            loss.backward()

            return loss

        # Iterate until maximum epoch number is reached
        for epoch in range(1, self.cfg.epochs + 1):
            dt = time.time()
            optimizer.step(closure)
            loss = closure()
            dt = time.time() - dt

            print(
                f"Training: Epoch [{epoch + 1}/{self.cfg.epochs}], "
                f"Training Loss: {loss.item():.4f}, Runtime: {dt:.4f} secs"
            )

            ret_pred_path = self.cfg.model_path / f"retPred_{epoch}.npy"
            np.save(ret_pred_path, self.retardation(u_ret).detach().numpy())

    def retardation_inv_scaled(self, u):
        return self.dudt_fun.flux_modules[0].ret_inv_fun(u)

    def retardation(self, u):
        return (
            1.0
            / self.dudt_fun.flux_modules[0].ret_inv_fun(u)
            / 10 ** self.dudt_fun.flux_modules[0].p_exp
        )


class ConcentrationChangeRatePredictor(nn.Module):
    def __init__(self, u0, cfg):
        """
        Constructor
        Inputs:
            u0      : initial condition, dim: [num_features, Nx]
            cfg     : configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
        """

        super(ConcentrationChangeRatePredictor, self).__init__()

        self.flux_modules = nn.ModuleList()
        self.num_vars = u0.size(0)
        self.cfg = cfg

        # Create flux kernel for each variable to be calculated
        for var_idx in range(self.num_vars):
            self.flux_modules.append(Flux_Kernels(u0[var_idx], self.cfg, var_idx))

    def forward(self, t, u):
        """Computes du/dt to be put into the ODE solver

        Args:
            t (float): time point
            u (tensor): the unknown variables to be calculated taken from the previous time step, dim: [num_features, Nx]

        Returns:
            tensor: the time derivative of u (du/dt), dim: [num_features, Nx]
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


def load_data(cfg: Initialize, is_testing: bool = False):
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

    folder = Path(folder).resolve()

    c_diss = pd.read_csv(
        folder / f'c_diss{"_test" if is_testing else ""}.csv', sep="\t", header=None
    )
    c_diss = torch.tensor(np.array(c_diss)).unsqueeze(1)
    c_diss = c_diss.permute(2, 0, 1).unsqueeze(1)

    c_tot = pd.read_csv(
        folder / f'c_tot{"_test" if is_testing else ""}.csv', sep="\t", header=None
    )
    c_tot = torch.tensor(np.array(c_tot)).unsqueeze(1)
    c_tot = c_tot.permute(2, 0, 1).unsqueeze(1)

    # Concatenate dissolved and total concentration data along the second dimension
    # (dim=1 in Python)
    return torch.cat((c_diss, c_tot), dim=1)


def main():
    cfg = Initialize()

    u0 = torch.zeros(cfg.num_vars, cfg.Nx, 1)
    model = ConcentrationPredictor(u0=u0, cfg=cfg)

    # Train the model
    train_data = load_data(cfg)
    x = torch.linspace(0.0, cfg.X, cfg.Nx)
    t = torch.linspace(0.0, cfg.T, cfg.Nt)
    model.run_training(t=t[:51], u_full_train=train_data[:51])

    # TODO: Evaluate the trained model with unseen test dataset


if __name__ == "__main__":
    main()
