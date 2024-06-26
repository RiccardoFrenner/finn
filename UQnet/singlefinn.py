import sys

sys.path.append("../../../../finn/UQnet")
sys.path.append("../../oldfinn/python/diffusion_sorption/synthetic_data/")

import time
from pathlib import Path

import numpy as np
import syn00_config as params
import torch
import torch.nn as nn
import torchpinn as tp3
from lib import AnalyticRetardation, Flux_Kernels, create_mlp, load_data
from torchdiffeq import odeint

print(tp3)


class RetardationInverse(torch.nn.Module):
    """
    The class RetardationInverse constructs a feedforward NN for calculating the retardation factor as a function of u.
    It will be called in the Flux_Kernels constructor if the cfg.is_retardation_a_func is set to be True.
    """

    def __init__(self, num_layers, num_nodes):
        """
        Constructor

        Inputs:
            num_layers  : number of hidden layers (excluding output layer)
            num_nodes   : number of hidden nodes in each hidden layer
        """

        super(RetardationInverse, self).__init__()

        layers = [1] + [num_nodes] * num_layers + [1]
        activation_fun = nn.Tanh()
        activation_fun_end = nn.Sigmoid()
        self.layers = create_mlp(layers, activation_fun, activation_fun_end)

    def forward(self, u):
        """Computes the approximation of retardation factor function."""
        return self.layers(u)


class ConcentrationPredictor(nn.Module):
    def __init__(self, u0: torch.Tensor, cfg, ret_inv_funs=None):
        """TODO: Docstring

        Args:
            u0 (tensor): initial condition, dim: [num_features, Nx]
            cfg (_type_): _description_
        """
        super(ConcentrationPredictor, self).__init__()
        if ret_inv_funs is None:
            ret_inv_funs = [None] * len(u0)

        self.cfg = cfg
        self.u0 = u0
        self.dudt_fun = ConcentrationChangeRatePredictor(
            u0, cfg, ret_inv_funs=ret_inv_funs
        )

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
    def __init__(self, u0, cfg, ret_inv_funs=None):
        """
        Constructor
        Inputs:
            u0      : initial condition, dim: [num_features, Nx]
            cfg     : configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
        """
        if ret_inv_funs is None:
            ret_inv_funs = [None] * len(u0)

        super(ConcentrationChangeRatePredictor, self).__init__()

        self.flux_modules = nn.ModuleList()
        self.num_vars = u0.size(0)
        self.cfg = cfg

        # Create flux kernel for each variable to be calculated
        for var_idx in range(self.num_vars):
            self.flux_modules.append(
                Flux_Kernels(
                    u0[var_idx], self.cfg, var_idx, ret_inv_fun=ret_inv_funs[var_idx]
                )
            )

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
            # TODO: This is weird. Why is u_main the same as u_coupled?
            flux.append(self.flux_modules[var_idx](u[[0]], u[[0]], t))

        du = torch.stack(flux)

        return du


def main():
    cfg = params

    u0 = torch.zeros(cfg.num_vars, cfg.Nx, 1)
    model = ConcentrationPredictor(
        u0=u0,
        cfg=cfg,
        ret_inv_funs=[
            (
                RetardationInverse(
                    cfg.num_layers_flux[var_idx],
                    cfg.num_nodes_flux[var_idx],
                ).to(cfg.device)
                if is_fun
                else None
            )
            for (var_idx, is_fun) in enumerate(cfg.is_retardation_a_func)
        ],
    )

    # Train the model
    train_data = load_data("freundlich")
    # x = torch.linspace(0.0, cfg.X, cfg.Nx)
    t = torch.linspace(0.0, cfg.T, cfg.Nt)
    model.run_training(t=t[:51], u_full_train=train_data[:51])

    # TODO: Evaluate the trained model with unseen test dataset


if __name__ == "__main__":
    main()
