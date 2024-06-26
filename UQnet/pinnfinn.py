# import sys

# sys.path.append("../../../../finn/UQnet")

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import params
import torch
import torch.nn as nn
import torchpinn as tp3
from lib import AnalyticRetardation, Flux_Kernels, create_mlp, load_data
from torchdiffeq import odeint

print(tp3)


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
        optimizer.zero_grad()

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
            loss = optimizer.step(closure)
            # loss = closure()
            dt = time.time() - dt

            print(
                f"Training: Epoch [{epoch + 1}/{self.cfg.epochs}], "
                f"Training Loss: {loss:.4f}, Runtime: {dt:.4f} secs"
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


class RetardationInverseStd(nn.Module):
    def __init__(self, layers: list[int]):
        super(RetardationInverseStd, self).__init__()

        self.layers = create_mlp(layers, nn.ReLU(), nn.Identity())

    def forward(self, u):
        u = torch.sqrt(torch.square(u) + 0.2)
        return u


def create_PI_training_data(y_pred_mean, X, Y) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate std training data"""
    # threshold = 30
    with torch.no_grad():
        diff_train = Y.detach() - y_pred_mean.detach()
        # print(f"{diff_train.shape=}")
        dist = torch.sum(diff_train**2, dim=[1, 2, 3])
        threshold = torch.quantile(dist, 0.5)
        # print(f"{threshold=}")
        mask = dist < threshold
        # print(f"{mask.shape=}")
        # print(f"{X.shape=}")
        # print(f"{Y.shape=}")

        X_std = X[~mask].clone().detach().requires_grad_(False)
        Y_std = diff_train[~mask].clone().detach().requires_grad_(False)
        # print(f"{X_std.shape=}")

    return X_std, Y_std


def main():
    cfg = params

    u0 = torch.zeros(cfg.num_vars, cfg.Nx, 1)
    ret_inv_mean_models = [
        (create_mlp([1, 15, 15, 15, 1], nn.Tanh(), nn.Sigmoid()) if is_fun else None)
        for (var_idx, is_fun) in enumerate(cfg.is_retardation_a_func)
    ]
    concentration_predictor = ConcentrationPredictor(
        u0=u0.clone(), cfg=cfg, ret_inv_funs=ret_inv_mean_models
    )

    # Load data
    u_analytical = load_data("freundlich")
    # x = torch.linspace(0.0, cfg.X, cfg.Nx)
    t = torch.linspace(0.0, cfg.T, cfg.Nt)

    train_split_index = 51
    x_train = t[:train_split_index]
    y_train = u_analytical[:train_split_index]
    x_valid = t[train_split_index:]
    y_valid = u_analytical[train_split_index:]

    # Train the concentration predictor
    concentration_predictor.run_training(t=x_train, u_full_train=y_train)
    with torch.no_grad():
        y_pred_mean = concentration_predictor(x_train)

    ret_inv_std_model = RetardationInverseStd([1, 15, 15, 15, 1])

    # Train the standard deviation concentration model to get a trained ret_inv_std_model
    # ========================
    # This works:
    x_train_std = torch.linspace(0, cfg.T, cfg.Nt)[:train_split_index]
    y_train_std = y_train.clone()
    # ========================
    # But this does not work:
    # torch.cuda.empty_cache()
    # x_train_std_, y_train_std_ = create_PI_training_data(
    #     y_pred_mean.clone().detach(),
    #     x_train.clone().detach(),
    #     y_train.clone().detach(),
    # )
    # # save as npy files
    # np.save("x_train_std.npy", x_train_std_.numpy())
    # np.save("y_train_std.npy", y_train_std_.numpy())
    # x_train_std = torch.from_numpy(np.load("x_train_std.npy")).to(cfg.device)
    # y_train_std = torch.from_numpy(np.load("y_train_std.npy")).to(cfg.device)
    # ========================
    concentration_predictor_std = ConcentrationPredictor(
        u0=u0.clone(), cfg=cfg, ret_inv_funs=[ret_inv_std_model, None]
    )
    concentration_predictor.zero_grad()
    concentration_predictor_std.zero_grad()
    print("Starting training for std model")
    concentration_predictor_std.run_training(t=x_train_std, u_full_train=y_train_std)

    # Evaluation
    def eval_networks(x, as_numpy: bool = False) -> dict[str, Any]:
        with torch.no_grad():
            d = {
                "mean": ret_inv_mean_models[0](x),
                "up": ret_inv_std_model(x),
                "down": -ret_inv_std_model(x),
            }
        if as_numpy:
            d = {k: v.numpy() for k, v in d.items()}
        return d

    c_up, c_down = tp3.compute_boundary_factors(
        y_train=y_train.numpy(),
        network_preds=eval_networks(x_train, as_numpy=True),
        quantile=0.95,
        verbose=1,
    )

    pred_train = eval_networks(x_train)
    pred_valid = eval_networks(x_valid)

    PICP_train, MPIW_train = tp3.caps_calculation(
        pred_train, c_up, c_down, y_train.numpy()
    )
    PICP_valid, MPIW_valid = tp3.caps_calculation(
        pred_valid, c_up, c_down, y_valid.numpy()
    )

    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, ".")
    y_U_PI_array_train = (
        (pred_train["mean"] + c_up * pred_train["up"]).numpy().flatten()
    )
    y_L_PI_array_train = (
        (pred_train["mean"] - c_down * pred_train["down"]).numpy().flatten()
    )
    y_mean = pred_train["mean"].numpy().flatten()
    sort_indices = np.argsort(x_train.flatten())
    ax.plot(x_train.flatten()[sort_indices], y_mean[sort_indices], "-")
    ax.plot(x_train.flatten()[sort_indices], y_U_PI_array_train[sort_indices], "-")
    ax.plot(x_train.flatten()[sort_indices], y_L_PI_array_train[sort_indices], "-")
    plt.show()


if __name__ == "__main__":
    main()
