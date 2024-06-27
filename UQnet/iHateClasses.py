import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchpinn as tp3
from lib import AnalyticRetardation, create_mlp
from torchdiffeq import odeint

print(tp3)


class Params:
    # -*- coding: utf-8 -*-

    clear_dirs = False

    # MODEL NAME & SETTING
    model_name = "syn_freundlich_01"
    main_path = Path().cwd()
    model_path = main_path / model_name
    if clear_dirs and model_path.exists():
        shutil.rmtree(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    device_name = "cpu"  # Choose between "cpu" or "cuda"

    # NETWORK HYPER-PARAMETERS
    error_mult = 1  # multiplier for the squared error in the loss function calculation
    breakthrough_mult = 1  # multiplier for the breakthrough curve error in the loss function calculation
    profile_mult = 1  # multiplier for the concentration profile error in the loss function calculation
    phys_mult = 100  # multiplier for the physical regularization in the loss function calculation
    epochs = 100  # maximum epoch for training
    lbfgs_optim = True  # Use L-BFGS as optimizer, else use ADAM
    train_breakthrough = False  # Train using only breakthrough curve data
    linear = False  # Training data generated with the linear isotherm
    freundlich = True  # Training data generated with the freundlich isotherm
    langmuir = False  # Training data generated with the langmuir isotherm

    # SIMULATION-RELATED INPUTS

    # Soil Parameters
    D = 0.0005  # effective diffusion coefficient [m^2/day]
    por = 0.29  # porosity [-]
    rho_s = 2880  # bulk density [kg/m^3]
    Kf = 1.016 / rho_s  # freundlich's K [(m^3/kg)^nf]
    nf = 0.874  # freundlich exponent [-]
    smax = 1 / 1700  # sorption capacity [m^3/kg]
    Kl = 1  # half-concentration [kg/m^3]
    Kd = 0.429 / 1000  # organic carbon partitioning [m^3/kg]
    solubility = 1.0  # top boundary value [kg/m^3]

    # Simulation Domain
    X = 1.0  # length of sample [m]
    dx = 0.04  # length of discrete control volume [m]
    T = 10000  # simulation time [days]
    dt = 5  # time step [days]
    Nx = int(X / dx + 1)
    Nt = int(T / dt + 1)
    cauchy_val = dx

    device = torch.device("cpu")

    # Inputs for Flux Kernels
    D_eff = (D / (dx**2), 0.25)

    ## Normalizer for functions that are approximated with a NN
    p_exp_flux = torch.tensor([0.0])
    ## Set boundary condition types
    dirichlet_bool = [True, False, False, False]
    neumann_bool = [False, False, True, True]
    cauchy_bool = [False, True, False, False]
    ## Set the Dirichlet and Neumann boundary values if necessary,
    ## otherwise set = 0
    dirichlet_val = torch.tensor([solubility, 0.0, 0.0, 0.0])
    neumann_val = (torch.tensor([0.0, 0.0, 0.0, 0.0]),)
    ## Set multiplier for the Cauchy boundary condition if necessary
    ## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
    cauchy_mult = dx


params = Params()


def flux_kernel(
    c: torch.Tensor, dx: float, D: tuple[float, float], BC: np.ndarray, scaled_ret_inv
):
    """Rate of change of concentration with respect to time (dc/dt)."""
    stencil = torch.tensor([-1.0, 1.0])

    # Approximate 1/retardation_factor
    # ret = (self.func_nn(c.unsqueeze(-1)) * 10**self.p_exp)
    ret = scaled_ret_inv(c).squeeze(-1)

    ## Calculate fluxes at the left boundary of control volumes i
    # Calculate the flux at the left domain boundary
    left_bound_flux_c = (
        D[0] * ret[0] * (stencil[0] * c[0] + stencil[1] * BC[0, 0])
    ).unsqueeze(0)

    # Calculate the fluxes between control volumes i and their left neighbors
    left_neighbors_c = D[0] * ret[1:] * (stencil[0] * c[1:] + stencil[1] * c[:-1])

    # Concatenate the left fluxes
    left_flux_c = torch.cat((left_bound_flux_c, left_neighbors_c))

    ## Calculate fluxes at the right boundary of control volumes i
    # Calculate the Cauchy condition for the right domain boundary
    right_BC = D[0] * dx * (c[-2] - c[-1])

    # Calculate the flux at the right domain boundary
    right_bound_flux_c = (
        D[0] * ret[-1] * (stencil[0] * c[-1] + stencil[1] * right_BC)
    ).unsqueeze(0)

    # Calculate the fluxes between control volumes i and their right neighbors
    right_neighbors_c = D[0] * ret[:-1] * (stencil[0] * c[:-1] + stencil[1] * c[1:])

    # Concatenate the right fluxes
    right_flux_c = torch.cat((right_neighbors_c, right_bound_flux_c))

    # Integrate the fluxes at all boundaries of control volumes i
    flux = left_flux_c + right_flux_c

    return flux


class ConcentrationPredictor(nn.Module):
    def __init__(self, c0: torch.Tensor):
        super(ConcentrationPredictor, self).__init__()

        self.c0 = c0
        self.ret_inv = create_mlp([1, 15, 15, 15, 1], nn.Tanh(), nn.Sigmoid())
        self.p_exp = nn.Parameter(torch.tensor([1.0]))

        self.scaled_ret_inv = lambda x: self.ret_inv(x.unsqueeze(-1) * 10**self.p_exp)

        BC = np.array([[1.0, 1.0], [0.0, 0.0]])
        self.dudt_fun = lambda t, u: flux_kernel(
            u, params.dx, params.D_eff, BC, self.scaled_ret_inv
        )

    def forward(self, t):
        return odeint(self.dudt_fun, self.c0, t, rtol=1e-5, atol=1e-6)

    def run_training(self, t: torch.Tensor, u_full_train: torch.Tensor):
        out_dir = Path("data_out")
        out_dir.mkdir(exist_ok=True, parents=True)

        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1)
        optimizer.zero_grad()

        u_ret = torch.linspace(0.0, 1.0, 100)
        # TODO: Should not be here
        ret_linear = AnalyticRetardation.linear(
            u_ret, por=params.por, rho_s=params.rho_s, Kd=params.Kd
        )
        ret_freundlich = AnalyticRetardation.freundlich(
            u_ret,
            por=params.por,
            rho_s=params.rho_s,
            Kf=params.Kf,
            nf=params.nf,
        )
        ret_langmuir = AnalyticRetardation.langmuir(
            u_ret,
            por=params.por,
            rho_s=params.rho_s,
            smax=params.smax,
            Kl=params.Kl,
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
            loss = params.error_mult * torch.sum((u_full_train - ode_pred) ** 2)

            # Physical regularization: value of the retardation factor should decrease with increasing concentration
            ret_inv_pred = self.scaled_ret_inv(u_ret)
            loss += params.phys_mult * torch.sum(
                torch.relu(ret_inv_pred[:-1] - ret_inv_pred[1:])
            )  # TODO: mean instead of sum?

            loss.backward()

            return loss

        # Iterate until maximum epoch number is reached
        for epoch in range(1, params.epochs + 1):
            dt = time.time()
            loss = optimizer.step(closure)
            # loss = closure()
            dt = time.time() - dt

            print(
                f"Training: Epoch [{epoch + 1}/{params.epochs}], "
                f"Training Loss: {loss:.4f}, Runtime: {dt:.4f} secs"
            )

            ret_pred_path = params.model_path / f"retPred_{epoch}.npy"
            np.save(ret_pred_path, self.retardation(u_ret).detach().numpy())

    def retardation(self, u):
        return 1.0 / self.scaled_ret_inv(u) / 10**self.p_exp


class RetardationInverseStd(nn.Module):
    def __init__(self, layers: list[int]):
        super(RetardationInverseStd, self).__init__()

        # self.layers = create_mlp(layers, nn.ReLU(), nn.Identity())
        # TODO:
        self.layers = create_mlp(layers, nn.ReLU(), nn.ReLU())

    def forward(self, u):
        u = torch.sqrt(torch.square(u) + 0.2)
        return u


def create_PI_training_data(
    y_pred_mean: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generate std training data"""
    assert isinstance(y_pred_mean, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)

    # threshold = 30
    diff_train = Y - y_pred_mean
    # print(f"{diff_train.shape=}")
    dist = np.sum(diff_train**2, axis=[1, 2, 3])
    threshold = np.quantile(dist, 0.5)
    # print(f"{threshold=}")
    mask = dist < threshold
    # print(f"{mask.shape=}")
    # print(f"{X.shape=}")
    # print(f"{Y.shape=}")

    X_std = X[~mask]
    Y_std = diff_train[~mask]
    # print(f"{X_std.shape=}")

    return X_std, Y_std


def load_data(
    synth_data_type: Literal["linear", "freundlich", "langmuir"],
    is_testing: bool = False,
) -> np.ndarray:
    # Load training data with the original dimension of [Nx, Nt]. Reshape into [Nt, Nx]

    folder = Path(
        f"../../oldfinn/python/diffusion_sorption/synthetic_data/data_{synth_data_type}"
    ).resolve()

    c_diss = pd.read_csv(
        folder / f'c_diss{"_test" if is_testing else ""}.csv', sep="\t", header=None
    )
    c_diss = np.array(c_diss).transpose(1, 0)

    return c_diss


def main():
    cfg = params

    c0 = torch.zeros(params.Nx)
    concentration_predictor = ConcentrationPredictor(c0)

    # Load data
    c_analytical = torch.Tensor(load_data("freundlich"))
    # x = torch.linspace(0.0, params.X, params.Nx)
    t = torch.linspace(0.0, params.T, params.Nt)

    train_split_index = 51
    x_train = t[:train_split_index]
    y_train = c_analytical[:train_split_index]
    x_valid = t[train_split_index:]
    y_valid = c_analytical[train_split_index:]

    # Train the concentration predictor
    concentration_predictor.run_training(t=x_train, u_full_train=y_train)
    with torch.no_grad():
        y_pred_mean = concentration_predictor(x_train).numpy()

    ret_inv_std_model = RetardationInverseStd([1, 15, 15, 15, 1])

    # Train the standard deviation concentration model to get a trained ret_inv_std_model
    # ========================
    # This works:
    # x_train_std = torch.linspace(0, params.T, params.Nt)[:train_split_index]
    # y_train_std = y_train.clone()
    # ========================
    # But this does not work:
    # torch.cuda.empty_cache()
    x_train_std, y_train_std = create_PI_training_data(
        y_pred_mean,
        x_train.numpy(),
        y_train.numpy(),
    )
    x_train_std = torch.Tensor(x_train_std)
    y_train_std = torch.Tensor(y_train_std)
    # # save as npy files
    # np.save("x_train_std.npy", x_train_std_.numpy())
    # np.save("y_train_std.npy", y_train_std_.numpy())
    # x_train_std = torch.from_numpy(np.load("x_train_std.npy"))
    # y_train_std = torch.from_numpy(np.load("y_train_std.npy"))
    # ========================
    concentration_predictor_std = ConcentrationPredictor(c0.clone())
    print("Starting training for std model")
    concentration_predictor_std.run_training(t=x_train_std, u_full_train=y_train_std)

    # # Evaluation
    # def eval_networks(x, as_numpy: bool = False) -> dict[str, Any]:
    #     with torch.no_grad():
    #         d = {
    #             "mean": ret_inv_mean_models[0](x),
    #             "up": ret_inv_std_model(x),
    #             "down": -ret_inv_std_model(x),
    #         }
    #     if as_numpy:
    #         d = {k: v.numpy() for k, v in d.items()}
    #     return d

    # c_up, c_down = tp3.compute_boundary_factors(
    #     y_train=y_train.numpy(),
    #     network_preds=eval_networks(x_train, as_numpy=True),
    #     quantile=0.95,
    #     verbose=1,
    # )

    # pred_train = eval_networks(x_train)
    # pred_valid = eval_networks(x_valid)

    # PICP_train, MPIW_train = tp3.caps_calculation(
    #     pred_train, c_up, c_down, y_train.numpy()
    # )
    # PICP_valid, MPIW_valid = tp3.caps_calculation(
    #     pred_valid, c_up, c_down, y_valid.numpy()
    # )

    # fig, ax = plt.subplots()
    # ax.plot(x_train, y_train, ".")
    # y_U_PI_array_train = (
    #     (pred_train["mean"] + c_up * pred_train["up"]).numpy().flatten()
    # )
    # y_L_PI_array_train = (
    #     (pred_train["mean"] - c_down * pred_train["down"]).numpy().flatten()
    # )
    # y_mean = pred_train["mean"].numpy().flatten()
    # sort_indices = np.argsort(x_train.flatten())
    # ax.plot(x_train.flatten()[sort_indices], y_mean[sort_indices], "-")
    # ax.plot(x_train.flatten()[sort_indices], y_U_PI_array_train[sort_indices], "-")
    # ax.plot(x_train.flatten()[sort_indices], y_L_PI_array_train[sort_indices], "-")
    # plt.show()


if __name__ == "__main__":
    main()
