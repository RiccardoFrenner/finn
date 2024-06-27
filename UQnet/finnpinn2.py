import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lib import EarlyStopper, Flux_Kernels, Flux_Kernels_No_Ret, load_data
from scipy.optimize import bisect

clear_dirs = False

# MODEL NAME & SETTING
model_name = "syn_freundlich_01"
main_path = Path().cwd()
model_path = main_path / model_name
if clear_dirs and model_path.exists():
    shutil.rmtree(model_path)
model_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Params:
    model_name = model_name
    main_path = main_path
    model_path = model_path

    # NETWORK HYPER-PARAMETERS
    flux_layers = 3  # number of hidden layers for the NN in the flux kernels
    state_layers = 3  # number of hidden layers for the NN in the state kernels
    flux_nodes = 15  # number of hidden nodes per layer for the NN in the flux kernels
    state_nodes = 15  # number of hidden nodes per layer for the NN in the flux kernels
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
    num_vars = 2

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

    # Inputs for Flux Kernels
    ## Set number of hidden layers and hidden nodes
    num_layers_flux = [flux_layers, flux_layers]
    num_nodes_flux = [flux_nodes, flux_nodes]
    ## Set numerical stencil to be learnable or not
    learn_stencil = [False, False]
    ## Effective diffusion coefficient for each variable
    # D_eff = [D / (dx**2), D * por / (rho_s/1000) / (dx**2)]
    D_eff = [D / (dx**2), 0.25]
    ## Set diffusion coefficient to be learnable or not
    learn_coeff = [False, True]
    ## Set if diffusion coefficient to be approximated as a function
    is_retardation_a_func = [True, False]
    ## Normalizer for functions that are approximated with a NN
    p_exp_flux = [0.0, 0.0]
    ## Set the variable index to be used when calculating the fluxes
    flux_calc_idx = [0, 0]
    ## Set the variable indices necessary to calculate the diffusion
    ## coefficient function
    flux_couple_idx = [0, 0]
    ## Set boundary condition types
    dirichlet_bool = [[True, False, False, False], [True, False, False, False]]
    neumann_bool = [[False, False, True, True], [False, False, True, True]]
    cauchy_bool = [[False, True, False, False], [False, True, False, False]]
    ## Set the Dirichlet and Neumann boundary values if necessary,
    ## otherwise set = 0
    dirichlet_val = [[solubility, 0.0, 0.0, 0.0], [solubility, 0.0, 0.0, 0.0]]
    neumann_val = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    ## Set multiplier for the Cauchy boundary condition if necessary
    ## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
    cauchy_mult = [dx, dx]

    # Inputs for State Kernels
    ## Set number of hidden layers and hidden nodes
    num_layers_state = [state_layers, state_layers]
    num_nodes_state = [state_nodes, state_nodes]
    ## Normalizer for the reaction functions that are approximated with a NN
    p_exp_state = [0.0, 0.0]
    ## Set the variable indices necessary to calculate the reaction function
    state_couple_idx = [0, 1]


params0 = Params()
params1 = Params()
params2 = Params()


class CL_dataLoader:
    def __init__(self, original_data_path=None, configs=None):
        if original_data_path:
            self.data_dir = original_data_path
        if configs:
            self.configs = configs

    def load_boston(self, Y_data="default"):
        """(1) Boston"""
        rawData_boston_np = np.loadtxt(
            os.path.join(self.data_dir, "boston-housing/boston_housing.txt")
        )
        X = rawData_boston_np[:, :-1]
        Y = rawData_boston_np[:, -1]
        return X, Y

    def load_single_dataset(self, name, Y_data="default"):
        if name == "boston":
            X, Y = self.load_boston(Y_data=Y_data)
        else:
            print("Loading artificial dataset")
            from pathlib import Path

            X = np.load(Path(self.data_dir).parent / "X_arti.npy").reshape(-1, 1)
            Y = np.load(Path(self.data_dir).parent / "Y_arti.npy")
        return X, Y

    def getNumInputsOutputs(self, inputsOutputs_np):
        if len(inputsOutputs_np.shape) == 1:
            numInputsOutputs = 1
        if len(inputsOutputs_np.shape) > 1:
            numInputsOutputs = inputsOutputs_np.shape[1]
        return numInputsOutputs


def caps_calculation(network_preds: dict[str, Any], c_up, c_down, Y, verbose=0):
    """Caps calculations for single quantile"""

    if verbose > 0:
        print("--- Start caps calculations for SINGLE quantile ---")
        print("**************** For Training data *****************")

    if len(Y.shape) == 2:
        Y = Y.flatten()

    bound_up = (network_preds["mean"] + c_up * network_preds["up"]).numpy().flatten()
    bound_down = (
        (network_preds["mean"] - c_down * network_preds["down"]).numpy().flatten()
    )

    y_U_cap = bound_up > Y  # y_U_cap
    y_L_cap = bound_down < Y  # y_L_cap

    y_all_cap = np.logical_or(y_U_cap, y_L_cap)  # y_all_cap
    PICP = np.count_nonzero(y_all_cap) / y_L_cap.shape[0]  # 0-1
    MPIW = np.mean(
        (network_preds["mean"] + c_up * network_preds["up"]).numpy().flatten()
        - (network_preds["mean"] - c_down * network_preds["down"]).numpy().flatten()
    )
    if verbose > 0:
        print(f"Num of train in y_U_cap: {np.count_nonzero(y_U_cap)}")
        print(f"Num of train in y_L_cap: {np.count_nonzero(y_L_cap)}")
        print(f"Num of train in y_all_cap: {np.count_nonzero(y_all_cap)}")
        print(f"np.sum results(train): {np.sum(y_all_cap)}")
        print(f"PICP: {PICP}")
        print(f"MPIW: {MPIW}")

    return (
        PICP,
        MPIW,
    )


def optimize_bound(
    *,
    mode: str,
    y_train: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    num_outliers: int,
    c0: float = 0.0,
    c1: float = 1e5,
    maxiter: int = 1000,
    verbose=0,
):
    def count_exceeding_upper_bound(c: float):
        bound = pred_mean + c * pred_std
        f = np.count_nonzero(y_train >= bound) - num_outliers
        return f

    def count_exceeding_lower_bound(c: float):
        bound = pred_mean - c * pred_std
        f = np.count_nonzero(y_train <= bound) - num_outliers
        return f

    objective_function = (
        count_exceeding_upper_bound if mode == "up" else count_exceeding_lower_bound
    )

    if verbose > 0:
        print(f"Initial bounds: [{c0}, {c1}]")

    try:
        optimal_c = bisect(objective_function, c0, c1, maxiter=maxiter)
        if verbose > 0:
            final_count = objective_function(optimal_c)
            print(f"Optimal c: {optimal_c}, Final count: {final_count}")
        return optimal_c
    except ValueError as e:
        if verbose > 0:
            print(f"Bisect method failed: {e}")
        raise e


def compute_boundary_factors(
    *, y_train: np.ndarray, network_preds: dict[str, Any], quantile: float, verbose=0
):
    n_train = y_train.shape[0]
    num_outlier = int(n_train * (1 - quantile) / 2)

    if verbose > 0:
        print(f"--- Start boundary optimizations for SINGLE quantile: {quantile}")
        print(f"--- Number of outlier based on the defined quantile: {num_outlier}")

    c_up, c_down = [
        optimize_bound(
            y_train=y_train,
            pred_mean=network_preds["mean"],
            pred_std=network_preds[mode],
            mode=mode,
            num_outliers=num_outlier,
        )
        for mode in ["up", "down"]
    ]

    if verbose > 0:
        print(f"--- c_up: {c_up}")
        print(f"--- c_down: {c_down}")

    return c_up, c_down


def create_PI_training_data(
    network_mean, X, Y
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Generate up and down training data"""
    with torch.no_grad():
        diff_train = Y - network_mean(X)
        dist = torch.sum(diff_train**2, dim=[1, 2, 3])
        threshold = np.quantile(dist, 0.5)
        up_idx = (dist > threshold).flatten()
        down_idx = (dist < threshold).flatten()

        X_up = X[up_idx.flatten()]
        Y_up = diff_train[up_idx]

        X_down = X[down_idx.flatten()]
        Y_down = -1.0 * diff_train[down_idx]

    return ((X_up, Y_up), (X_down, Y_down))


def train_network(
    model, optimizer, criterion, train_loader, val_loader, max_epochs: int
) -> None:
    """
    Train network
    Args:
        model (_type_): Concentration prediction model
    """
    early_stopper = EarlyStopper(patience=300, verbose=False)

    def closure():
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            u_ret = torch.linspace(0.0, 1.0, 100).view(-1, 1)
            ret_inv_pred = model.retardation_inv_scaled(u_ret)
            loss += torch.sum(
                torch.relu(ret_inv_pred[:-1] - ret_inv_pred[1:])
            )
            loss.backward()

        return loss

    for epoch in range(1, max_epochs + 1):
        # Training phase
        data, target = train_loader[0]
        loss_train = optimizer.step(closure)
        # print(f"Epoch: {epoch}, Loss: {loss_train}")

        # Validation phase
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss_valid = criterion(output, target)
                loss_val += loss_valid.item()
        loss_val = loss_val / len(val_loader)

        # if epoch % max(1, max_epochs // 100) == 0:
        print(f"{epoch=}, {loss_val=:.6f}, {loss_train=:.6f}")

        # Check early stopping condition
        early_stopper.update(loss_val, model)
        if early_stopper.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    # TODO: This does not actually look better in the plot?
    # model.load_state_dict(torch.load('checkpoint.pt'))


class CL_trainer:
    def __init__(
        self,
        configs,
        net_mean,
        net_up,
        net_down,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test=None,
        y_test=None,
    ):
        """Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance"""

        self.configs = configs

        self.networks = {
            "mean": net_mean,
            "up": net_up,
            "down": net_down,
        }
        self.optimizers = {
            # network_type: torch.optim.Adam(
            #     network.parameters(), lr=0.02, weight_decay=2e-2
            # )
            # for network_type, network in self.networks.items()
            network_type: torch.optim.LBFGS(network.parameters(), lr=0.01)
            for network_type, network in self.networks.items()
        }

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        train_network(
            model=self.networks["mean"],
            optimizer=self.optimizers["mean"],
            criterion=nn.MSELoss(),
            train_loader=[(self.x_train, self.y_train)],
            val_loader=[(self.x_valid, self.y_valid)],
            max_epochs=self.configs["Max_iter"],
        )

        print("--- Start PI training ---")
        data_train_up, data_train_down = create_PI_training_data(
            self.networks["mean"], X=self.x_train, Y=self.y_train
        )
        data_val_up, data_val_down = create_PI_training_data(
            self.networks["mean"], X=self.x_valid, Y=self.y_valid
        )

        train_network(
            model=self.networks["up"],
            optimizer=self.optimizers["up"],
            criterion=nn.MSELoss(),
            train_loader=[data_train_up],
            val_loader=[data_val_up],
            max_epochs=self.configs["Max_iter"],
        )
        # train_network(
        #     model=self.networks["down"],
        #     optimizer=self.optimizers["down"],
        #     criterion=nn.MSELoss(),
        #     train_loader=[data_train_down],
        #     val_loader=[data_val_down],
        #     max_epochs=self.configs["Max_iter"],
        # )

    def eval_networks(self, x, as_numpy: bool = False) -> dict[str, Any]:
        with torch.no_grad():
            d = {k: network(x) for k, network in self.networks.items()}
        if as_numpy:
            d = {k: v.numpy() for k, v in d.items()}
        return d


# TODO: Add regularization to optimizer step
# every dense layer had it: regularizers.l1_l2(l1=0.02, l2=0.02)
class UQ_Net_mean(nn.Module):
    def __init__(self, configs, num_inputs, num_outputs):
        super(UQ_Net_mean, self).__init__()
        self.configs = configs
        self.num_nodes_list = self.configs["num_neurons_mean"]

        self.inputLayer = nn.Linear(num_inputs, self.num_nodes_list[0])
        self.fcs = nn.ModuleList()
        for i in range(len(self.num_nodes_list) - 1):
            self.fcs.append(
                nn.Linear(self.num_nodes_list[i], self.num_nodes_list[i + 1])
            )
        self.outputLayer = nn.Linear(self.num_nodes_list[-1], num_outputs)

        # Initialize weights with a mean of 0.1 and stddev of 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.1, std=0.1)
                nn.init.zeros_(m.bias)

        self.u0 = torch.zeros(params1.num_vars, params1.Nx, 1)
        self.flux_kernel_diss = Flux_Kernels(
            self.u0[0].clone(), params1, 0, ret_inv_fun=self.retardation_inv_scaled
        )
        self.flux_kernel_tot = Flux_Kernels(
            self.u0[1].clone(), params1, 1, ret_inv_fun=None
        )

    def dudt_fun(self, t, u):
        flux_diss = self.flux_kernel_diss(u[[0]], u[[0]], t)
        flux_tot = self.flux_kernel_tot(u[[0]], u[[0]], t)
        return torch.stack([flux_diss, flux_tot])

    def retardation_inv_scaled(self, x):
        x = torch.relu(self.inputLayer(x))
        for i in range(len(self.fcs)):
            x = torch.relu(self.fcs[i](x))
        x = self.outputLayer(x)
        return x

    def forward(self, t):
        # return odeint(self.dudt_fun, self.u0, t, rtol=1e-5, atol=1e-6)
        # return torch.randn(size=(len(t), *self.u0.shape))
        y = self.dudt_fun(t[0], self.u0).unsqueeze(0)
        y = y.repeat(len(t), 1, 1, 1)
        return y


class UQ_Net_std(nn.Module):
    def __init__(self, configs, num_inputs, num_outputs, net=None, bias=None):
        super(UQ_Net_std, self).__init__()
        self.configs = configs
        if net == "up":
            self.num_nodes_list = self.configs["num_neurons_up"]
        elif net == "down":
            self.num_nodes_list = self.configs["num_neurons_down"]

        self.inputLayer = nn.Linear(num_inputs, self.num_nodes_list[0])
        self.fcs = nn.ModuleList()
        for i in range(len(self.num_nodes_list) - 1):
            self.fcs.append(
                nn.Linear(self.num_nodes_list[i], self.num_nodes_list[i + 1])
            )
        self.outputLayer = nn.Linear(self.num_nodes_list[-1], num_outputs)

        # Initialize weights with a mean of 0.1 and stddev of 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.1, std=0.1)
                nn.init.zeros_(m.bias)

        # Custom bias
        if bias is None:
            self.custom_bias = torch.nn.Parameter(torch.tensor([3.0]))
        else:
            self.custom_bias = torch.nn.Parameter(torch.tensor([bias]))

        # self.u0 = torch.zeros(params2.num_vars, params2.Nx, 1)
        self.c_diss_0 = torch.zeros(params2.Nx, 1)
        self.c_tot_0 = torch.zeros(params2.Nx, 1)
        self.flux_kernel_diss = Flux_Kernels(
            self.c_diss_0.clone(),
            params2,
            0,
            ret_inv_fun=self.retardation_inv_scaled,
            # self.u0[0].clone(), params2, 0, ret_inv_fun=self.retardation_inv_scaled
        )
        self.flux_kernel_tot = Flux_Kernels_No_Ret()
        # self.flux_kernel_tot = Flux_Kernels(self.u0[1].clone(), params2, 1, ret_inv_fun=None)

    def dudt_fun(self, t, u):
        flux_diss = self.flux_kernel_diss(u[0].clone(), u[0].clone(), t)
        flux_tot = self.flux_kernel_tot(u[0].clone(), u[0].clone(), t)
        # return torch.stack([flux_diss, flux_diss])
        return torch.stack([flux_diss, flux_tot])

    def retardation_inv_scaled(self, x):
        x = torch.relu(self.inputLayer(x))
        for i in range(len(self.fcs)):
            x = torch.relu(self.fcs[i](x))
        x = self.outputLayer(x)
        x = x + self.custom_bias
        x = torch.sqrt(torch.square(x) + 0.2)
        return x

    def forward(self, t):
        # return odeint(self.dudt_fun, self.u0, t, rtol=1e-5, atol=1e-6)
        # return torch.randn(size=(len(t), *self.u0.shape))
        y = self.dudt_fun(t[0], torch.stack((self.c_diss_0, self.c_tot_0))).unsqueeze(0)
        y = y.repeat(len(t), 1, 1, 1)
        return y


def main():
    matplotlib.use("TkAgg")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="artificial",
        help="example data names: boston, artificial",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()

    cfg = Params()

    ##########################################################
    ################## Data Loading Section ##################
    ##########################################################
    X = np.linspace(0.0, cfg.T, cfg.Nt)
    Y = load_data("freundlich").float().numpy()
    # print(X.dtype)
    # print(Y.dtype)
    # exit()
    test_split_index = 51
    xTrain = torch.from_numpy(X[:test_split_index].copy())
    yTrain = torch.from_numpy(Y[:test_split_index].copy())
    xValid = torch.from_numpy(X[test_split_index:].copy())
    yValid = torch.from_numpy(Y[test_split_index:].copy())

    #########################################################
    ############## End of Data Loading Section ##############
    #########################################################

    num_inputs = 1
    num_outputs = 1

    configs = {}
    ### Some other general input info
    configs["quantile"] = (
        args.quantile
    )  # # target percentile for optimization step# target percentile for optimization step,
    # 0.95 by default if not specified

    ######################################################################################
    # TODO: Re-Implement this
    # Multiple quantiles, comment out this line in order to run single quantile estimation
    # configs['quantile_list'] = np.arange(0.05, 1.00, 0.05) # 0.05-0.95
    ######################################################################################

    ### specify hypar-parameters for the training
    configs["seed"] = 10  # general random seed
    configs["num_neurons_mean"] = [50]  # hidden layer(s) for the 'MEAN' network
    configs["num_neurons_up"] = [50]  # hidden layer(s) for the 'UP' network
    configs["num_neurons_down"] = [50]  # hidden layer(s) for the 'DOWN' network
    configs["Max_iter"] = 1  # 5000,
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])

    """ Create network instances"""
    net_mean = UQ_Net_mean(configs, num_inputs, num_outputs)
    net_up = UQ_Net_std(configs, num_inputs, num_outputs, net="up")
    net_down = UQ_Net_std(configs, num_inputs, num_outputs, net="down")
    print("Set up network instances")

    # Initialize trainer and conduct training/optimizations
    trainer = CL_trainer(
        configs,
        net_mean,
        net_up,
        net_down,
        x_train=xTrain,
        y_train=yTrain,
        x_valid=xValid,
        y_valid=yValid,
    )
    print("Start training")
    trainer.train()  # training for 3 networks

    print("Start computing CUP, CDOWN")
    c_up, c_down = compute_boundary_factors(
        y_train=yTrain.numpy(),
        network_preds=trainer.eval_networks(xTrain, as_numpy=True),
        quantile=configs["quantile"],
        verbose=1,
    )

    pred_train = trainer.eval_networks(xTrain)
    pred_valid = trainer.eval_networks(xValid)

    PICP_train, MPIW_train = caps_calculation(pred_train, c_up, c_down, yTrain.numpy())
    PICP_valid, MPIW_valid = caps_calculation(pred_valid, c_up, c_down, yValid.numpy())

    fig, ax = plt.subplots()
    ax.plot(xTrain, yTrain, ".")
    y_U_PI_array_train = (
        (pred_train["mean"] + c_up * pred_train["up"]).numpy().flatten()
    )
    y_L_PI_array_train = (
        (pred_train["mean"] - c_down * pred_train["down"]).numpy().flatten()
    )
    y_mean = pred_train["mean"].numpy().flatten()
    sort_indices = np.argsort(xTrain.flatten())
    ax.plot(xTrain.flatten()[sort_indices], y_mean[sort_indices], "-")
    ax.plot(xTrain.flatten()[sort_indices], y_U_PI_array_train[sort_indices], "-")
    ax.plot(xTrain.flatten()[sort_indices], y_L_PI_array_train[sort_indices], "-")
    plt.show()


if __name__ == "__main__":
    main()
