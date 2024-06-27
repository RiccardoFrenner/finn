import argparse
import os
import random
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import params
import torch
import torch.nn as nn
from lib import EarlyStopper, Flux_Kernels, load_data
from scipy.optimize import bisect
from torchdiffeq import odeint


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
    early_stopper = EarlyStopper(patience=300, verbose=False)

    def closure():
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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
        train_network(
            model=self.networks["down"],
            optimizer=self.optimizers["down"],
            criterion=nn.MSELoss(),
            train_loader=[data_train_down],
            val_loader=[data_val_down],
            max_epochs=self.configs["Max_iter"],
        )

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

    def forward(self, x):
        x = torch.relu(self.inputLayer(x))
        for i in range(len(self.fcs)):
            x = torch.relu(self.fcs[i](x))
        x = self.outputLayer(x)
        return x


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

    def forward(self, x):
        x = torch.relu(self.inputLayer(x))
        for i in range(len(self.fcs)):
            x = torch.relu(self.fcs[i](x))
        x = self.outputLayer(x)
        x = x + self.custom_bias
        x = torch.sqrt(torch.square(x) + 0.2)
        return x


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

    ##########################################################
    ################## Data Loading Section ##################
    ##########################################################
    data_dir = "./datasets/UCI_datasets/"
    dataLoader = CL_dataLoader(original_data_path=data_dir)
    X, Y = dataLoader.load_single_dataset(args.data)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    # random split
    xTrainValid, xTest, yTrainValid, yTest = train_test_split(
        X, Y, test_size=0.1, random_state=1, shuffle=True
    )
    ## Split the validation data
    xTrain, xValid, yTrain, yValid = train_test_split(
        xTrainValid, yTrainValid, test_size=0.1, random_state=1, shuffle=True
    )

    ### Data normalization
    scalar_x = StandardScaler()
    scalar_y = StandardScaler()

    xTrain = scalar_x.fit_transform(xTrain)
    xValid = scalar_x.fit_transform(xValid)
    xTest = scalar_x.transform(xTest)

    yTrain = scalar_y.fit_transform(yTrain)
    yValid = scalar_y.fit_transform(yValid)
    yTest = scalar_y.transform(yTest)

    ### To tensors
    xTrain = torch.Tensor(xTrain)
    xValid = torch.Tensor(xValid)
    xTest = torch.Tensor(xTest)

    yTrain = torch.Tensor(yTrain)
    yValid = torch.Tensor(yValid)
    yTest = torch.Tensor(yTest)

    print(xTrain.shape)
    print(xValid.shape)
    print(xTest.shape)
    print(yTrain.shape)
    print(yValid.shape)
    print(yTest.shape)
    # exit()

    #########################################################
    ############## End of Data Loading Section ##############
    #########################################################

    num_inputs = dataLoader.getNumInputsOutputs(xTrain)
    num_outputs = dataLoader.getNumInputsOutputs(yTrain)

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
    configs["Max_iter"] = 5000  # 5000,
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
