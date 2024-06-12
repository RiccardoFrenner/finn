import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CL_plotter:
    def __init__(self):
        pass

    def plotTrainValidationLoss(
        self,
        train_loss,
        valid_loss,
        test_loss=None,
        trainPlotLabel=None,
        validPlotLabel=None,
        testPlotLabel=None,
        xlabel="",
        ylabel="",
        title="",
        gridOn=False,
        legendOn=True,
        xlim=None,
        ylim=None,
        saveFigPath=None,
    ):
        iter_arr = np.arange(len(train_loss))
        plt.plot(iter_arr, valid_loss, label=validPlotLabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if gridOn is True:
            plt.grid()
        if legendOn is True:
            plt.legend()
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if saveFigPath is not None:
            plt.savefig(saveFigPath)
            plt.clf()
        if saveFigPath is None:
            plt.show()


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


@dataclass
class TrainingStatistics:
    loss_train: list[float] = field(init=False, default_factory=list)
    loss_valid: list[float] = field(init=False, default_factory=list)
    loss_test: list[float] = field(init=False, default_factory=list)
    loss_iter: list[int] = field(init=False, default_factory=list)


@dataclass
class TrainingData:
    x_train: Any
    y_train: Any
    x_valid: Any
    y_valid: Any
    x_test: Any
    y_test: Any


class CL_trainer:
    def __init__(
        self,
        configs,
        net_mean,
        net_up,
        net_down,
        xTrain,
        yTrain,
        xValid=None,
        yValid=None,
        xTest=None,
        yTest=None,
    ):
        """Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance"""

        self.configs = configs

        self.networks = {
            "mean": net_mean,
            "up": net_up,
            "down": net_down,
        }

        self.training_data = TrainingData(xTrain, yTrain, xValid, yValid, xTest, yTest)

        self.trainSteps = CL_UQ_Net_train_steps(self.networks)
        self.plotter = CL_plotter()
        self.training_statistics = {
            k: TrainingStatistics() for k in self.networks.keys()
        }
        self.saveFigPrefix = self.configs["data_name"]  # prefix for the saved plots

    def main_train_step(
        self,
        network_type: Literal["mean", "up", "down"],
        training_data: TrainingData,
        max_iter: int,
    ):
        """Training for the different networks"""

        network = self.trainSteps.networks[network_type]
        optimizer = self.trainSteps.optimizers[network_type]

        print(f"--- Start training for {network_type} ---")
        stop_training = False
        early_stop_wait = 0
        min_delta = 0

        stopped_baseline = None
        if stopped_baseline is not None:
            best_loss = stopped_baseline
        else:
            best_loss = np.Inf
        best_weights = None

        iterations = self.training_statistics[network_type].loss_iter
        train_losses = self.training_statistics[network_type].loss_train
        valid_losses = self.training_statistics[network_type].loss_valid
        for i in range(max_iter):
            current_train_loss, current_valid_loss = self.trainSteps.train_step(
                network,
                optimizer,
                training_data.x_train,
                training_data.y_train,
                training_data.x_valid,
                training_data.y_valid,
            )

            if math.isnan(current_train_loss) or math.isnan(current_valid_loss):
                print(
                    "--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters..."
                )
                break

            if i % 100 == 0:
                print(
                    "Epoch: {}, train loss: {}, valid loss: {}".format(
                        i, current_train_loss, current_valid_loss
                    )
                )

            train_losses.append(current_train_loss)
            valid_losses.append(current_valid_loss)

            if (
                self.configs["early_stop"]
                and i >= self.configs["early_stop_start_iter"]
            ):
                if np.less(current_valid_loss - min_delta, best_loss):
                    best_loss = current_valid_loss
                    early_stop_wait = 0
                else:
                    early_stop_wait += 1
                    # print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
                    if early_stop_wait >= self.configs["wait_patience"]:
                        stop_training = True
            iterations.append(i)
            if stop_training:
                print(
                    "--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}".format(
                        i + 1, current_train_loss, current_valid_loss
                    )
                )
                break

        if self.configs["plot_loss_history"]:
            self.plotter.plotTrainValidationLoss(
                train_losses,
                valid_losses,
                trainPlotLabel="training loss",
                validPlotLabel="valid loss",
                xlabel="iterations",
                ylabel="Loss",
                title="("
                + self.saveFigPrefix
                + f")Train/valid (and test) loss for {network_type} values",
                gridOn=True,
                legendOn=True,
                saveFigPath=self.configs["plot_loss_history_path"]
                + self.saveFigPrefix
                + f"_{network_type}_loss_seed_"
                + str(self.configs["split_seed"])
                + "_"
                + str(self.configs["seed"])
                + ".png",
            )

        if self.configs["save_loss_history"]:
            loss_dict = {
                "iter": iterations,
                "train_loss": train_losses,
                "valid_loss": valid_losses,
            }

            df_loss = pd.DataFrame(loss_dict)
            df_loss.to_csv(
                self.configs["save_loss_history_path"]
                + self.configs["data_name"]
                + f"_{network_type}_loss_seed_"
                + str(self.configs["seed"])
                + ".csv"
            )

    def train(self):
        results_path = (
            "./Results_PI3NN/" + self.configs["data_name"] + "_PI3NN_results.txt"
        )
        with open(results_path, "a") as fwrite:
            fwrite.write(
                "EXP "
                + "random_seed "
                + "PICP_test "
                + "MPIW_test "
                + "RMSE "
                + "R2"
                + "\n"
            )

        self.main_train_step(
            network_type="mean",
            training_data=self.training_data,
            max_iter=self.configs["Max_iter"],
        )

        # IMPORTANT: has to be created after mean network finished training
        training_data_up, training_data_down = self.createUpDownTrainingData()
        self.main_train_step(
            network_type="up",
            training_data=training_data_up,
            max_iter=self.configs["Max_iter"],
        )
        self.main_train_step(
            network_type="down",
            training_data=training_data_down,
            max_iter=self.configs["Max_iter"],
        )

    def createUpDownTrainingData(self) -> tuple[TrainingData, TrainingData]:
        """Generate up and down training/validation data"""
        xTrain = self.training_data.x_train
        xValid = self.training_data.x_valid
        yTrain = self.training_data.y_train
        yValid = self.training_data.y_valid
        with torch.no_grad():  # No need to track gradients for this
            diff_train = yTrain.reshape(yTrain.shape[0], -1) - self.trainSteps.networks[
                "mean"
            ](xTrain)
            up_idx = diff_train > 0
            down_idx = diff_train < 0

            xTrain_up = xTrain[up_idx.flatten()]
            yTrain_up = diff_train[up_idx].unsqueeze(
                1
            )  # Unsqueeze for single-element dim

            xTrain_down = xTrain[down_idx.flatten()]
            yTrain_down = -1.0 * diff_train[down_idx].unsqueeze(1)

            diff_valid = yValid.reshape(yValid.shape[0], -1) - self.trainSteps.networks[
                "mean"
            ](xValid)
            up_idx = diff_valid > 0
            down_idx = diff_valid < 0

            xValid_up = xValid[up_idx.flatten()]
            yValid_up = diff_valid[up_idx].unsqueeze(1)

            xValid_down = xValid[down_idx.flatten()]
            yValid_down = -1.0 * diff_valid[down_idx].unsqueeze(1)

        return (
            TrainingData(xTrain_up, yTrain_up, xValid_up, yValid_up, None, None),
            TrainingData(
                xTrain_down, yTrain_down, xValid_down, yValid_down, None, None
            ),
        )

    def boundaryOptimization(self, quantile: float, verbose=0):
        with torch.no_grad():
            all_output = self.eval_networks(self.training_data.x_train)
        output = all_output["mean"]
        output_up = all_output["up"]
        output_down = all_output["down"]

        Ntrain = self.training_data.x_train.shape[0]
        num_outlier = int(Ntrain * (1 - quantile) / 2)

        if verbose > 0:
            print(
                "--- Start boundary optimizations for SINGLE quantile: {}".format(
                    quantile
                )
            )
            print(
                "--- Number of outlier based on the defined quantile: {}".format(
                    num_outlier
                )
            )

        boundaryOptimizer = CL_boundary_optimizer(
            self.training_data.y_train,
            output,
            output_up,
            output_down,
            num_outlier=num_outlier,
            c_up0_ini=0.0,
            c_up1_ini=100000.0,
            c_down0_ini=0.0,
            c_down1_ini=100000.0,
            max_iter=1000,
        )

        c_up = boundaryOptimizer.optimize_up(verbose=0)
        c_down = boundaryOptimizer.optimize_down(verbose=0)

        if verbose > 0:
            print("--- c_up: {}".format(c_up))
            print("--- c_down: {}".format(c_down))

        return c_up, c_down

    def eval_networks(self, x) -> dict[str, Any]:
        with torch.no_grad():
            d = {k: network(x) for k, network in self.networks.items()}
        return d


@dataclass
class PredictionIntervalComputer:
    pred_train: dict[str, Any]
    pred_valid: dict[str, Any]
    pred_test: dict[str, Any]

    @property
    def train_output(self):
        return self.pred_train["mean"]

    @property
    def train_output_up(self):
        return self.pred_train["up"]

    @property
    def train_output_down(self):
        return self.pred_train["down"]

    @property
    def valid_output(self):
        return self.pred_valid["mean"]

    @property
    def valid_output_up(self):
        return self.pred_valid["up"]

    @property
    def valid_output_down(self):
        return self.pred_valid["down"]

    @property
    def test_output(self):
        return self.pred_test["mean"]

    @property
    def test_output_up(self):
        return self.pred_test["up"]

    @property
    def test_output_down(self):
        return self.pred_test["down"]

    def capsCalculation(self, c_up, c_down, yTrain, yValid, yTest, verbose=0):
        ### caps calculations for single quantile
        if verbose > 0:
            print("--- Start caps calculations for SINGLE quantile ---")
            print("**************** For Training data *****************")
        if len(yTrain.shape) == 2:
            yTrain = yTrain.flatten()
        y_U_cap_train = (
            self.train_output + c_up * self.train_output_up
        ).numpy().flatten() > yTrain
        y_L_cap_train = (
            self.train_output - c_down * self.train_output_down
        ).numpy().flatten() < yTrain

        y_all_cap_train = y_U_cap_train * y_L_cap_train  # logic_or
        self.PICP_train = np.sum(y_all_cap_train) / y_L_cap_train.shape[0]  # 0-1
        self.MPIW_train = np.mean(
            (self.train_output + c_up * self.train_output_up).numpy().flatten()
            - (self.train_output - c_down * self.train_output_down).numpy().flatten()
        )
        if verbose > 0:
            print(
                "Num of train in y_U_cap_train: {}".format(
                    np.count_nonzero(y_U_cap_train)
                )
            )
            print(
                "Num of train in y_L_cap_train: {}".format(
                    np.count_nonzero(y_L_cap_train)
                )
            )
            print(
                "Num of train in y_all_cap_train: {}".format(
                    np.count_nonzero(y_all_cap_train)
                )
            )
            print("np.sum results(train): {}".format(np.sum(y_all_cap_train)))
            print("PICP_train: {}".format(self.PICP_train))
            print("MPIW_train: {}".format(self.MPIW_train))

        ### for validation data
        if verbose > 0:
            print("**************** For Validation data *****************")
        if len(yValid.shape) == 2:
            yValid = yValid.flatten()
        y_U_cap_valid = (
            self.valid_output + c_up * self.valid_output_up
        ).numpy().flatten() > yValid
        y_L_cap_valid = (
            self.valid_output - c_down * self.valid_output_down
        ).numpy().flatten() < yValid
        y_all_cap_valid = y_U_cap_valid * y_L_cap_valid  # logic_or
        self.PICP_valid = np.sum(y_all_cap_valid) / y_L_cap_valid.shape[0]  # 0-1
        self.MPIW_valid = np.mean(
            (self.valid_output + c_up * self.valid_output_up).numpy().flatten()
            - (self.valid_output - c_down * self.valid_output_down).numpy().flatten()
        )
        if verbose > 0:
            print(
                "Num of valid in y_U_cap_valid: {}".format(
                    np.count_nonzero(y_U_cap_valid)
                )
            )
            print(
                "Num of valid in y_L_cap_valid: {}".format(
                    np.count_nonzero(y_L_cap_valid)
                )
            )
            print(
                "Num of valid in y_all_cap_valid: {}".format(
                    np.count_nonzero(y_all_cap_valid)
                )
            )
            print("np.sum results(valid): {}".format(np.sum(y_all_cap_valid)))
            print("PICP_valid: {}".format(self.PICP_valid))
            print("MPIW_valid: {}".format(self.MPIW_valid))

        if verbose > 0:
            print("**************** For Testing data *****************")
        if len(yTest.shape) == 2:
            yTest = yTest.flatten()
        y_U_cap_test = (
            self.test_output + c_up * self.test_output_up
        ).numpy().flatten() > yTest
        y_L_cap_test = (
            self.test_output - c_down * self.test_output_down
        ).numpy().flatten() < yTest
        y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
        self.PICP_test = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
        self.MPIW_test = np.mean(
            (self.test_output + c_up * self.test_output_up).numpy().flatten()
            - (self.test_output - c_down * self.test_output_down).numpy().flatten()
        )
        # print('y_U_cap: {}'.format(y_U_cap))
        # print('y_L_cap: {}'.format(y_L_cap))
        if verbose > 0:
            print("Num of true in y_U_cap: {}".format(np.count_nonzero(y_U_cap_test)))
            print("Num of true in y_L_cap: {}".format(np.count_nonzero(y_L_cap_test)))
            print(
                "Num of true in y_all_cap: {}".format(np.count_nonzero(y_all_cap_test))
            )
            print("np.sum results: {}".format(np.sum(y_all_cap_test)))
            print("PICP_test: {}".format(self.PICP_test))
            print("MPIW_test: {}".format(self.MPIW_test))

        # print(y_all_cap)
        # print(np.sum(y_all_cap_test))

        self.MSE_test = np.mean(np.square(self.test_output.numpy().flatten() - yTest))
        self.RMSE_test = np.sqrt(self.MSE_test)
        self.R2_test = r2_score(yTest, self.test_output.numpy().flatten())
        if verbose > 0:
            print("Test MSE: {}".format(self.MSE_test))
            print("Test RMSE: {}".format(self.RMSE_test))
            print("Test R2: {}".format(self.R2_test))

        return (
            self.PICP_train,
            self.PICP_valid,
            self.PICP_test,
            self.MPIW_train,
            self.MPIW_valid,
            self.MPIW_test,
        )

    def saveResultsToTxt(
        self, configs: dict[str, Any], PICP_test, MPIW_test, RMSE_test, R2_test
    ):
        """Save results to txt file"""
        results_path = "./Results_PI3NN/" + configs["data_name"] + "_PI3NN_results.txt"
        with open(results_path, "a") as fwrite:
            fwrite.write(
                str(configs["experiment_id"])
                + " "
                + str(configs["seed"])
                + " "
                + str(round(PICP_test, 3))
                + " "
                + str(round(MPIW_test, 3))
                + " "
                + str(round(RMSE_test, 3))
                + " "
                + str(round(R2_test, 3))
                + "\n"
            )

    def save_PI(self, c_up: float, c_down: float, bias=0.0):
        y_U_PI_array_train = (
            (self.train_output + c_up * self.train_output_up).numpy().flatten()
        )
        y_L_PI_array_train = (
            (self.train_output - c_down * self.train_output_down).numpy().flatten()
        )

        y_U_PI_array_test = (
            (self.test_output + c_up * self.test_output_up).numpy().flatten()
        )
        y_L_PI_array_test = (
            (self.test_output - c_down * self.test_output_down).numpy().flatten()
        )

        path = "./Results_PI3NN/npy/"
        train_bounds = np.vstack((y_U_PI_array_train, y_L_PI_array_train))
        test_bounds = np.vstack((y_U_PI_array_test, y_L_PI_array_test))
        np.save(path + "train_bounds" + "_bias_" + str(bias) + ".npy", train_bounds)
        np.save(path + "test_bounds" + "_bias_" + str(bias) + ".npy", test_bounds)
        np.save(path + "yTrain" + "_bias_" + str(bias) + ".npy", yTrain)
        np.save(path + "yTest" + "_bias_" + str(bias) + ".npy", self.yTest)
        print("--- results npy saved")


def load_and_plot_PI(bias=0.0):
    path = "./Results_PI3NN/npy/"
    path_fig = "./Results_PI3NN/plots/"

    train_bounds = np.load(path + "train_bounds" + "_bias_" + str(bias) + ".npy")
    test_bounds = np.load(path + "test_bounds" + "_bias_" + str(bias) + ".npy")
    yTrain = np.load(path + "yTrain" + "_bias_" + str(bias) + ".npy")
    yTest = np.load(path + "yTest" + "_bias_" + str(bias) + ".npy")

    fig, ax = plt.subplots(1)
    x_train_arr = np.arange(len(train_bounds[0]))
    x_test_arr = np.arange(
        len(train_bounds[0]), len(train_bounds[0]) + len(test_bounds[0])
    )

    ax.scatter(x_train_arr, train_bounds[0], s=0.01, label="Train UP")
    ax.scatter(x_train_arr, train_bounds[1], s=0.01, label="Train DOWN")

    ax.scatter(x_test_arr, test_bounds[0], s=0.01, label="Test UP")
    ax.scatter(x_test_arr, test_bounds[1], s=0.01, label="Test DOWN")

    ax.scatter(x_train_arr, yTrain, s=0.01, label="yTrain")
    ax.scatter(x_test_arr, yTest, s=0.01, label="yTest")

    plt.title("PI3NN bounds prediction for flight delay data, bias:{}".format(bias))
    plt.grid()
    plt.legend()
    plt.savefig(path_fig + "bounds" + "_bias_" + str(bias) + ".png", dpi=300)
    # plt.show()


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


class CL_UQ_Net_train_steps:
    def __init__(
        self,
        networks: dict[str, Any],
        lrs={"mean": 0.01, "up": 0.01, "down": 0.01},
    ):
        self.criterion_mean = nn.MSELoss()
        self.criterion_std = nn.MSELoss()

        self.networks = networks

        # TODO: LR schedule

        # weight decay is (very similar) to L2 regularization
        self.optimizers = {
            k: torch.optim.Adam(
                self.networks[k].parameters(), lr=lrs[k], weight_decay=1e-5
            )
            for k in self.networks.keys()
        }

    def train_step(self, network, optimizer, xTrain, yTrain, xValid, yValid):
        network.train()  # Set model to training mode
        yPred = network(xTrain)
        train_loss = self.criterion_mean(yTrain, yPred)

        # Add regularization losses (example using L2)
        l2_loss = 0
        for param in network.parameters():
            l2_loss += torch.sum(torch.square(param))
        train_loss += 0.01 * l2_loss  # Add regularization loss to training loss

        optimizer.zero_grad()  # Clear gradients
        train_loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights

        # Validation
        network.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            valid_predictions = network(xValid)
            valid_loss = self.criterion_mean(yValid, valid_predictions)

        return train_loss.item(), valid_loss.item()


class CL_boundary_optimizer:
    def __init__(
        self,
        yTrain,
        output_mean,
        output_up,
        output_down,
        num_outlier=None,
        c_up0_ini=None,
        c_up1_ini=None,
        c_down0_ini=None,
        c_down1_ini=None,
        max_iter=None,
    ):
        self.yTrain = yTrain.numpy().flatten()

        self.output_mean = output_mean
        self.output_up = output_up
        self.output_down = output_down
        if num_outlier is not None:
            self.num_outlier = num_outlier
        self.c_up0_ini = c_up0_ini
        self.c_up1_ini = c_up1_ini
        self.c_down0_ini = c_down0_ini
        self.c_down1_ini = c_down1_ini
        self.max_iter = max_iter

    def optimize_up(self, outliers=None, verbose=0):
        if outliers is not None:
            self.num_outlier = outliers
        c_up0 = self.c_up0_ini
        c_up1 = self.c_up1_ini
        f0 = (
            np.count_nonzero(
                self.yTrain
                >= self.output_mean.numpy().flatten()
                + c_up0 * self.output_up.numpy().flatten()
            )
            - self.num_outlier
        )
        f1 = (
            np.count_nonzero(
                self.yTrain
                >= self.output_mean.numpy().flatten()
                + c_up1 * self.output_up.numpy().flatten()
            )
            - self.num_outlier
        )

        iter = 0
        while iter <= self.max_iter and f0 != 0 and f1 != 0:
            c_up2 = (c_up0 + c_up1) / 2.0
            f2 = (
                np.count_nonzero(
                    self.yTrain
                    >= self.output_mean.numpy().flatten()
                    + c_up2 * self.output_up.numpy().flatten()
                )
                - self.num_outlier
            )
            if f2 == 0:
                break
            elif f2 > 0:
                c_up0 = c_up2
                f0 = f2
            else:
                c_up1 = c_up2
                f1 = f2
            iter += 1
            if verbose > 1:
                print("{}, f0: {}, f1: {}, f2: {}".format(iter, f0, f1, f2))
                print("c_up0: {}, c_up1: {}, c_up2: {}".format(c_up0, c_up1, c_up2))
        if verbose > 0:
            print("f0 : {}".format(f0))
            print("f1 : {}".format(f1))

        c_up = c_up2
        return c_up

    def optimize_down(self, outliers=None, verbose=0):
        if outliers is not None:
            self.num_outlier = outliers
        c_down0 = self.c_down0_ini
        c_down1 = self.c_down1_ini
        f0 = (
            np.count_nonzero(
                self.yTrain
                <= self.output_mean.numpy().flatten()
                - c_down0 * self.output_down.numpy().flatten()
            )
            - self.num_outlier
        )
        f1 = (
            np.count_nonzero(
                self.yTrain
                <= self.output_mean.numpy().flatten()
                - c_down1 * self.output_down.numpy().flatten()
            )
            - self.num_outlier
        )

        iter = 0
        while iter <= self.max_iter and f0 != 0 and f1 != 0:
            c_down2 = (c_down0 + c_down1) / 2.0
            f2 = (
                np.count_nonzero(
                    self.yTrain
                    <= self.output_mean.numpy().flatten()
                    - c_down2 * self.output_down.numpy().flatten()
                )
                - self.num_outlier
            )
            if f2 == 0:
                break
            elif f2 > 0:
                c_down0 = c_down2
                f0 = f2
            else:
                c_down1 = c_down2
                f1 = f2
            iter += 1
            if verbose > 1:
                print("{}, f0: {}, f1: {}, f2: {}".format(iter, f0, f1, f2))
                print(
                    "c_down0: {}, c_down1: {}, c_down2: {}".format(
                        c_down0, c_down1, c_down2
                    )
                )
        if verbose > 0:
            print("f0 : {}".format(f0))
            print("f1 : {}".format(f1))

        c_down = c_down2
        return c_down


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
    configs["data_name"] = "bostonHousing"
    configs["quantile"] = (
        args.quantile
    )  # # target percentile for optimization step# target percentile for optimization step,
    # 0.95 by default if not specified
    configs["split_seed"] = "WhatIsThis?"
    configs["experiment_id"] = 1
    configs["verbose"] = 1
    configs["save_loss_history"] = True
    configs["save_loss_history_path"] = "./Results_PI3NN/loss_history/"
    configs["plot_loss_history"] = True
    configs["plot_loss_history_path"] = "./Results_PI3NN/loss_curves/"

    ######################################################################################
    # TODO: Re-Implement this
    # Multiple quantiles, comment out this line in order to run single quantile estimation
    # configs['quantile_list'] = np.arange(0.05, 1.00, 0.05) # 0.05-0.95
    ######################################################################################

    print("--- Running on manual mode.")
    ### specify hypar-parameters for the training
    configs["seed"] = 10  # general random seed
    configs["num_neurons_mean"] = [50]  # hidden layer(s) for the 'MEAN' network
    configs["num_neurons_up"] = [50]  # hidden layer(s) for the 'UP' network
    configs["num_neurons_down"] = [50]  # hidden layer(s) for the 'DOWN' network
    configs["Max_iter"] = 5000  # 5000,
    configs["lr"] = [0.02, 0.02, 0.02]  # 0.02         # learning rate
    configs["optimizers"] = ["Adam", "Adam", "Adam"]  # ['SGD', 'SGD', 'SGD'],
    configs["exponential_decay"] = True
    configs["decay_steps"] = 3000  # 3000  # 10
    configs["decay_rate"] = 0.9  # 0.6
    configs["early_stop"] = True
    configs["early_stop_start_iter"] = 100  # 60
    configs["wait_patience"] = 300
    print("--- Dataset: {}".format(configs["data_name"]))
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])

    """ Create network instances"""
    net_mean = UQ_Net_mean(configs, num_inputs, num_outputs)
    net_up = UQ_Net_std(configs, num_inputs, num_outputs, net="up")
    net_down = UQ_Net_std(configs, num_inputs, num_outputs, net="down")

    # ''' Initialize trainer and conduct training/optimizations '''
    trainer = CL_trainer(
        configs,
        net_mean,
        net_up,
        net_down,
        xTrain,
        yTrain,
        xValid=xValid,
        yValid=yValid,
        xTest=xTest,
        yTest=yTest,
    )
    trainer.train()  # training for 3 networks
    c_up, c_down = trainer.boundaryOptimization(configs["quantile"], verbose=1)

    pred_train = trainer.eval_networks(xTrain)
    pred_valid = trainer.eval_networks(xValid)
    pred_test = trainer.eval_networks(xTest)

    pic = PredictionIntervalComputer(pred_train, pred_valid, pred_test)
    pic.capsCalculation(
        c_up, c_down, yTrain.numpy(), yValid.numpy(), yTest.numpy(), verbose=1
    )
    # pic.saveResultsToTxt()

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
