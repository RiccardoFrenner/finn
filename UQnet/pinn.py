import argparse
import math
import os
import random
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


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
        xlabel=None,
        ylabel=None,
        title=None,
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


class CL_trainer:
    def __init__(
        self,
        configs,
        net_mean,
        net_std_up,
        net_std_down,
        xTrain,
        yTrain,
        xValid=None,
        yValid=None,
        xTest=None,
        yTest=None,
    ):
        """Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance"""

        self.configs = configs
        self.net_mean = net_mean
        self.net_std_up = net_std_up
        self.net_std_down = net_std_down

        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xValid = xValid
        self.yValid = yValid
        self.xTest = xTest
        self.yTest = yTest

        self.trainSteps = CL_UQ_Net_train_steps(
            self.net_mean,
            self.net_std_up,
            self.net_std_down,
            optimizers=self.configs["optimizers"],  ## 'Adam', 'SGD'
            lr=self.configs["lr"],  ## order: mean, up, down
            exponential_decay=self.configs["exponential_decay"],
            decay_steps=self.configs["decay_steps"],
            decay_rate=self.configs["decay_rate"],
        )

        self.plotter = CL_plotter()

        self.train_loss_mean_list = []
        self.valid_loss_mean_list = []
        self.test_loss_mean_list = []
        self.iter_mean_list = []

        self.train_loss_up_list = []
        self.valid_loss_up_list = []
        self.test_loss_up_list = []
        self.iter_up_list = []

        self.train_loss_down_list = []
        self.valid_loss_down_list = []
        self.test_loss_down_list = []
        self.iter_down_list = []

        self.saveFigPrefix = self.configs["data_name"]  # prefix for the saved plots

    def main_train_step(
        self,
        network_type: Literal["mean", "up", "down"],
        network,
        train_step_fun,
        train_loss,
        valid_loss,
        test_loss,
        x_train,
        y_train,
        x_valid,
        y_valid,
        iterations: list,
        train_losses: list,
        valid_losses: list,
        max_iter: int,
    ):
        """Training for the different networks"""

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

        train_loss.reset_state()
        valid_loss.reset_state()
        test_loss.reset_state()

        for i in range(max_iter):
            train_loss.reset_state()  # TODO: This line was not here before
            valid_loss.reset_state()
            test_loss.reset_state()

            train_step_fun(x_train, y_train, x_valid, y_valid)

            current_train_loss = train_loss.result()
            current_valid_loss = valid_loss.result()

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

            train_losses.append(current_train_loss.numpy())
            valid_losses.append(current_valid_loss.numpy())

            if (
                self.configs["early_stop"]
                and i >= self.configs["early_stop_start_iter"]
            ):
                if np.less(current_valid_loss - min_delta, best_loss):
                    best_loss = current_valid_loss
                    early_stop_wait = 0
                    if self.configs["restore_best_weights"]:
                        best_weights = network.get_weights()
                else:
                    early_stop_wait += 1
                    # print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
                    if early_stop_wait >= self.configs["wait_patience"]:
                        stop_training = True
                        if self.configs["restore_best_weights"]:
                            if best_weights is not None:
                                if self.configs["verbose"] > 0:
                                    print(
                                        f"--- Restoring {network_type} model weights from the end of the best iteration"
                                    )
                                network.set_weights(best_weights)
                        if self.configs["saveWeights"]:
                            print(
                                f"--- Saving best model weights to h5 file: {self.configs['data_name']}_best_{network_type}_iter_{i+1}.h5"
                            )
                            network.save_weights(
                                os.getcwd()
                                + f"/Results_PI3NN/checkpoints_{network_type}/"
                                + self.configs["data_name"]
                                + f"_best_{network_type}_iter_"
                                + str(i + 1)
                                + ".h5"
                            )
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
            network=self.trainSteps.net_mean,
            train_step_fun=self.trainSteps.train_step_mean,
            train_loss=self.trainSteps.train_loss_net_mean,
            valid_loss=self.trainSteps.valid_loss_net_mean,
            test_loss=self.trainSteps.test_loss_net_mean,
            x_train=self.xTrain,
            y_train=self.yTrain,
            x_valid=self.xValid,
            y_valid=self.yValid,
            iterations=self.iter_mean_list,
            train_losses=self.train_loss_mean_list,
            valid_losses=self.valid_loss_mean_list,
            max_iter=self.configs["Max_iter"],
        )

        # IMPORTANT: has to be created after mean network finished training
        self.xValid_up, self.yValid_up, self.xValid_down, self.yValid_down = (
            self.createUpDownTrainingData()
        )

        self.main_train_step(
            network_type="up",
            network=self.trainSteps.net_std_up,
            train_step_fun=self.trainSteps.train_step_up,
            train_loss=self.trainSteps.train_loss_net_std_up,
            valid_loss=self.trainSteps.valid_loss_net_std_up,
            test_loss=self.trainSteps.test_loss_net_std_up,
            x_train=self.xTrain_up,
            y_train=self.yTrain_up,
            x_valid=self.xValid_up,
            y_valid=self.yValid_up,
            iterations=self.iter_up_list,
            train_losses=self.train_loss_up_list,
            valid_losses=self.valid_loss_up_list,
            max_iter=self.configs["Max_iter"],
        )

        self.main_train_step(
            network_type="down",
            network=self.trainSteps.net_std_down,
            train_step_fun=self.trainSteps.train_step_down,
            train_loss=self.trainSteps.train_loss_net_std_down,
            valid_loss=self.trainSteps.valid_loss_net_std_down,
            test_loss=self.trainSteps.test_loss_net_std_down,
            x_train=self.xTrain_down,
            y_train=self.yTrain_down,
            x_valid=self.xValid_down,
            y_valid=self.yValid_down,
            iterations=self.iter_down_list,
            train_losses=self.train_loss_down_list,
            valid_losses=self.valid_loss_down_list,
            max_iter=self.configs["Max_iter"],
        )

    def createUpDownTrainingData(self):
        """Generate up and down training/validation data"""
        diff_train = self.yTrain.reshape(
            self.yTrain.shape[0], -1
        ) - self.trainSteps.net_mean(self.xTrain, training=False)
        yTrain_up_data = tf.expand_dims(diff_train[diff_train > 0], axis=1)
        xTrain_up_data = self.xTrain[(diff_train > 0).numpy().flatten(), :]
        yTrain_down_data = -1.0 * tf.expand_dims(diff_train[diff_train < 0], axis=1)
        xTrain_down_data = self.xTrain[(diff_train < 0).numpy().flatten(), :]

        self.xTrain_up = xTrain_up_data
        self.yTrain_up = yTrain_up_data.numpy()
        self.xTrain_down = xTrain_down_data
        self.yTrain_down = yTrain_down_data.numpy()

        diff_valid = self.yValid.reshape(
            self.yValid.shape[0], -1
        ) - self.trainSteps.net_mean(self.xValid, training=False)
        yValid_up_data = tf.expand_dims(diff_valid[diff_valid > 0], axis=1)
        xValid_up_data = self.xValid[(diff_valid > 0).numpy().flatten(), :]
        yValid_down_data = -1.0 * tf.expand_dims(diff_valid[diff_valid < 0], axis=1)
        xValid_down_data = self.xValid[(diff_valid < 0).numpy().flatten(), :]

        return (
            xValid_up_data,
            yValid_up_data.numpy(),
            xValid_down_data,
            yValid_down_data.numpy(),
        )

    def boundaryOptimization(self, verbose=0):
        Ntrain = self.xTrain.shape[0]
        output = self.trainSteps.net_mean(self.xTrain, training=False)
        output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
        output_down = self.trainSteps.net_std_down(self.xTrain, training=False)

        try:
            self.configs["quantile_list"]
            if (
                self.configs["quantile_list"] is not None
            ):  ### use quantile_list to override the single quantile'
                if verbose > 0:
                    print("--- Start boundary optimizations for MULTIPLE quantiles...")
                self.quantile_list = self.configs["quantile_list"]
                num_outlier_list = [
                    int(Ntrain * (1 - x) / 2) for x in self.quantile_list
                ]
                # if verbose > 0:
                #     print('-- Number of outlier based on the defined quantile:')
                #     print(num_outlier_list)
                boundaryOptimizer = CL_boundary_optimizer(
                    self.yTrain,
                    output,
                    output_up,
                    output_down,
                    c_up0_ini=0.0,
                    c_up1_ini=100000.0,
                    c_down0_ini=0.0,
                    c_down1_ini=100000.0,
                    max_iter=1000,
                )

                self.c_up_list = [
                    boundaryOptimizer.optimize_up(outliers=x, verbose=0)
                    for x in num_outlier_list
                ]
                self.c_down_list = [
                    boundaryOptimizer.optimize_down(outliers=x, verbose=0)
                    for x in num_outlier_list
                ]
                if verbose > 0:
                    for idx, item in enumerate(self.quantile_list):
                        print(
                            "--- Quantile: {:.4f}, outlisers: {}, c_up: {:.4f}, c_down: {:.4f}".format(
                                item,
                                num_outlier_list[idx],
                                self.c_up_list[idx],
                                self.c_down_list[idx],
                            )
                        )
                    # print('c_up: {}'.format(self.c_up_list))
                    # print('c_down: {}'.format(self.c_down_list))
            else:
                print(
                    "--- WARNING: configs['quantile_list'] exist but missing values (list of quantile values) or is None"
                )
                print(
                    "--- Code stop running. Please assign values or comment out the line: configs['quantile_list'] = xxx"
                )
                exit()

        except KeyError:
            num_outlier = int(Ntrain * (1 - self.configs["quantile"]) / 2)
            if verbose > 0:
                print(
                    "--- Start boundary optimizations for SINGLE quantile: {}".format(
                        self.configs["quantile"]
                    )
                )
                print(
                    "--- Number of outlier based on the defined quantile: {}".format(
                        num_outlier
                    )
                )
            boundaryOptimizer = CL_boundary_optimizer(
                self.yTrain,
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
            self.c_up = boundaryOptimizer.optimize_up(verbose=0)
            self.c_down = boundaryOptimizer.optimize_down(verbose=0)
            if verbose > 0:
                print("--- c_up: {}".format(self.c_up))
                print("--- c_down: {}".format(self.c_down))

    def testDataPrediction(self):
        self.train_output = self.trainSteps.net_mean(self.xTrain, training=False)
        self.train_output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
        self.train_output_down = self.trainSteps.net_std_down(
            self.xTrain, training=False
        )

        self.valid_output = self.trainSteps.net_mean(self.xValid, training=False)
        self.valid_output_up = self.trainSteps.net_std_up(self.xValid, training=False)
        self.valid_output_down = self.trainSteps.net_std_down(
            self.xValid, training=False
        )

        self.test_output = self.trainSteps.net_mean(self.xTest, training=False)
        self.test_output_up = self.trainSteps.net_std_up(self.xTest, training=False)
        self.test_output_down = self.trainSteps.net_std_down(self.xTest, training=False)

    def capsCalculation(self, final_evaluation=False, verbose=0):
        if hasattr(self, "quantile_list"):
            if self.quantile_list is not None:
                if verbose > 0:
                    print("--- Start caps calculations for MULTIPLE quantiles...")
                    print("**************** For Training data *****************")
                ### caps calculations for multiple quantiles (training data)
                if len(self.yTrain.shape) == 2:
                    self.yTrain = self.yTrain.flatten()  # single y output
                y_U_cap_train_list = [
                    (self.train_output + x * self.train_output_up).numpy().flatten()
                    > self.yTrain
                    for x in self.c_up_list
                ]
                y_L_cap_train_list = [
                    (self.train_output - x * self.train_output_down).numpy().flatten()
                    < self.yTrain
                    for x in self.c_down_list
                ]
                y_all_cap_train_list = [
                    (x * y) for x, y in zip(y_U_cap_train_list, y_L_cap_train_list)
                ]
                self.PICP_train_list = [
                    np.sum(x / y.shape[0])
                    for x, y in zip(y_all_cap_train_list, y_L_cap_train_list)
                ]
                self.MPIW_train_list = [
                    np.mean(
                        (self.train_output + x * self.train_output_up).numpy().flatten()
                        - (self.train_output - y * self.train_output_down)
                        .numpy()
                        .flatten()
                    )
                    for x, y in zip(self.c_up_list, self.c_down_list)
                ]
                if verbose > 0:
                    for idx, item in enumerate(self.quantile_list):
                        print(
                            "--- Train quantile: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}".format(
                                item,
                                self.PICP_train_list[idx],
                                self.MPIW_train_list[idx],
                            )
                        )

                ### caps calculations for multiple quantiles (validation data)
                if verbose > 0:
                    print("**************** For Validation data *****************")
                if len(self.yValid.shape) == 2:
                    self.yValid = self.yValid.flatten()
                y_U_cap_valid_list = [
                    (self.valid_output + x * self.valid_output_up).numpy().flatten()
                    > self.yValid
                    for x in self.c_up_list
                ]
                y_L_cap_valid_list = [
                    (self.valid_output - x * self.valid_output_down).numpy().flatten()
                    < self.yValid
                    for x in self.c_down_list
                ]
                y_all_cap_valid_list = [
                    (x * y) for x, y in zip(y_U_cap_valid_list, y_L_cap_valid_list)
                ]
                self.PICP_valid_list = [
                    np.sum(x / y.shape[0])
                    for x, y in zip(y_all_cap_valid_list, y_L_cap_valid_list)
                ]
                self.MPIW_valid_list = [
                    np.mean(
                        (self.valid_output + x * self.valid_output_up).numpy().flatten()
                        - (self.valid_output - y * self.valid_output_down)
                        .numpy()
                        .flatten()
                    )
                    for x, y in zip(self.c_up_list, self.c_down_list)
                ]
                if verbose > 0:
                    print("--- Quantiles for validation data:")
                    for idx, item in enumerate(self.quantile_list):
                        print(
                            "--- Valid quantile: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}".format(
                                item,
                                self.PICP_valid_list[idx],
                                self.MPIW_valid_list[idx],
                            )
                        )

                ### caps calculations for multiple quantiles (testing data, for final evaluation)
                if final_evaluation:
                    if verbose > 0:
                        print("**************** For Testing data *****************")
                    if len(self.yTest.shape) == 2:
                        self.yTest = self.yTest.flatten()
                    y_U_cap_test_list = [
                        (self.test_output + x * self.test_output_up).numpy().flatten()
                        > self.yTest
                        for x in self.c_up_list
                    ]
                    y_L_cap_test_list = [
                        (self.test_output - x * self.test_output_down).numpy().flatten()
                        < self.yTest
                        for x in self.c_down_list
                    ]
                    y_all_cap_test_list = [
                        (x * y) for x, y in zip(y_U_cap_test_list, y_L_cap_test_list)
                    ]
                    self.PICP_test_list = [
                        np.sum(x / y.shape[0])
                        for x, y in zip(y_all_cap_test_list, y_L_cap_test_list)
                    ]
                    self.MPIW_test_list = [
                        np.mean(
                            (self.test_output + x * self.test_output_up)
                            .numpy()
                            .flatten()
                            - (self.test_output - y * self.test_output_down)
                            .numpy()
                            .flatten()
                        )
                        for x, y in zip(self.c_up_list, self.c_down_list)
                    ]
                    if verbose > 0:
                        print("--- Quantiles for testing data:")
                        for idx, item in enumerate(self.quantile_list):
                            print(
                                "--- Test quantile: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}".format(
                                    item,
                                    self.PICP_test_list[idx],
                                    self.MPIW_test_list[idx],
                                )
                            )

                    self.MSE_test = np.mean(
                        np.square(self.test_output.numpy().flatten() - self.yTest)
                    )
                    self.RMSE_test = np.sqrt(self.MSE_test)
                    self.R2_test = r2_score(
                        self.yTest, self.test_output.numpy().flatten()
                    )
                    if verbose > 0:
                        print("Test MSE: {}".format(self.MSE_test))
                        print("Test RMSE: {}".format(self.RMSE_test))
                        print("Test R2: {}".format(self.R2_test))
                # return
            else:
                print(
                    "--- WARNING: configs['quantile_list'] exist but missing values (list of quantile values) or is None"
                )
                print(
                    "--- Code stop running. Please assign values or comment out the line: configs['quantile_list'] = xxx"
                )
                exit()
        else:
            ### caps calculations for single quantile
            if verbose > 0:
                print(
                    "--- Start caps calculations for SINGLE quantile: {}".format(
                        self.configs["quantile"]
                    )
                )
                print("**************** For Training data *****************")
            if len(self.yTrain.shape) == 2:
                self.yTrain = self.yTrain.flatten()
            y_U_cap_train = (
                self.train_output + self.c_up * self.train_output_up
            ).numpy().flatten() > self.yTrain
            y_L_cap_train = (
                self.train_output - self.c_down * self.train_output_down
            ).numpy().flatten() < self.yTrain

            y_all_cap_train = y_U_cap_train * y_L_cap_train  # logic_or
            self.PICP_train = np.sum(y_all_cap_train) / y_L_cap_train.shape[0]  # 0-1
            self.MPIW_train = np.mean(
                (self.train_output + self.c_up * self.train_output_up).numpy().flatten()
                - (self.train_output - self.c_down * self.train_output_down)
                .numpy()
                .flatten()
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
            if len(self.yValid.shape) == 2:
                self.yValid = self.yValid.flatten()
            y_U_cap_valid = (
                self.valid_output + self.c_up * self.valid_output_up
            ).numpy().flatten() > self.yValid
            y_L_cap_valid = (
                self.valid_output - self.c_down * self.valid_output_down
            ).numpy().flatten() < self.yValid
            y_all_cap_valid = y_U_cap_valid * y_L_cap_valid  # logic_or
            self.PICP_valid = np.sum(y_all_cap_valid) / y_L_cap_valid.shape[0]  # 0-1
            self.MPIW_valid = np.mean(
                (self.valid_output + self.c_up * self.valid_output_up).numpy().flatten()
                - (self.valid_output - self.c_down * self.valid_output_down)
                .numpy()
                .flatten()
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

            if final_evaluation:
                if verbose > 0:
                    print("**************** For Testing data *****************")
                if len(self.yTest.shape) == 2:
                    self.yTest = self.yTest.flatten()
                y_U_cap_test = (
                    self.test_output + self.c_up * self.test_output_up
                ).numpy().flatten() > self.yTest
                y_L_cap_test = (
                    self.test_output - self.c_down * self.test_output_down
                ).numpy().flatten() < self.yTest
                y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
                self.PICP_test = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
                self.MPIW_test = np.mean(
                    (self.test_output + self.c_up * self.test_output_up)
                    .numpy()
                    .flatten()
                    - (self.test_output - self.c_down * self.test_output_down)
                    .numpy()
                    .flatten()
                )
                # print('y_U_cap: {}'.format(y_U_cap))
                # print('y_L_cap: {}'.format(y_L_cap))
                if verbose > 0:
                    print(
                        "Num of true in y_U_cap: {}".format(
                            np.count_nonzero(y_U_cap_test)
                        )
                    )
                    print(
                        "Num of true in y_L_cap: {}".format(
                            np.count_nonzero(y_L_cap_test)
                        )
                    )
                    print(
                        "Num of true in y_all_cap: {}".format(
                            np.count_nonzero(y_all_cap_test)
                        )
                    )
                    print("np.sum results: {}".format(np.sum(y_all_cap_test)))
                    print("PICP_test: {}".format(self.PICP_test))
                    print("MPIW_test: {}".format(self.MPIW_test))

                # print(y_all_cap)
                # print(np.sum(y_all_cap_test))

                self.MSE_test = np.mean(
                    np.square(self.test_output.numpy().flatten() - self.yTest)
                )
                self.RMSE_test = np.sqrt(self.MSE_test)
                self.R2_test = r2_score(self.yTest, self.test_output.numpy().flatten())
                if verbose > 0:
                    print("Test MSE: {}".format(self.MSE_test))
                    print("Test RMSE: {}".format(self.RMSE_test))
                    print("Test R2: {}".format(self.R2_test))
            else:
                self.PICP_test = None
                self.MPIW_test = None

            return (
                self.PICP_train,
                self.PICP_valid,
                self.PICP_test,
                self.MPIW_train,
                self.MPIW_valid,
                self.MPIW_test,
            )

    def saveResultsToTxt(self):
        """Save results to txt file"""
        results_path = (
            "./Results_PI3NN/" + self.configs["data_name"] + "_PI3NN_results.txt"
        )
        with open(results_path, "a") as fwrite:
            fwrite.write(
                str(self.configs["experiment_id"])
                + " "
                + str(self.configs["seed"])
                + " "
                + str(round(self.PICP_test, 3))
                + " "
                + str(round(self.MPIW_test, 3))
                + " "
                + str(round(self.RMSE_test, 3))
                + " "
                + str(round(self.R2_test, 3))
                + "\n"
            )

    def save_PI(self, bias=0.0):
        y_U_PI_array_train = (
            (self.train_output + self.c_up * self.train_output_up).numpy().flatten()
        )
        y_L_PI_array_train = (
            (self.train_output - self.c_down * self.train_output_down).numpy().flatten()
        )

        y_U_PI_array_test = (
            (self.test_output + self.c_up * self.test_output_up).numpy().flatten()
        )
        y_L_PI_array_test = (
            (self.test_output - self.c_down * self.test_output_down).numpy().flatten()
        )

        path = "./Results_PI3NN/npy/"
        train_bounds = np.vstack((y_U_PI_array_train, y_L_PI_array_train))
        test_bounds = np.vstack((y_U_PI_array_test, y_L_PI_array_test))
        np.save(path + "train_bounds" + "_bias_" + str(bias) + ".npy", train_bounds)
        np.save(path + "test_bounds" + "_bias_" + str(bias) + ".npy", test_bounds)
        np.save(path + "yTrain" + "_bias_" + str(bias) + ".npy", self.yTrain)
        np.save(path + "yTest" + "_bias_" + str(bias) + ".npy", self.yTest)
        print("--- results npy saved")

    def load_and_plot_PI(self, bias=0.0):
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


class UQ_Net_mean_TF2(Model):
    def __init__(self, configs, num_inputs, num_outputs):
        super(UQ_Net_mean_TF2, self).__init__()
        self.configs = configs
        self.num_nodes_list = list(self.configs["num_neurons_mean"])

        self.inputLayer = Dense(num_inputs, activation="linear")
        initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)

        self.fcs = []
        for i in range(len(self.num_nodes_list)):
            self.fcs.append(
                Dense(
                    self.num_nodes_list[i],
                    activation="relu",
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02),
                )
            )

        self.outputLayer = Dense(num_outputs)

    def call(self, x):
        x = self.inputLayer(x)
        for i in range(len(self.num_nodes_list)):
            x = self.fcs[i](x)
        x = self.outputLayer(x)
        return x


class UQ_Net_std_TF2(Model):
    def __init__(self, configs, num_inputs, num_outputs, net=None, bias=None):
        super(UQ_Net_std_TF2, self).__init__()
        self.configs = configs
        if net == "up":
            self.num_nodes_list = list(self.configs["num_neurons_up"])
        elif net == "down":
            self.num_nodes_list = list(self.configs["num_neurons_down"])

        self.inputLayer = Dense(num_inputs, activation="linear")
        initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)

        self.fcs = []
        for i in range(len(self.num_nodes_list)):
            self.fcs.append(
                Dense(
                    self.num_nodes_list[i],
                    activation="relu",
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02),
                )
            )

        self.outputLayer = Dense(num_outputs)
        if bias is None:
            self.custom_bias = tf.Variable([3.0])
        else:
            self.custom_bias = tf.Variable([bias])

    def call(self, x):
        x = self.inputLayer(x)
        for i in range(len(self.num_nodes_list)):
            x = self.fcs[i](x)
        x = self.outputLayer(x)
        x = tf.nn.bias_add(x, self.custom_bias)
        x = tf.math.sqrt(tf.math.square(x) + 0.2)  # 1e-10 0.2
        return x


class CL_UQ_Net_train_steps:
    def __init__(
        self,
        net_mean,
        net_std_up,
        net_std_down,
        optimizers=["Adam", "Adam", "Adam"],
        lr=[0.01, 0.01, 0.01],
        exponential_decay=False,
        decay_steps=None,
        decay_rate=None,
    ):
        self.exponential_decay = exponential_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.criterion_mean = tf.keras.losses.MeanSquaredError()
        self.criterion_std = tf.keras.losses.MeanSquaredError()

        # accumulate the loss and compute the mean until .reset_state()
        self.train_loss_net_mean = tf.keras.metrics.Mean(name="train_loss_net_mean")
        self.train_loss_net_std_up = tf.keras.metrics.Mean(name="train_loss_net_std_up")
        self.train_loss_net_std_down = tf.keras.metrics.Mean(
            name="train_loss_net_std_down"
        )
        self.valid_loss_net_mean = tf.keras.metrics.Mean(name="valid_loss_net_mean")
        self.valid_loss_net_std_up = tf.keras.metrics.Mean(name="valid_loss_net_std_up")
        self.valid_loss_net_std_down = tf.keras.metrics.Mean(
            name="valid_loss_net_std_down"
        )
        self.test_loss_net_mean = tf.keras.metrics.Mean(name="test_loss_net_mean")
        self.test_loss_net_std_up = tf.keras.metrics.Mean(name="test_loss_net_std_up")
        self.test_loss_net_std_down = tf.keras.metrics.Mean(
            name="test_loss_net_std_down"
        )

        self.net_mean = net_mean
        if exponential_decay is False:
            if optimizers[0] == "Adam":
                self.optimizer_net_mean = tf.keras.optimizers.Adam(learning_rate=lr[0])
            elif optimizers[0] == "SGD":
                self.optimizer_net_mean = tf.keras.optimizers.SGD(learning_rate=lr[0])
        else:
            self.global_step_0 = tf.Variable(0, trainable=False)
            decayed_l_rate_0 = tf.compat.v1.train.exponential_decay(
                lr[0],
                self.global_step_0,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )
            if optimizers[0] == "Adam":
                self.optimizer_net_mean = tf.keras.optimizers.Adam(
                    learning_rate=decayed_l_rate_0
                )
            elif optimizers[0] == "SGD":
                self.optimizer_net_mean = tf.keras.optimizers.SGD(
                    learning_rate=decayed_l_rate_0
                )

        self.net_std_up = net_std_up
        if exponential_decay is False:
            if optimizers[1] == "Adam":
                self.optimizer_net_std_up = tf.keras.optimizers.Adam(
                    learning_rate=lr[1]
                )
            elif optimizers[1] == "SGD":
                self.optimizer_net_std_up = tf.keras.optimizers.SGD(learning_rate=lr[1])
        else:
            self.global_step_1 = tf.Variable(0, trainable=False)
            decayed_l_rate_1 = tf.compat.v1.train.exponential_decay(
                lr[1],
                self.global_step_1,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )
            if optimizers[1] == "Adam":
                self.optimizer_net_std_up = tf.keras.optimizers.Adam(
                    learning_rate=decayed_l_rate_1
                )
            elif optimizers[1] == "SGD":
                self.optimizer_net_std_up = tf.keras.optimizers.SGD(
                    learning_rate=decayed_l_rate_1
                )

        self.net_std_down = net_std_down
        if exponential_decay is False:
            if optimizers[2] == "Adam":
                self.optimizer_net_std_down = tf.keras.optimizers.Adam(
                    learning_rate=lr[2]
                )
            elif optimizers[2] == "SGD":
                self.optimizer_net_std_down = tf.keras.optimizers.SGD(
                    learning_rate=lr[2]
                )
        else:
            self.global_step_2 = tf.Variable(0, trainable=False)
            decayed_l_rate_2 = tf.compat.v1.train.exponential_decay(
                lr[2],
                self.global_step_2,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )
            if optimizers[1] == "Adam":
                self.optimizer_net_std_down = tf.keras.optimizers.Adam(
                    learning_rate=decayed_l_rate_2
                )
            elif optimizers[1] == "SGD":
                self.optimizer_net_std_down = tf.keras.optimizers.SGD(
                    learning_rate=decayed_l_rate_2
                )

    def add_model_regularizer_loss(self, model):
        loss = 0
        for l in model.layers:
            # if hasattr(l, 'layers') and l.layers:  # the layer itself is a model
            #     loss += add_model_loss(l)
            if hasattr(l, "kernel_regularizer") and l.kernel_regularizer:
                loss += l.kernel_regularizer(l.kernel)
            if hasattr(l, "bias_regularizer") and l.bias_regularizer:
                loss += l.bias_regularizer(l.bias)
        return loss

    @tf.function
    def train_step_mean(self, xTrain, yTrain, xValid, yValid):
        """Training/validation for mean values (For Non-batch training)"""
        with tf.GradientTape() as tape:
            train_predictions = self.net_mean(xTrain, training=True)
            train_loss = self.criterion_mean(yTrain, train_predictions)
            """ Add regularization losses """
            train_loss += self.add_model_regularizer_loss(self.net_mean)

            valid_predictions = self.net_mean(xValid, training=False)
            valid_loss = self.criterion_mean(yValid, valid_predictions)

            test_loss = 0
        gradients = tape.gradient(train_loss, self.net_mean.trainable_variables)
        self.optimizer_net_mean.apply_gradients(
            zip(gradients, self.net_mean.trainable_variables)
        )

        self.train_loss_net_mean(
            train_loss
        )  # accumulate the loss and compute the mean until .reset_state()
        self.valid_loss_net_mean(valid_loss)

        if self.exponential_decay:
            self.global_step_0.assign_add(1)

    @tf.function
    def train_step_up(self, xTrain, yTrain, xValid, yValid):
        """Training/validation for upper boundary (For Non-batch training)"""
        with tf.GradientTape() as tape:
            train_predictions = self.net_std_up(xTrain, training=True)
            train_loss = self.criterion_std(yTrain, train_predictions)
            """ Add regularization losses """
            train_loss += self.add_model_regularizer_loss(self.net_std_up)

            valid_predictions = self.net_std_up(xValid, training=False)
            valid_loss = self.criterion_std(yValid, valid_predictions)

        gradients = tape.gradient(train_loss, self.net_std_up.trainable_variables)
        self.optimizer_net_std_up.apply_gradients(
            zip(gradients, self.net_std_up.trainable_variables)
        )

        self.train_loss_net_std_up(train_loss)
        self.valid_loss_net_std_up(valid_loss)

        if self.exponential_decay:
            self.global_step_1.assign_add(1)

    @tf.function
    def train_step_down(self, xTrain, yTrain, xValid, yValid):
        """Training/validation for lower boundary (For Non-batch training)"""
        with tf.GradientTape() as tape:
            train_predictions = self.net_std_down(xTrain, training=True)
            train_loss = self.criterion_std(yTrain, train_predictions)
            """ Add regularization losses """
            train_loss += self.add_model_regularizer_loss(self.net_std_down)

            valid_predictions = self.net_std_down(xValid, training=False)
            valid_loss = self.criterion_std(yValid, valid_predictions)

        gradients = tape.gradient(train_loss, self.net_std_down.trainable_variables)
        self.optimizer_net_std_down.apply_gradients(
            zip(gradients, self.net_std_down.trainable_variables)
        )

        self.train_loss_net_std_down(train_loss)
        self.valid_loss_net_std_down(valid_loss)

        if self.exponential_decay:
            self.global_step_2.assign_add(1)


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
        self.yTrain = yTrain.flatten()

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

    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    num_threads = 8
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
    os.environ["TF_NUM_INTEROP_THREADS"] = "8"
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.set_soft_device_placement(True)

    # Force using CPU globally by hiding GPU(s), comment the line of code below to enable GPU
    tf.config.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="artificial",
        help="example data names: boston, artificial",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()

    """ 
    If you would like to customize the data loading and pre-processing, we recommend you to write the 
    functions in src/DataLoaders/data_loaders.py and call it from here. Or write them directly here.
    We provide an example on 'boston_housing.txt' dataset below:      

    """

    ##########################################################
    ################## Data Loading Section ##################
    ##########################################################
    data_dir = "./datasets/UCI_datasets/"
    dataLoader = CL_dataLoader(original_data_path=data_dir)
    X, Y = dataLoader.load_single_dataset(args.data)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    # ### Split train/test data  (manual specification or random split using sklearn)
    # # Ntest = 100
    # # xTrain, xTest = X[:-Ntest, :], X[-Ntest:, :]
    # # yTrain, yTest = Y[:-Ntest, :], Y[-Ntest:, :]

    # ## or random split
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
    configs["saveWeights"] = False
    configs["loadWeights_test"] = False
    configs["early_stop"] = True
    configs["early_stop_start_iter"] = 100  # 60
    configs["wait_patience"] = 300
    configs["restore_best_weights"] = True
    print("--- Dataset: {}".format(configs["data_name"]))
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    tf.random.set_seed(configs["seed"])

    """ Create network instances"""
    net_mean = UQ_Net_mean_TF2(configs, num_inputs, num_outputs)
    net_up = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net="up")
    net_down = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net="down")

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
    trainer.boundaryOptimization(verbose=1)  # boundary optimization
    trainer.testDataPrediction()  # evaluation of the trained nets on testing data
    trainer.capsCalculation(final_evaluation=True, verbose=1)  # metrics calculation
    trainer.saveResultsToTxt()  # save results to txt file

    fig, ax = plt.subplots()
    ax.plot(xTrain, yTrain, ".")
    y_U_PI_array_train = (
        (trainer.train_output + trainer.c_up * trainer.train_output_up)
        .numpy()
        .flatten()
    )
    y_L_PI_array_train = (
        (trainer.train_output - trainer.c_down * trainer.train_output_down)
        .numpy()
        .flatten()
    )
    y_mean = trainer.train_output.numpy().flatten()
    sort_indices = np.argsort(xTrain.flatten())
    ax.plot(xTrain.flatten()[sort_indices], y_mean[sort_indices], "-")
    ax.plot(xTrain.flatten()[sort_indices], y_U_PI_array_train[sort_indices], "-")
    ax.plot(xTrain.flatten()[sort_indices], y_L_PI_array_train[sort_indices], "-")
    plt.show()


if __name__ == "__main__":
    main()
