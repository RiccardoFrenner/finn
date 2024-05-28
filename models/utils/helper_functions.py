import torch as th
import os


def determine_device(print_progress=False):
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    :return: The device where tensor calculations shall be made on
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if print_progress:
        print("Using device:", device, "\n")

    # Additional Info when using cuda
    if device.type == "cuda" and print_progress:
        print(th.cuda.get_device_name(0))
        print("Memory Usage:")
        print("\tAllocated:",
              round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("\tCached:   ", round(th.cuda.memory_reserved(0) / 1024 ** 3, 1),
              "GB")
        print()
        
    return device


def save_model_to_file(model_src_path, config, epoch, epoch_errors_train,
                       epoch_errors_valid, net):
    """
    This function writes the model weights along with the network configuration
    and current performance to file.
    :param model_src_path: The source path where the model will be saved to
    :param config: The configurations of the model
    :param epoch: The current epoch
    :param epoch_errors_train: The training epoch errors
    :param epoch_errors_valid: The validation epoch errors,
    :param net: The actual model
    :return: Nothing
    """

    model_save_path = os.path.join(
        model_src_path, "checkpoints", config.model.name
    )

    os.makedirs(model_save_path, exist_ok=True)

    # Save model weights to file
    th.save(net.state_dict(), 
            os.path.join(model_save_path, config.model.name + ".pt"))

    # Copy the configurations and add a results entry
    config["results"] = {
        "current_epoch": epoch + 1,
        "current_training_error": epoch_errors_train[-1],
        "lowest_train_error": min(epoch_errors_train),
        "current_validation_error": epoch_errors_valid[-1],
        "lowest_validation_error": min(epoch_errors_valid)
    }

    # Save the configuration and current performance to file
    config.save(model_save_path)
