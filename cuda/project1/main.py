# entry point for the training pipeline

# source /home/staff/brachwal/public/set_conda.sh
# conda activate monai_hface_accelerate

import torch
import numpy as np
import random
import sys

from preprocessing import get_data
from model import CNNBuilder
from train import train, test
from utils import load_config, permute_hyperparams, global_log


if __name__ == "__main__":
    if sys.argv[1]:
        config = load_config(sys.argv[1])
    else:
        config = load_config()

    basic_config = config.get("basic")
    log_config = config.get("logging")

    # logging setup (enable/disable and set paths)
    study_name = basic_config.get("study_name", "run")
    logs_dir = f"./logs/{study_name}"
    global_log.log_path = f"{logs_dir}/all.log"
    if log_config:
        if not log_config.get("time_logging"):
            global_log.time_path = None
        else:
            global_log.time_path = f"{logs_dir}/time.log"

        if not log_config.get("error_logging"):
            global_log.error_path = None
        else:
            global_log.error_path = f"{logs_dir}/error.log"

        if not log_config.get("gpu_logging"):
            global_log.gpu_path = None
        else:
            global_log.gpu_path = f"{logs_dir}/gpu.log"

    # random seed and numworkers setup 
    random_seed = basic_config.get("random_seed", 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    num_workers = basic_config.get("num_workers", 4)

    # device specification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # data loading
    train_ds, test_ds = get_data(config['data'])

    hyperparameters = permute_hyperparams(config["hyperparameters"])
    for model_nr, hpset in enumerate(hyperparameters):
        # model initialization
        model = CNNBuilder(config["model_layers"]).to(device)

        # model training
        train(model, model_nr, train_ds, device, hpset, num_workers)

        # model evaluation
        test(model, test_ds, device)