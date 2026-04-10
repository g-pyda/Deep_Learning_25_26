# entry point for the training pipeline

# source /home/staff/brachwal/public/set_conda.sh
# conda activate monai_hface_accelerate

import torch
import sys

from preprocessing import get_data
from model import CNNBuilder
from train import train, test
from utils import load_config


if __name__ == "__main__":
    if sys.argv[1]:
        config = load_config(sys.argv[1])
    else:
        config = load_config()

    # device specification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # data loading
    train_ds, test_ds = get_data(config['data'])

    # model initialization
    model = CNNBuilder(config["model_layers"]).to(device)

    # model training
    train(model, train_loader, device)

    # model evaluation
    test(model, test_loader, device)