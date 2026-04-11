# training and validation loops

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch import nn, optim
from torch.utils.data import DataLoader

def train(model, train_data, device, hpset, num_workers):
    # getting the training assets
    batch_size, optimizer, num_epochs = get_train_assets(hpset, model)

    # creating the data loader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        )

    # Create a TensorBoard writer
    writer = SummaryWriter(log_dir="runs/baseline_experiment")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        # writer.add_scalar("Loss/train", epoch_loss, epoch)

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")



    # Example: log images (first batch of training data)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    img_grid = vutils.make_grid(images[:8])
    writer.add_image("Sample Inputs", img_grid, 0)

    # Example: log weight histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, epoch)


    # log hyperparameters
    hparams = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": train_loader.batch_size,
        "optimizer": "Adam",
        "epochs": 10
    }

    metrics = {
        "hparam/train_loss": epoch_loss
    }

    writer.add_hparams(hparams, metrics)

    # Run `tensorboard --logdir=runs` to visualize (in terminal)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # disables gradient tracking
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# =======================================
# HELPERS
# =======================================

def get_train_assets(hpset, model):
    batch_size = hpset.get("batch_size")
    num_epochs = hpset.get("num_epochs")
    opt = hpset.get("optimizer")
    optimizer = None
    match opt:
        case "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=hpset.get("learning_rate"),
                weight_decay=hpset.get("weight_decay")
            )
        case "SGD" | "Sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=hpset.get("learning_rate"),
                weight_decay=hpset.get("weight_decay")
            )
        case "AdamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=hpset.get("learning_rate"),
                weight_decay=hpset.get("weight_decay")
            )
        case "RMSProp":
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=hpset.get("learning_rate"),
                weight_decay=hpset.get("weight_decay")
            )
        case "Adagrad":
            optimizer = optim.Adagrad(
                model.parameters(),
                lr=hpset.get("learning_rate"),
                weight_decay=hpset.get("weight_decay")
            )
        case "Adadelta":
            optimizer = optim.Adadelta(
                model.parameters(),
                lr=hpset.get("learning_rate"),
                weight_decay=hpset.get("weight_decay")
            )
        case _:
            raise ValueError(f"Unknown optimizer: {opt}")

    return batch_size, optimizer, num_epochs