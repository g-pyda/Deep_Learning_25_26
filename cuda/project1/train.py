# training and validation loops

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch import nn, optim

def train(model, train_loader, device):
    # Create a TensorBoard writer
    writer = SummaryWriter(log_dir="runs/baseline_experiment")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # PyTorch accumulates gradients by default, so we need to clear them before computing new ones
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