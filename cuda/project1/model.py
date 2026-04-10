# model architecture definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for layer in config:
            pass

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x