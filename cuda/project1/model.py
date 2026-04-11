# model architecture definition
import torch
import torch.nn as nn
import torch.nn.functionals as F

# ===============================================
# IMPORTED FUNCTIONS
# ===============================================

class CNNBuilder2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [] # contains tuples (name, layer, batch_norm, activation)
        for layer_config in config:
            # retrieving the data
            name = layer_config["name"]
            activation = load_layer_activation(layer_config)
            layer, batch_norm = load_layer_type(layer_config)

            # saving the layer chunk
            self.layers.append((name, layer, batch_norm, activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer[1](x)
            if layer[3]:
                x = layer[3](x)
            if layer[2]:
                x = layer[2](x)
        return x
    

# ===============================================
# HELPERS
# ===============================================

def load_layer_activation(layer_config):
    activation = None
    match layer_config["activation"]:
        # Non-linearities
        case "relu":
            activation = F.relu
        case "relu6":
            activation = F.relu6
        case "elu":
            activation = F.elu
        case "selu":
            activation = F.selu
        case "leaky_relu":
            activation = F.leaky_relu
        case "prelu":
            activation = F.prelu
        # Sigmoid/Tanh
        case "sigmoid":
            activation = F.sigmoid
        case "tanh":
            activation = F.tanh
        # Softmax variants
        case "softmax":
            activation = F.softmax
        case "log_softmax":
            activation = F.log_softmax
        case "gumbel_softmax":
            activation = F.gumbel_softmax
        # Gated/Modern
        case "gelu":
            activation = F.gelu
        case "silu":
            activation = F.silu
        case "mish":
            activation = F.mish
        # No activation
        case None:
            activation = None
        case _:
            raise ValueError(f"Unknown activation function: {layer_config['activation']}")
        
    return activation

def load_layer_type(layer_config):
    layer = None
    batch_norm = None
    
    match layer_config["type"]:
        case "Conv2d":
            layer = nn.Conv2d(
                in_channels=layer_config.get("in_channels"),
                out_channels=layer_config.get("out_channels"),
                kernel_size=layer_config.get("kernel_size"),
                stride=layer_config.get("stride", 1),
                padding=layer_config.get("padding", 0)
            )
            # Create appropriate batch norm if required
            if layer_config.get("batch_norm", False):
                batch_norm = nn.BatchNorm2d(layer_config.get("out_channels"))
        
        case "MaxPool2d" | "max_pool2d":
            layer = nn.MaxPool2d(
                kernel_size=layer_config.get("kernel_size"),
                stride=layer_config.get("stride"),
                padding=layer_config.get("padding", 0)
            )
            # Batch norm not applicable to pooling layers
            batch_norm = None
        
        case "AvgPool2d" | "avg_pool2d":
            layer = nn.AvgPool2d(
                kernel_size=layer_config.get("kernel_size"),
                stride=layer_config.get("stride"),
                padding=layer_config.get("padding", 0)
            )
            # Batch norm not applicable to pooling layers
            batch_norm = None
        
        case "AdaptiveAvgPool2d" | "adaptive_avg_pool2d":
            layer = nn.AdaptiveAvgPool2d(
                output_size=layer_config.get("output_size")
            )
            batch_norm = None
        
        case "AdaptiveMaxPool2d" | "adaptive_max_pool2d":
            layer = nn.AdaptiveMaxPool2d(
                output_size=layer_config.get("output_size")
            )
            batch_norm = None
        
        case "Linear" | "linear":
            layer = nn.Linear(
                in_features=layer_config.get("in_features"),
                out_features=layer_config.get("out_features")
            )
            # Create appropriate batch norm if required
            if layer_config.get("batch_norm", False):
                batch_norm = nn.BatchNorm1d(layer_config.get("out_features"))
        
        case "Dropout" | "dropout":
            layer = nn.Dropout(
                p=layer_config.get("p", 0.5)
            )
            # Batch norm not applicable to dropout
            batch_norm = None
        
        case _:
            raise ValueError(f"Unknown layer type: {layer_config['type']}")
        
    return layer, batch_norm