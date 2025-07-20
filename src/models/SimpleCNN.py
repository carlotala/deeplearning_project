"""
src/models/SimpleCNN.py

Implementation of a simple, flexible CNN for image classification.
"""
import torch
import torch.nn as nn

from src.config.config_file import NUM_CLASSES

class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification.
    Architecture:
      - 3 convolutional blocks (Conv2d → BatchNorm (optional) → ReLU → MaxPool)
      - Dropout after last pooling (optional)
      - Linear classifier head
    Parameters can be customized for experimentation.
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        conv_channels: tuple = (32, 64, 128),
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.5,
        linear_hidden: int = 256,
        image_size: tuple = (224, 224),
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        h, w = image_size
        for c_out in conv_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            c_in = c_out
            h, w = h // 2, w // 2
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        flatten_dim = c_in * h * w
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, linear_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(linear_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
