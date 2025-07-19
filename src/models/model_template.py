"""
src/models/model_template.py

Template file for defining a PyTorch model architecture.
Use this as a starting point for implementing custom CNNs or transfer learning models.
"""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # TODO: Define model layers here
        # Example:
        # self.conv1 = nn.Conv2d(...)
        # self.fc = nn.Linear(...)

        pass

    def forward(self, x):
        # TODO: Define forward pass
        # x = self.conv1(x)
        # x = F.relu(x)
        # ...
        # return self.fc(x)

        pass
