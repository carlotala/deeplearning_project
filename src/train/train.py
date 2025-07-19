"""
src/train/train.py

Module to train a model on the training set.
"""

# Imports
import torch

# Config and utils
from src.config.config_file import DEVICE, EPOCHS

def train(model, train_loader, criterion, optimizer, device=DEVICE, epochs=EPOCHS):
    """
    Main training loop.

    Args:
        model: Torch model instance.
        train_loader: DataLoader with training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU.
        epochs: Number of epochs.

    Returns:
        Trained model.
    """

    # Set model to training mode
    model.train()

    for epoch in range(epochs):
        # Loop over batches
        for inputs, labels in train_loader:
            # Move data to device
            # Forward pass
            # Compute loss
            # Backward pass and optimizer step
            pass

        # Optionally print loss/accuracy per epoch

    return model
