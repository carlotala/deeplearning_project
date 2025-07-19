"""
src/predict/predict.py

Module to generate predictions using a trained model.
"""

# Imports
import torch

# Config and utils
from src.config.config_file import DEVICE

def predict(model, dataloader, device=DEVICE):
    """
    Perform inference on new data using the trained model.

    Args:
        model: Trained torch model.
        dataloader: DataLoader with test or unseen data.
        device: CPU or GPU.

    Returns:
        predictions: List or tensor of model predictions.
    """
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Forward pass
            # Apply softmax / argmax if needed
            # Store predictions
            pass

    return predictions
