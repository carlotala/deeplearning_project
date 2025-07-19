"""
src/evaluate/evaluate.py

Module to evaluate a trained model using a test set.
"""

# Imports
import torch
from sklearn.metrics import classification_report

# Config
from src.config.config_file import DEVICE

def evaluate(model, dataloader, criterion, device=DEVICE):
    """
    Evaluate model performance on a labeled dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader with test/validation data.
        criterion: Loss function.
        device: CPU or GPU.

    Returns:
        Dictionary of evaluation results (e.g., loss, accuracy, metrics).
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move data to device
            # Forward pass
            # Compute loss
            # Accumulate predictions and true labels
            pass

    # Compute metrics, print or return them
    # Optionally generate and save plots

    return {
        "loss": total_loss,
        "other_metrics": "to be defined"
    }
