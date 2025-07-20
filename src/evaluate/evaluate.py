"""
src/evaluate/evaluate.py

Module to evaluate a trained model using a test set.
"""
import torch
from src.config.config_file import DEVICE


def evaluate(model, val_loader, criterion, device=DEVICE):
    """
    Evaluates the model on a validation or test set.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    val_loader : DataLoader
        Validation or test data loader.
    criterion : loss function
        The same loss used during training.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    tuple
        Average loss and accuracy percentage.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100

    return avg_loss, accuracy
