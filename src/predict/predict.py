"""
src/predict/predict.py

Module to generate predictions using a trained model.
"""

import torch
import torch.nn.functional as F

# Config
from src.config.config_file import DEVICE

def predict(model, dataloader, device=DEVICE):
    """
    Perform inference on new data using the trained model.

    Args:
        model: Trained torch model.
        dataloader: DataLoader with test or unseen data. May yield (inputs,) or (inputs, labels).
        device: 'cuda' or 'cpu'.

    Returns:
        predictions: List of predicted class indices for each sample.
    """
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Desempaqueta inputs (y descarta labels si vienen)
            if isinstance(batch, (list, tuple)) and len(batch) > 1:
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            # Si el modelo devolviera una tupla/lista, toma el primer elemento como logits
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # Softmax + argmax para obtener índices de clase
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            predictions.extend(preds.cpu().tolist())

    return predictions
