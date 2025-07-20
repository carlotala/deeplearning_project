"""
src/predict/predict.py

Module to generate predictions using a trained model.
"""

import torch
import torch.nn.functional as F

# Config
from src.config.config_file import DEVICE

def predict(model, dataloader, device=DEVICE, return_probs=False):
    """
    Returns:
        predictions: List of predicted class indices.
        (opcional) probabilities: List of probability vectors (lista de listas).
    """
    model.eval()
    model.to(device)

    predictions = []
    all_probs = [] if return_probs else None

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            predictions.extend(preds.cpu().tolist())
            if return_probs:
                all_probs.extend(probs.cpu().tolist())

    return (predictions, all_probs) if return_probs else predictions

