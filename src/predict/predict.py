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
    Perform inference on new data using the trained classification model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch classification model.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing input batches. Each batch can be a tensor of inputs or a tuple/list
        where the first element is inputs (labels, if present, are ignored).
    device : str or torch.device, optional (default=DEVICE)
        Device to run inference on ('cpu' or 'cuda').
    return_probs : bool, optional (default=False)
        If True, also return the probability distributions for each prediction.

    Returns
    -------
    predictions : List[int]
        List of predicted class indices for each sample.
    probabilities : List[List[float]], optional
        Probability vectors for each sample if return_probs is True, each summing to 1 across classes.
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

