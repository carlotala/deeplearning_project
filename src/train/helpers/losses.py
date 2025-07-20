"""
src/train/helpers/losses.py

Module to define and retrieve loss functions for training models.
"""
import torch.nn as nn


def get_criterion(name="cross_entropy"):
    """
    Get the loss function for training.
    
    Parameters
    ----------
    name : str
        Name of the loss function.

    Returns
    -------
    torch.nn.Module
        The loss function for training.
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss {name} not implemented.")
