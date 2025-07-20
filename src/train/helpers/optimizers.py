"""
src/train/helpers/optimizers.py

Module to define and retrieve optimizers for training models.
"""
import torch.optim as optim


def get_optimizer(model, lr=1e-3, name="adam"):
    """
    Get the optimizer for training.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to optimize.
    lr : float
        Learning rate.
    name : str
        Name of the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer for the model.
    """
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {name} not implemented.")
