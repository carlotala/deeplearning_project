"""
src/train/train.py

Module to train a model on the training set.
"""
from src.config.config_file import DEVICE, EPOCHS
from src.utils.logger import logger_all as logger
from src.train.helpers.losses import get_criterion
from src.train.helpers.optimizers import get_optimizer
from src.evaluate.evaluate import evaluate


def train(model, train_loader, val_loader, loss_name="cross_entropy", optimizer_name="adam", lr=1e-3, device=DEVICE, epochs=EPOCHS):
    """
    Train a model using specified loss and optimizer, with validation after each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    loss_name : str
        Name of the loss function.
    optimizer_name : str
        Name of the optimizer.
    lr : float
        Learning rate.
    device : str
        'cuda' or 'cpu'.
    epochs : int
        Number of training epochs.

    Returns
    -------
    torch.nn.Module
        Trained model.
    """
    model.to(device)
    criterion = get_criterion(loss_name)
    optimizer = get_optimizer(model, lr=lr, name=optimizer_name)

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

    logger.info("Training completed.")
    return model
