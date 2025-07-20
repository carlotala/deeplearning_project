"""
src/main.py

This module orchestrates the classification pipeline using only the model type as input.
Uses the preprocessing pipeline to prepare data.

INPUTS:
    · model_type: One of 'simple_cnn', 'vgg', 'resnet'

OUTPUTS:
    · Trained model checkpoint saved as 'outputs/models/model_<model_type>.pth'
    · Logged training and evaluation metrics
"""
import os
import torch
from src.config.config_file import (
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    OUTPUT_MODELS_DIR
)
from src.preprocessing.preprocessing import data_preparation
from src.models.SimpleCNN import SimpleCNN
from src.models.VGGTransferModel import VGGTransferModel
from src.models.ResNetTransferModel import ResNetTransferModel
from src.train.train import train
from src.predict.predict import predict
from src.utils.logger import logger_all as logger
from src.evaluate.analyze import analyze_classification


def main(model_type: str) -> None:
    """
    Load and preprocess data, train the specified model, evaluate, and save the checkpoint.

    Parameters
    ----------
    model_type : {'simple_cnn', 'vgg', 'resnet'}
        Model architecture to instantiate and run.

    Returns
    -------
    None
    """
    try:
        logger.info(f"Starting pipeline for model type: {model_type}")

        # 1. Prepare data
        logger.info("Preparing data...")
        dataloaders = data_preparation(
            input_dir=RAW_DATA_DIR,
            output_dir=SPLIT_DATA_DIR,
            batch_size=BATCH_SIZE,
        )
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        num_classes = len(train_loader.dataset.classes)
        logger.info(
            f"Loaded {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples across {num_classes} classes."
        )

        # 2. Instantiate model
        logger.info(f"Instantiating model: {model_type}")
        if model_type == "simple_cnn":
            model = SimpleCNN(num_classes)
        elif model_type == "vgg":
            model = VGGTransferModel(num_classes)
        elif model_type == "resnet":
            model = ResNetTransferModel(num_classes)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return

        # 3. Train model
        logger.info("Starting training...")
        trained_model = train(
            model,
            train_loader,
            val_loader,
            loss_name="cross_entropy",
            optimizer_name="adam",
            lr=LEARNING_RATE,
            device=DEVICE,
            epochs=EPOCHS,
        )
        logger.info("Training completed.")

        # 4. Generate predictions on test set
        logger.info("Generating predictions on test set...")
        test_preds, test_probs = predict(
            trained_model,
            test_loader,
            device=DEVICE,
            return_probs=True
        )
        logger.info(f"Generated {len(test_preds)} test predictions.")

        # 6. Save checkpoint
        os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_MODELS_DIR, f"model_{model_type}.pth")
        torch.save(trained_model.state_dict(), output_path)
        logger.info(f"Saved model checkpoint: {output_path}")

        # 7. Analyze results
        logger.info("Analyzing classification results...")
        class_names = [test_loader.dataset.classes[i] for i in range(len(test_loader.dataset.classes))]
        metrics = analyze_classification(test_loader.dataset, test_preds, test_probs, class_names)

        logger.info("Pipeline finished successfully.")

        return metrics
    except Exception:
        logger.error("Pipeline failed with an exception.", exc_info=True)
        raise
