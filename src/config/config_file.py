"""
src/config/config_file.py

This file contains configuration variables used in the project.
It includes settings such as labels, class names, training hyperparameters,
paths, and model-specific parameters.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "split"

IMAGE_SIZE = (224, 224)  # or (128, 128)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean for ImageNet normalization
IMAGENET_STD = [0.229, 0.224, 0.225]  # Std for ImageNet normalization

NUM_CLASSES = 12  # Number of classes in the dataset

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model save path (can be used in train/predict)
MODEL_SAVE_PATH = "outputs/models/best_model.pt"
