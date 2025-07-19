"""
src/config/config_file.py

This file contains configuration variables used in the project.
It includes settings such as labels, class names, training hyperparameters,
paths, and model-specific parameters.
"""

# Target label (used for decoding or evaluation)
TARGET_LABEL = "waste_type"

# Class names (used in plotting, decoding predictions, etc.)
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Columns for structured data (if used)
NUMERIC_COLUMNS = [
    # e.g., "weight", "brightness_mean"
]

CATEGORICAL_COLUMNS = [
    # e.g., "source_location"
]

# Paths — optionally used by preprocessing or main pipeline
DATA_DIR = "data/"
RAW_DATA_PATH = "data/raw"
SPLIT_DATA_DIR = "data/split"
TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"
TEST_DIR = "data/split/test"

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image properties
IMAGE_SIZE = (224, 224)

# Model save path (can be used in train/predict)
MODEL_SAVE_PATH = "outputs/models/best_model.pt"
