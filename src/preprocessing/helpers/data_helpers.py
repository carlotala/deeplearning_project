"""
src/preprocessing/helpers/data_helpers.py

Helper functions for data preparation.
Includes:
- dataset splitting
- transform definitions
"""
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import WeightedRandomSampler

from src.utils.logger import logger_all as logger
from src.config.config_file import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RAW_DATA_DIR, SPLIT_DATA_DIR


def get_raw_dataset(input_dir, transform=None):
    """
    Returns the raw dataset from the input directory.
    
    Parameters
    ----------
    input_dir : str
        Path to the dataset with class folders.
    transform : torchvision.transforms.Compose, optional
        Transformations to apply to the images.

    Returns
    -------
    torchvision.datasets.ImageFolder
        Dataset object containing images and labels.
    """
    return datasets.ImageFolder(root=input_dir, transform=transform)


def split_dataset(input_dir=RAW_DATA_DIR, output_dir=SPLIT_DATA_DIR, val_pct=0.1, test_pct=0.1, seed=42):
    """
    Splits a dataset into train, validation, and test sets by copying files into new folders.

    Parameters
    ----------
    input_dir : str
        Path to the raw dataset folder (organized by class subfolders).
    output_dir : str
        Path to the output folder where train/val/test splits will be stored.
    val_pct : float
        Proportion of data to use for validation.
    test_pct : float
        Proportion of data to use for testing.
    seed : int
        Random seed for reproducibility.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if (output_dir / "train").exists():
        logger.info(f"Found existing split in {output_dir}, skipping split.")
        return
    
    train_pct = 1 - val_pct - test_pct  # Adjust train_pct to fill the rest

    logger.info("Splitting dataset...")

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.*"))
        class_name = class_dir.name

        # First split into train and temp
        train_imgs, temp_imgs = train_test_split(
            images, train_size=train_pct, random_state=seed
        )

        # Then split temp into val and test
        relative_val_pct = val_pct / (val_pct + test_pct)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=relative_val_pct, random_state=seed
        )

        # Organize outputs
        split_map = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        for split, split_imgs in split_map.items():
            split_class_dir = output_dir / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy(img_path, split_class_dir)

    logger.info("Dataset successfully split into train/val/test.")


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transforms(phase):
    """
    Returns torchvision transforms for a given phase.
    
    Parameters
    ----------
    phase : str
        One of 'train', 'val', or 'test'.

    Returns
    -------
    torchvision.transforms.Compose
        Transformation pipeline to apply to the images.
    """
    if phase == "train":
        return transforms.Compose([
            convert_to_rgb,
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            convert_to_rgb,
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_train_sampler(dataset):
    # Get class counts
    class_counts = np.bincount(dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
