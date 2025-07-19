"""
src/preprocessing/preprocessing.py

Main function to prepare image data for classification using PyTorch.
- Splits data into train/val/test folders (if needed)
- Applies appropriate transforms per phase
- Loads data using ImageFolder
- Returns DataLoaders
"""
import os
from torchvision import datasets
from torch.utils.data import DataLoader

from src.preprocessing.helpers.data_helpers import split_dataset, get_transforms, get_train_sampler
from src.config.config_file import RAW_DATA_DIR, SPLIT_DATA_DIR
from src.utils.logger import logger_all as logger


def data_preparation(input_dir=RAW_DATA_DIR, output_dir=SPLIT_DATA_DIR, batch_size=32):
    """
    Prepares PyTorch DataLoaders for training, validation, and test datasets.

    Parameters
    ----------
    input_dir : str 
        Path to dataset with class folders (not yet split).
    output_dir : str
        Path where train/val/test folders will be created.
    batch_size : int
        Batch size for DataLoaders.

    Returns
    -------
    dict
        Dictionary with DataLoaders for 'train', 'val', and 'test'.
    """
    logger.info("Starting data preparation pipeline...")
    logger.debug(f"Input directory: {input_dir}")
    logger.debug(f"Output directory (split): {output_dir}")
    logger.debug(f"Batch size: {batch_size}")

    split_dataset(input_dir, output_dir)
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        data_path = os.path.join(output_dir, phase)
        logger.info(f"Loading {phase} data from {data_path}")
        dataset = datasets.ImageFolder(root=data_path, transform=get_transforms(phase))

        if phase == "train":
            sampler = get_train_sampler(dataset)
            dataloaders[phase] = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
            logger.info("Using weighted random sampler to address class imbalance in training data.")
        else:
            dataloaders[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info("Data preparation complete. DataLoaders for train, val, and test are ready.")
    return dataloaders
