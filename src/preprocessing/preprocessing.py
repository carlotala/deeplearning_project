"""
src/preprocessing/preprocessing.py

Main function to prepare image data for classification using PyTorch.
- Splits data into train/val/test folders (if needed)
- Applies appropriate transforms per phase
- Loads data using ImageFolder
- Returns DataLoaders
"""

from torchvision import datasets
from torch.utils.data import DataLoader
from src.preprocessing.helpers.data_helpers import split_dataset, get_transforms


def data_preparation(input_dir, output_dir, batch_size=32):
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
    split_dataset(input_dir, output_dir)
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        # TODO: Check that folder structure exists: output_dir/phase/
        # Use torchvision.datasets.ImageFolder for automatic labeling by folder name
        dataset = datasets.ImageFolder(
            root=f"{output_dir}/{phase}",
            transform=get_transforms(phase)
        )

        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == 'train')  # Shuffle only for training
        )

    return dataloaders
