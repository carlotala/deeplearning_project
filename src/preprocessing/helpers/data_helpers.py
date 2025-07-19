"""
src/preprocessing/helpers/data_helpers.py

Helper functions for data preparation.
Includes:
- dataset splitting
- transform definitions
"""

from torchvision import transforms


def split_dataset(input_dir, output_dir, val_size=0.1, test_size=0.1, seed=42):
    """
    Split the dataset from `input_dir` (class folders) into
    train/val/test folders inside `output_dir`.

    - Should preserve ImageFolder structure.
    - You can use shutil to copy or symlink.
    - Use sklearn.model_selection.train_test_split.
    """
    # TODO: Implement or call splitting logic
    pass


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
    # TODO: Customize resizing, augmentation, and normalization as needed
    if phase == "train":
        return transforms.Compose([
            # e.g., transforms.RandomHorizontalFlip(),
            #       transforms.RandomRotation(10),
            #       transforms.Resize((128, 128)),
            #       transforms.ToTensor(),
            #       transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            # e.g., transforms.Resize((128, 128)),
            #       transforms.ToTensor(),
            #       transforms.Normalize(mean, std)
        ])
