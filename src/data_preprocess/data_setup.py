import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """

    # use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform)
    test_data = datasets.ImageFolder(test_dir, transform)

    # get class names
    class_names = train_data.classes

    # turn images into data loaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names


# example implementation

# Import data_setup.py
# from src import data_setup

# # Create train/test dataloader and get class names as a list
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
