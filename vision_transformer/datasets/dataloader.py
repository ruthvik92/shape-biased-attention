# Define your custom audio dataset class
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from yacs.config import CfgNode
import logging

def make_imagenet_data_loaders(cfg, device="cuda", val_shuffle=False):
    """
    Creates DataLoaders for ImageNet training and validation sets.

    Args:
        cfg (CfgNode): Configuration object containing dataset and solver settings.
        device (str): Device to load data onto ('cuda' or 'cpu').
        val_shuffle (bool): Whether to shuffle the validation dataset (default: False).

    Returns:
        tuple: (train_loader, val_loader)
    """
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_workers = cfg.DATALOADER.NUM_WORKERS
    imagenet_root = cfg.DATASET.ROOT_DIR  # Path to ImageNet dataset

    size = cfg.DATASET.SIZE  # Standard ImageNet input size

    # Training data transformations
    transform_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Apply RandAugment if enabled in config
    if hasattr(cfg.DATASET, "USE_RANDAUG") and cfg.DATASET.USE_RANDAUG:
        N = cfg.DATASET.RANDAUG_N if hasattr(cfg.DATASET, "RANDAUG_N") else 2
        M = cfg.DATASET.RANDAUG_M if hasattr(cfg.DATASET, "RANDAUG_M") else 14
        transform_train.transforms.insert(0, transforms.RandAugment(num_ops=N, magnitude=M))

    # Validation data transformations (no augmentations)
    transform_val = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Load ImageNet datasets
    trainset = ImageNet(root=imagenet_root, split="train", transform=transform_train)
    valset = ImageNet(root=imagenet_root, split="val", transform=transform_val)

    logging.info(f"Loaded ImageNet dataset: {len(trainset)} train samples, {len(valset)} validation samples.")

    # Create DataLoaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle training data
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=val_shuffle,  # Optional shuffle for validation
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )

    return train_loader, val_loader

if __name__ == "__main__":
    cfg = {
    "DATASET": {
        "ROOT_DIR": "/home/ruthvik/data/imagenet",
        "USE_RANDAUG": True,  # Enable RandAugment
        "RANDAUG_N": 2,       # Number of augmentations
        "RANDAUG_M": 14,       # Magnitude of augmentation
        "SIZE": 224
    },
    "SOLVER": {
        "BATCH_SIZE": 128  # Set batch size
    },
    "DATALOADER": {
        "NUM_WORKERS": 4  # Number of workers
    }
    }

    train_loader, val_loader = make_imagenet_data_loaders(cfg)

