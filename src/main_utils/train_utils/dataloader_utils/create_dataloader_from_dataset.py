# src/train_utils/dataloader_utils/create_dataloader_from_dataset.py

import logging
from torch.utils.data import DataLoader, Subset
from main_utils.datasets_utils import HDF5Dataset, TransformWrapper
from typing import Any, Optional
import numpy as np

def create_dataloader_from_dataset(
    dataset: HDF5Dataset,
    indices: np.ndarray,
    config: Any,
    loader_type: str,
    dataset_transform: Any,
    processor: Optional[Any],
    for_vision: bool
) -> DataLoader:
    """
    Build a DataLoader for a subset of an HDF5Dataset, applying transforms or
    vision processors as needed.

    Args:
        dataset (HDF5Dataset): Already-opened dataset (optionally cached in memory).
        indices (np.ndarray): Array of sample indices to include in this loader.
        config (Any): Config object with data/loader parameters.
        loader_type (str): One of 'train', 'validate', or 'test'.
        dataset_transform (Any): Albumentations or torchvision transform to apply.
        processor (Optional[Any]): HF vision processor (if using a Transformer).
        for_vision (bool): Whether to use the HF processor instead of transforms.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # 1) Create a Subset view of the base dataset using the provided indices
    subset = Subset(dataset, indices)

    # 2) Wrap the subset with TransformWrapper to apply transforms/processing
    wrapped_dataset = TransformWrapper(
        base_dataset=subset,
        transform=dataset_transform,
        processor=processor,
        for_vision=for_vision
    )

    # 3) Common DataLoader settings
    batch_size = config.training.batch_size
    num_workers = config.data.num_workers
    prefetch = config.data.prefetch_factor

    # 4) Determine whether to shuffle:
    #    - Always shuffle for train/validate (if transform provided)
    #    - Never shuffle for test
    if loader_type.lower() == 'test':
        shuffle = False
    else:
        shuffle = bool(dataset_transform)  # shuffle only if applying augmentations

    # 5) Build the DataLoader
    loader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=prefetch
    )

    # 6) Log the creation for debugging
    logging.info(f"[{loader_type.capitalize()} DataLoader] Created with {len(indices)} samples, "
                 f"batch size={batch_size}, shuffle={shuffle}.")

    return loader
