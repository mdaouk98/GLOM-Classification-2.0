# src/train_utils/dataloader_utils/initialize_test_loader.py

import logging
from torch.utils.data import DataLoader
from main_utils.datasets_utils import HDF5Dataset
from typing import Any, Optional
import numpy as np

def initialize_test_loader(
    config: Any,
    test_indices: np.ndarray,
    transform: Optional[Any],
    processor: Optional[Any],
    for_vision: bool
) -> DataLoader:
    """
    Create an HDF5Dataset for test data and wrap it in a PyTorch DataLoader.

    Args:
        config (Any): Configuration object with paths and data settings.
        test_indices (np.ndarray): Array of indices to include in the test set.
        transform (Optional[Any]): Image transform pipeline (e.g., Albumentations).
        processor (Optional[Any]): HF vision processor for transformer models.
        for_vision (bool): If True, uses 'processor'; otherwise uses 'transform'.

    Returns:
        DataLoader: Configured DataLoader for the test dataset.
    """
    # 1) Instantiate the HDF5Dataset for test data
    test_dataset = HDF5Dataset(
        hdf5_file=config.paths.hdf5_path,
        indices=test_indices,
        transform=transform,
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=config.HDF5Dataset.cache_in_memory
    )

    # 2) Configure DataLoader parameters
    batch_size = config.training.batch_size
    num_workers = config.data.num_workers
    prefetch = config.data.prefetch_factor

    # 3) Create the DataLoader (no shuffling for test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=prefetch
    )

    # 4) Log loader creation details
    logging.info(
        f"[TestLoader] Created test DataLoader with {len(test_indices)} samples, "
        f"batch_size={batch_size}."
    )

    return test_loader

