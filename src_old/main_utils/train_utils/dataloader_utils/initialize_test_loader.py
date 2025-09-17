# src/train_utils/dataloader_utils/initialize_test_loader.py

import logging
from torch.utils.data import DataLoader
from main_utils.datasets_utils import HDF5Dataset
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def initialize_test_loader(config: Any, test_indices: np.ndarray, transform: Optional[Any],
                           processor: Optional[Any], for_vision: bool) -> DataLoader:
    """
    Create test dataset and DataLoader.

    Args:
        config (Any): Configuration object.
        test_indices (np.ndarray): Indices for test data.
        transform (Optional[Any]): Transformation pipeline.
        processor (Optional[Any]): Vision processor if applicable.
        for_vision (bool): Whether using a vision model.

    Returns:
        DataLoader: The test DataLoader.
    """
    test_dataset = HDF5Dataset(
        hdf5_file=config.paths.hdf5_path,
        indices=test_indices,
        transform=transform,
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=config.HDF5Dataset.cache_in_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=config.data.prefetch_factor
    )
    logging.info(f"[TestLoader] Created test DataLoader with {len(test_indices)} samples.")
    return test_loader
