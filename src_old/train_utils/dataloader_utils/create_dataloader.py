# src/train_utils/dataloader_utils/create_dataloader.py

import logging
from os import makedirs
from os.path import join
from torch.utils.data import DataLoader
from datasets import HDF5Dataset
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def create_dataloader(hdf5_path: str, indices: np.ndarray, transform: Optional[Any], processor: Optional[Any],
                      for_vision: bool, config: Any) -> DataLoader:
    """
    Initialize the dataset and corresponding DataLoader.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        indices (np.ndarray): Indices to load.
        transform (Optional[Any]): Transformation pipeline.
        processor (Optional[Any]): Vision processor if applicable.
        for_vision (bool): Whether using a vision model.
        config (Any): Configuration object.

    Returns:
        DataLoader: The PyTorch DataLoader for the dataset.
    """
    dataset = HDF5Dataset(
        hdf5_file=hdf5_path,
        indices=indices,
        transform=transform,
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=config.HDF5Dataset.cache_in_memory
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True if transform is not None else False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=config.data.prefetch_factor
    )
    logging.info(f"[DataLoader] Created DataLoader with {len(indices)} samples.")
    return loader