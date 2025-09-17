# src/mc_utils/mc_clustering/mc_clustering_main.py

import numpy as np
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from main_utils.datasets_utils import HDF5Dataset

def create_dataloader(indices: np.ndarray,
                      transform: Optional[Any],
                      processor: Optional[Any],
                      for_vision: bool,
                      config: Any,
                      cache_in_memory: Any = None) -> DataLoader:
    """
    Creates a DataLoader for a given set of indices.
    """
    if cache_in_memory == False:
        cache_in_memory_activation = False
    elif cache_in_memory == True:
        cache_in_memory_activation = True
    else:
        cache_in_memory_activation = config.HDF5Dataset.cache_in_memory
    
    dataset = HDF5Dataset(
        hdf5_file=config.paths.hdf5_path,
        indices=indices,
        transform=transform,
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=cache_in_memory_activation
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers= 20, #config.data.num_workers
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor= 10  #config.data.prefetch_factor
    )
    return dataloader

