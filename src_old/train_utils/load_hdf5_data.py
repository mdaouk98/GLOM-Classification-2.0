# src/train_utils/load_hdf5_data.py

import logging
import numpy as np
from h5py import File
from typing import Any, Dict, List, Optional, Tuple

def load_hdf5_data(config: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load indices and labels from the HDF5 dataset.

    Args:
        config (Any): Configuration object with the HDF5 file path.

    Returns:
        Tuple containing:
          - all_indices: Numpy array of indices.
          - all_labels: Numpy array of labels.
          - total_samples: Total number of samples.
    """
    try:
        with File(config.paths.hdf5_path, 'r') as f:
            if 'images' not in f or 'labels' not in f:
                raise DataLoadingError("HDF5 file must contain 'images' and 'labels' datasets.")
            total_samples = len(f['labels'])
            all_indices = np.arange(total_samples)
            all_labels = f['labels'][:]
            logging.info(f"[Data Loading] Loaded {total_samples} samples from HDF5.")
    except FileNotFoundError as fnfe:
        logging.error(f"[Data Loading] HDF5 file not found at path: {config.paths.hdf5_path}")
        raise fnfe
    except Exception as e:
        logging.error(f"[Data Loading] Error reading HDF5 file: {e}")
        raise DataLoadingError(f"Error reading HDF5 file: {e}") from e

    return all_indices, all_labels, total_samples