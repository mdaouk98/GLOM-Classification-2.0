import logging
import numpy as np
from h5py import File
from typing import Any, Dict, List, Optional, Tuple
from main_utils.train_utils import DataLoadingError

def load_hdf5_data(
    config: Any
) -> Tuple[
    np.ndarray,  # all_indices
    np.ndarray,  # all_images
    np.ndarray,  # all_labels
    np.ndarray,  # all_wsis
    np.ndarray,  # all_stains
    np.ndarray,  # all_scanners
    int          # total_samples
]:
    """
    Load indices, images, labels, and metadata from the HDF5 dataset.

    Args:
        config (Any): Configuration object with the HDF5 file path.

    Returns:
        all_indices: Numpy array of indices.
        all_images:  Numpy array of all images (shape: N x H x W x C).
        all_labels:  Numpy array of labels.
        all_wsis:    Numpy array of WSI identifiers.
        all_stains:  Numpy array of stain codes.
        all_scanners:Numpy array of scanner codes.
        total_samples: Total number of samples.
    """
    try:
        with File(config.paths.hdf5_path, 'r') as f:
            # verify all expected datasets are present
            expected = ['images', 'labels', 'wsis', 'stains', 'scanners']
            missing = [d for d in expected if d not in f]
            if missing:
                raise DataLoadingError(
                    "HDF5 file is missing datasets: " + ", ".join(missing)
                )

            total_samples = len(f['labels'])
            all_indices = np.arange(total_samples)

            # load everything into memory (if that’s what you want)
            all_images   = f['images'][:]    # shape: (N, H, W, C)
            all_labels   = f['labels'][:]
            all_wsis     = f['wsis'][:]
            all_stains   = f['stains'][:]
            all_scanners = f['scanners'][:]

            logging.info(f"[Data Loading] Loaded {total_samples} samples with images, labels, wsis, stains, scanners.")

    except FileNotFoundError as fnfe:
        logging.error(f"[Data Loading] HDF5 file not found at path: {config.paths.hdf5_path}")
        raise fnfe
    except Exception as e:
        logging.error(f"[Data Loading] Error reading HDF5 file: {e}")
        raise DataLoadingError(f"Error reading HDF5 file: {e}") from e

    return (
        all_indices,
        all_images,
        all_labels,
        all_wsis,
        all_stains,
        all_scanners,
        total_samples
    )
