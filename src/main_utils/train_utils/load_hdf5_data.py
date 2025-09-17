# src/main_utils/train_utils/load_hdf5_data.py

import logging
import numpy as np
import h5py
from h5py import File
from typing import Any, Dict, Tuple, Union
from main_utils.train_utils import DataLoadingError

def load_hdf5_data(
    config: Any
) -> Tuple[
    np.ndarray,
    Union[np.ndarray, Dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int
]:
    """
    Load indices and labels (plus metadata) from an HDF5 dataset.

    Expects the file to contain:
      - 'images' dataset of shape (N, H, W, C)
      - 'labels' group containing:
          * one dataset per head (1D arrays)
          * 'wsi', 'stain', 'scanner' metadata arrays (1D)

    Single-head vs. multi-head:
      - If config.model.multihead is False, reads only config.model.label.
      - If True, reads each name in config.multihead.labels.

    Returns:
        all_indices (np.ndarray): [0, 1, ..., N-1]
        all_labels  (np.ndarray or dict): 1D label array or dict of arrays per head
        all_wsis    (np.ndarray): WSI identifiers per sample
        all_stains  (np.ndarray): Stain identifiers per sample
        all_scanners(np.ndarray): Scanner identifiers per sample
        total_samples (int): Number of samples N
    """
    try:
        # 1) Open HDF5 file in read-only mode
        with File(config.paths.hdf5_path, 'r') as f:
            # 2) Validate required groups
            if 'images' not in f or 'labels' not in f:
                raise DataLoadingError("HDF5 must contain both 'images' and 'labels' groups.")

            # 3) Determine total number of samples
            total_samples = f['images'].shape[0]
            all_indices = np.arange(total_samples)

            lbl_grp = f['labels']  # Group of all label datasets

            # 4) Load labels depending on single- vs. multi-head setting
            if not getattr(config.model, 'multihead', False):
                # Single-head: load one 1D array
                head = config.model.label
                if head not in lbl_grp:
                    raise DataLoadingError(f"Label '{head}' not found under '/labels'.")
                arr = lbl_grp[head][:]
                if arr.ndim != 1:
                    raise DataLoadingError(f"Expected 1D array for '{head}', got shape {arr.shape}")
                all_labels = arr
                logging.info(f"[Data Loading] Loaded {total_samples} labels for single-head '{head}'")
            else:
                # Multi-head: load each named head into a dict
                names = config.multihead.labels
                missing = [n for n in names if n not in lbl_grp]
                if missing:
                    raise DataLoadingError(f"Missing label heads under '/labels': {missing}")
                all_labels = {n: lbl_grp[n][:] for n in names}
                logging.info(f"[Data Loading] Loaded {total_samples} samples for multi-head: {names}")

            # 5) Load metadata arrays (all 1D of length N)
            all_wsis     = lbl_grp['wsi'][:]
            all_stains   = lbl_grp['stain'][:]
            all_scanners = lbl_grp['scanner'][:]

    except FileNotFoundError:
        logging.error(f"[Data Loading] File not found: {config.paths.hdf5_path}")
        raise
    except DataLoadingError:
        # Re-raise our custom error after logging
        raise
    except Exception as e:
        logging.error(f"[Data Loading] Error reading HDF5: {e}", exc_info=True)
        raise DataLoadingError(f"Error reading HDF5 file: {e}") from e

    # 6) Return all loaded arrays and sample count
    return (
        all_indices,
        all_labels,
        all_wsis,
        all_stains,
        all_scanners,
        total_samples
    )
