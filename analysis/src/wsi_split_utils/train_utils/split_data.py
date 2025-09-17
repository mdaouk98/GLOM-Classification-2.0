# src/wsi_split_utils/train_utils/split_data.py

import logging
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def split_data(
    all_indices: np.ndarray,
    all_labels:  np.ndarray,
    all_wsis:    np.ndarray,
    config:      Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation and test sets.

    Args:
        all_indices (np.ndarray): Array of indices.
        all_labels (np.ndarray): Array of labels.
        all_wsis (np.ndarray): Array of wsis.
        config (Any): Configuration object with split settings.

    Returns:
        Tuple containing:
          - train_val_indices: Indices for training and validation.
          - train_val_labels: Corresponding labels.
          - test_indices: Indices for testing.
          - test_labels: Corresponding test labels.
    """
    try:
        # 1) carve out a test set by WSI
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=config.training.test_size,   # e.g. 0.2
            random_state=config.training.seed
        )
        train_val_idx, test_idx = next(
            gss.split(X=all_indices, y=all_labels, groups=all_wsis)
        )
    
        train_val_indices = all_indices[train_val_idx]
        train_val_labels  = all_labels[train_val_idx]
        test_indices      = all_indices[test_idx]
        test_labels       = all_labels[test_idx]
    
        

    except Exception as e:
        logging.error(f"[Data Splitting] Error during splitting data: {e}")
        raise

    if len(set(train_val_indices).intersection(test_indices)) != 0:
        error_msg = "Training and test sets overlap!"
        logging.error(f"[Data Splitting] {error_msg}")
        raise ValueError(error_msg)

    logging.info(f"[Data Splitting] Total images: {len(all_indices)}")
    logging.info(f"[Data Splitting] Train/Val  images: {len(train_val_indices)} (from {len(np.unique(all_wsis[train_val_idx]))} WSIs)")
    logging.info(f"[Data Splitting] Test       images: {len(test_indices)} (from {len(np.unique(all_wsis[test_idx]))} WSIs)")
    return train_val_indices, train_val_labels, test_indices, test_labels
