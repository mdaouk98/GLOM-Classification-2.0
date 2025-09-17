# src/train_utils/split_data.py

import logging
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def split_data(all_indices: np.ndarray, all_labels: np.ndarray, config: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation and test sets.

    Args:
        all_indices (np.ndarray): Array of indices.
        all_labels (np.ndarray): Array of labels.
        config (Any): Configuration object with split settings.

    Returns:
        Tuple containing:
          - train_val_indices: Indices for training and validation.
          - train_val_labels: Corresponding labels.
          - test_indices: Indices for testing.
          - test_labels: Corresponding test labels.
    """
    try:
        train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
            all_indices,
            all_labels,
            test_size=config.training.test_size,
            random_state=config.training.seed,
            stratify=all_labels
        )
    except ValueError as e:
        logging.error(f"[Data Splitting] Error during splitting data: {e}")
        raise

    if len(set(train_val_indices).intersection(test_indices)) != 0:
        error_msg = "Training and test sets overlap!"
        logging.error(f"[Data Splitting] {error_msg}")
        raise ValueError(error_msg)

    logging.info(f"[Data Splitting] Total samples: {len(all_indices)}")
    logging.info(f"[Data Splitting] Training/Validation samples: {len(train_val_indices)}")
    logging.info(f"[Data Splitting] Testing samples: {len(test_indices)}")
    return train_val_indices, train_val_labels, test_indices, test_labels
