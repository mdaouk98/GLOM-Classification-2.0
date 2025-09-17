# src/main_utils/train_utils/split_data.py

import logging
import numpy as np
from typing import Any, Dict, Tuple, Union
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def split_data(
    all_indices: np.ndarray,
    all_labels: Union[np.ndarray, Dict[str, np.ndarray]],
    all_wsis:    np.ndarray,
    config: Any
) -> Tuple[
    np.ndarray,
    Union[np.ndarray, Dict[str, np.ndarray]],
    np.ndarray,
    Union[np.ndarray, Dict[str, np.ndarray]]
]:
    """
    Split dataset into train/validation and test sets, either at the WSI-level
    (no WSI in both sets) or at the patch-level.

    - If config.split_level == "wsi":
        Uses GroupShuffleSplit to hold out a fraction (config.training.test_size) of WSIs.
    - If config.split_level == "patch":
        Uses train_test_split on the flat list of patches.
    - Honors config.model.multihead to split single- vs. multi-head labels.

    Returns:
        train_val_indices, train_val_labels, test_indices, test_labels
    """
    test_size = config.training.test_size
    seed      = config.training.seed
    split_level = config.training.split_level

    try:
        if split_level == "wsi":
            # ----------------------------
            # 1) Group-based split on WSI
            # ----------------------------
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=seed
            )
            
            # ----------------------------
            # 2) Perform split
            # ----------------------------

            if not getattr(config.model, 'multihead', False):
                # Single-head: labels is a 1D array
                train_val_idx, test_idx = next(
                    gss.split(
                        X=all_indices,
                        y=all_labels,
                        groups=all_wsis
                    )
                )
                train_val_indices = all_indices[train_val_idx]
                train_val_labels  = all_labels[train_val_idx]
                test_indices      = all_indices[test_idx]
                test_labels       = all_labels[test_idx]

            else:
                # Multi-head: labels is a dict head -> 1D array
                # Use the first head for stratification
                first_head = next(iter(all_labels))
                train_val_idx, test_idx = next(
                    gss.split(
                        X=all_indices,
                        y=all_labels[first_head],
                        groups=all_wsis
                    )
                )
                train_val_indices = all_indices[train_val_idx]
                test_indices      = all_indices[test_idx]

                train_val_labels = {
                    head: arr[train_val_idx]
                    for head, arr in all_labels.items()
                }
                test_labels = {
                    head: arr[test_idx]
                    for head, arr in all_labels.items()
                }

        elif split_level == "patch":
            # ----------------------------
            # 3) Flat train/test split on patches
            # ----------------------------
            if not getattr(config.model, 'multihead', False):
                # Single-head: split indices + labels together
                train_val_indices, test_indices, \
                train_val_labels,  test_labels = train_test_split(
                    all_indices,
                    all_labels,
                    test_size=test_size,
                    random_state=seed,
                    shuffle=True
                )

            else:
                # Multi-head: split by positional indices, then slice each head
                num_samples = len(all_indices)
                positions = np.arange(num_samples)
                train_pos, test_pos = train_test_split(
                    positions,
                    test_size=test_size,
                    random_state=seed,
                    shuffle=True
                )

                train_val_indices = all_indices[train_pos]
                test_indices      = all_indices[test_pos]

                train_val_labels = {
                    head: arr[train_pos]
                    for head, arr in all_labels.items()
                }
                test_labels = {
                    head: arr[test_pos]
                    for head, arr in all_labels.items()
                }

        else:
            raise ValueError(f"[Data Splitting] Unknown split_level: {split_level!r}")

    except Exception as e:
        logging.error(f"[Data Splitting] Error during data split: {e}", exc_info=True)
        raise

    # ----------------------------
    # 3) Sanity check: no overlap
    # ----------------------------
    if set(train_val_indices).intersection(test_indices):
        msg = "Train/Val and test index sets overlap!"
        logging.error(f"[Data Splitting] {msg}")
        raise ValueError(msg)

    # ----------------------------
    # 4) Log split summary
    # ----------------------------
    logging.info(f"[Data Splitting] split_level      : {split_level}")
    logging.info(f"[Data Splitting] Total samples     : {len(all_indices)}")
    logging.info(f"[Data Splitting] Train/Val samples : {len(train_val_indices)}")
    logging.info(f"[Data Splitting] Test samples      : {len(test_indices)}")

    return train_val_indices, train_val_labels, test_indices, test_labels
