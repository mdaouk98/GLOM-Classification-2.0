# src/mc_utils/mc_clustering/process_main_fold.py

import numpy as np
import logging
import torch
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from mc_utils.mc_clustering import extract_features, extract_softmax_probabilities, load_model



def process_main_fold(fold_idx: int,
                      indices: np.ndarray,
                      dataloader: DataLoader,
                      device: torch.device,
                      config: Any) -> Optional[np.ndarray]:
    """
    Process one fold for the main dataset (train+val) and extract features and softmax probabilities.
    """
    logging.info(f"[Main Dataset] Processing fold {fold_idx} with {len(indices)} samples.")
    model = load_model(fold_idx, device, config)
    if model is None:
        return None
    features = extract_features(model, dataloader, device, mc_dropout=False)
    softmax_probabilities = extract_softmax_probabilities(model, dataloader, device, mc_dropout=False)
    if features.size == 0:
        logging.warning(f"No features extracted for fold {fold_idx}.")
        return None
    if softmax_probabilities.size == 0:
        logging.warning(f"No softmax probabilities extracted for fold {fold_idx}.")
        return None
    logging.info(f"Fold {fold_idx}: Extracted {features.shape[0]} features.")
    logging.info(f"Fold {fold_idx}: Extracted {softmax_probabilities.shape[0]} softmax probabilities.")
    return features, softmax_probabilities




