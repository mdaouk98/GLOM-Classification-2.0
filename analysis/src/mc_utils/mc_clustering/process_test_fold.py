# src/mc_utils/mc_clustering/process_test_fold.py

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from mc_utils.mc_clustering import extract_features, extract_softmax_probabilities, load_model


def process_test_fold(fold_idx: int,
                            indices: np.ndarray,
                            dataloader: DataLoader,
                            device: torch.device,
                            config: Any,
                            retrained: bool = False) -> Optional[np.ndarray]:
    """
    Process one fold for the new images dataset using MC-dropout to extract features and softmax probabilities.
    Returns an array of shape (MC_iterations, N_samples, D) and an array of shape (MC_iterations, N_samples, 2).
    """
    model = load_model(fold_idx, device, config,retrained)
    if model is None:
        return None
    mc_iterations = config.training.mc_iterations
    fold_mc_softmax_probabilities: List[np.ndarray] = []
    for mc in range(mc_iterations):
        logging.info(f"Fold {fold_idx} - MC Iteration {mc+1}/{mc_iterations}")
        softmax_probabilities = extract_softmax_probabilities(model, dataloader, device, mc_dropout=True)
        
        if softmax_probabilities.size == 0:
            logging.warning(f"No softmax probabilities extracted in MC iteration {mc+1} for fold {fold_idx}.")
            continue
        fold_mc_softmax_probabilities.append(softmax_probabilities)

    if not fold_mc_softmax_probabilities:
        logging.error(f"No softmax probabilities extracted for fold {fold_idx} in any MC iteration.")
        return None

    fold_mc_softmax_probabilities_array = np.stack(fold_mc_softmax_probabilities, axis=0)
    logging.info(f"Fold {fold_idx}: Collected MC softmax probabilities with shape {fold_mc_softmax_probabilities_array.shape}.")
    return fold_mc_softmax_probabilities_array

