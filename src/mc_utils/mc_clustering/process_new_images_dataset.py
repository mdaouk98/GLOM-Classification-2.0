# src/mc_utils/mc_clustering/process_new_images_dataset.py

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from mc_utils.mc_clustering import (
    process_new_images_fold,
    create_dataloader,
    compute_mc_statistics,
    compute_distance_metrics,
    compute_mc_distance_metrics,
    compute_uncertainty
)

def process_new_images_dataset(
    test_indices: np.ndarray,
    test_labels: np.ndarray,
    transform: Optional[Any],
    processor: Optional[Any],
    for_vision: bool,
    device: torch.device,
    config: Any,
    main_stats: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Process a subset (10%) of the test dataset-referred to as "new_images"-using MC dropout.
    
    This function:
      - Splits 10% of the test set as new_images.
      - Processes the new_images over multiple folds to extract MC features and softmax probabilities.
      - Aggregates these features and softmax probabilities and computes per-image MC statistics.
      - Computes distance metrics between the new images and the main dataset statistics.
    
    Args:
        test_indices (np.ndarray): Indices for the test dataset.
        test_labels (np.ndarray): Labels for the test dataset.
        transform (Optional[Any]): Data transformation/augmentation pipeline.
        processor (Optional[Any]): Image processor (if applicable).
        for_vision (bool): Flag indicating if the model is a vision model.
        device (torch.device): Computation device (CPU or GPU).
        config (Any): Configuration object.
        main_stats (Dict[str, Any]): Precomputed statistics from the main dataset.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing MC feature statistics and distance metrics, softmax probabilities
                                  or None if processing fails.
    """
    # Split out new_images from test indices (10% for new images).
    new_images_indices, remaining_test_indices, new_images_labels, remaining_test_labels = train_test_split(
        test_indices,
        test_labels,
        test_size=0.9,
        random_state=config.training.seed,
        stratify=test_labels
    )
    logging.info(f"New_images samples: {len(new_images_indices)}; Remaining test samples (unused): {len(remaining_test_indices)}")

    num_folds = config.training.folds
    mc_features_list: List[np.ndarray] = []
    mc_softmax_probabilities_list: List[np.ndarray] = []
    for fold_idx in range(1, num_folds + 1):
        try:
            logging.info(f"Processing new_images dataset - Fold {fold_idx} started.")
            dataloader = create_dataloader(new_images_indices, transform, processor, for_vision, config)
            fold_mc_features, fold_mc_softmax_probabilities = process_new_images_fold(fold_idx, new_images_indices, dataloader, device, config)
            
            if fold_mc_features is not None:
                mc_features_list.append(fold_mc_features)
            if fold_mc_softmax_probabilities is not None:
                mc_softmax_probabilities_list.append(fold_mc_softmax_probabilities)
            logging.info(f"Processing new_images dataset - Fold {fold_idx} completed.")
        except Exception as e:
            logging.error(f"Error processing new_images dataset fold {fold_idx}: {e}")

    if not mc_features_list:
        logging.error("No MC features were extracted for new_images across folds.")
        return None
    if not mc_softmax_probabilities_list:
        logging.error("No MC softmax probabilities were extracted for new_images across folds.")
        return None

    # Concatenate features from all folds (the result has shape (5MC, new_images_dataset, D)).
    aggregated_mc_features = np.concatenate(mc_features_list, axis=0)
    
    # Compute distance metrics between new images and the main dataset statistics.
    mc_euclidean_dists, mc_cosine_sims, mc_mahalanobis_dists, mean_mc_euclidean_dists, mean_mc_cosine_sims, mean_mc_mahalanobis_dists = compute_mc_distance_metrics( aggregated_mc_features, main_stats["mean"], main_stats["cov"]
    )

    # Compute per-image MC statistics.
    new_means, new_stds, new_vars, new_covs = compute_mc_statistics(aggregated_mc_features)

    # Compute distance metrics between new images mean and the main dataset statistics.
    euclidean_dists, cosine_sims, mahalanobis_dists = compute_distance_metrics(
        new_means, main_stats["mean"], main_stats["cov"]
    )
    
    # Concatenate softmax probabilities from all folds (the result has shape (5MC, new_images_dataset, 2)).
    aggregated_mc_softmax_probabilities = np.concatenate(mc_softmax_probabilities_list, axis=0)
    
    #Compute Uncertainty for each image (take shape (5MC, new_images_dataset, 2) and returns shape (new_images_dataset,))
    uncertainty_vector = compute_uncertainty (aggregated_mc_softmax_probabilities)
    
    # Extract class 1 probabilities (second column) (the result has shape (5MC, new_images_dataset, )).
    mc_softmax_class_1_probabilities = aggregated_mc_softmax_probabilities[:,:,1]
    
    # Compute per-image MC softmax probabilities mean.
    softmax_probabilities_mean, _, _, _ = compute_mc_statistics(aggregated_mc_softmax_probabilities)
    
    # Extract per-image class 1 probabilities (second column) (the result has shape (new_images_dataset, )).
    mc_softmax_class_1_probabilities_mean = softmax_probabilities_mean[:, 1]

    results_new_images: Dict[str, Any] = {
        "means": new_means,
        "stds": new_stds,
        "vars": new_vars,
        "covs": new_covs,
        "mc_euclidean_dists": mc_euclidean_dists,
        "mc_cosine_sims": mc_cosine_sims,
        "mc_mahalanobis_dists": mc_mahalanobis_dists,
        "mean_mc_euclidean_dists": mean_mc_euclidean_dists,
        "mean_mc_cosine_sims": mean_mc_cosine_sims,
        "mean_mc_mahalanobis_dists": mean_mc_mahalanobis_dists,
        "euclidean_dists": euclidean_dists,
        "cosine_sims": cosine_sims,
        "mahalanobis_dists": mahalanobis_dists,
        "uncertainty": uncertainty_vector,
        "mc_softmax_class_1_probabilities": mc_softmax_class_1_probabilities,
        "mc_softmax_class_1_probabilities_mean": mc_softmax_class_1_probabilities_mean
    }
    
    
    return results_new_images
