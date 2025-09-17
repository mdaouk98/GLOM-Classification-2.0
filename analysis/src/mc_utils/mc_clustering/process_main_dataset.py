# src/mc_utils/mc_clustering/process_main_dataset.py

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from mc_utils.mc_clustering import (
    compute_statistics,
    compute_main_dataset_metrics,
    process_main_fold,
    create_dataloader
)

def process_main_dataset(
    train_val_indices: np.ndarray,
    train_val_labels: np.ndarray,
    transform: Optional[Any],
    processor: Optional[Any],
    for_vision: bool,
    device: torch.device,
    config: Any
) -> Optional[Dict[str, Any]]:
    """
    Process the main dataset (train+validation) over multiple folds to extract features and compute statistics.
    
    This function:
      - Splits the main dataset into folds using stratified sampling.
      - For each fold, creates a dataloader and extracts feature vectors and softmax probabilities.
      - Aggregates features across folds and computes overall statistics and thresholds.
      - Aggregates softmax probabiliies across folds and split based on the maximum probability (i.e., predicted class).
    
    Args:
        train_val_indices (np.ndarray): Indices for the main (train+validation) dataset.
        train_val_labels (np.ndarray): Corresponding labels for the main dataset.
        transform (Optional[Any]): Data transformation/augmentation pipeline.
        processor (Optional[Any]): Image processor (if applicable).
        for_vision (bool): Flag indicating if the model is a vision model.
        device (torch.device): Computation device (CPU or GPU).
        config (Any): Configuration object.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary with aggregated statistics and computed distance metrics, per class softmax probabilities 
                                  or None if no features and softmax probabilities were extracted.
    """
    main_features_list: List[np.ndarray] = []
    main_softmax_probabilities_list: List[np.ndarray] = []
    num_folds = config.training.folds

    # Stratified K-Fold splitting to maintain label distribution in each fold.
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.training.seed)
    splits = list(skf.split(train_val_indices, train_val_labels))

    for fold_idx, (_, _) in enumerate(splits, start=1):
        try:
            logging.info(f"Processing main dataset - Fold {fold_idx} started.")
            dataloader = create_dataloader(train_val_indices, transform, processor, for_vision, config, cache_in_memory = False)
            features, softmax_probabilities = process_main_fold(fold_idx, train_val_indices, dataloader, device, config)
            if features is not None:
                main_features_list.append(features)
            if softmax_probabilities is not None:
                main_softmax_probabilities_list.append(softmax_probabilities)
            logging.info(f"Processing main dataset - Fold {fold_idx} completed.")
        except Exception as e:
            logging.error(f"Error processing main dataset fold {fold_idx}: {e}")

    if not main_features_list:
        logging.error("No features were extracted for the main dataset across folds.")
        return None
    if not main_softmax_probabilities_list:
        logging.error("No softmax probabilities were extracted for the main dataset across folds.")
        return None

    # Concatenate features from all folds and compute statistics.
    aggregated_main_features = np.concatenate(main_features_list, axis=0)
    avg_mean, avg_std, avg_var, avg_cov = compute_statistics(aggregated_main_features)
    
    # Concatenate softmax probabilities from all folds and compute statistics.
    aggregated_main_softmax_probabilities = np.concatenate(main_softmax_probabilities_list, axis=0)
    
    # Split based on the maximum probability (i.e., predicted class)
    predicted_classes = np.argmax(aggregated_main_softmax_probabilities, axis=1)
    class0_softmax = aggregated_main_softmax_probabilities[predicted_classes == 0]
    class1_softmax = aggregated_main_softmax_probabilities[predicted_classes == 1]

    # Extract class 1 probabilities (second column)
    main_softmax_class_1_probabilities = aggregated_main_softmax_probabilities[:, 1]
    # Compute distance metrics to determine thresholds.
    main_metrics = compute_main_dataset_metrics(aggregated_main_features, avg_mean, avg_cov, class0_softmax, class1_softmax)
    
    # Create dictionary containing all results
    results_main: Dict[str, Any] = {
        "mean": avg_mean,
        "std": avg_std,
        "var": avg_var,
        "cov": avg_cov,
        "main_softmax_class_1_probabilities": main_softmax_class_1_probabilities,
        "euclidean_dists": main_metrics['euclidean_dists'],
        "class0_euclidean_dists": main_metrics['euclidean_dists'][predicted_classes == 0],
        "class1_euclidean_dists": main_metrics['euclidean_dists'][predicted_classes == 1],
        "cosine_sims": main_metrics['cosine_sims'],
        "class0_cosine_sims": main_metrics['cosine_sims'][predicted_classes == 0],
        "class1_cosine_sims": main_metrics['cosine_sims'][predicted_classes == 1],
        "mahalanobis_dists": main_metrics['mahalanobis_dists'],
        "class0_mahalanobis_dists": main_metrics['mahalanobis_dists'][predicted_classes == 0],
        "class1_mahalanobis_dists": main_metrics['mahalanobis_dists'][predicted_classes == 1],
        "class0_probabilities": main_metrics['class0_probabilities'],
        "class1_probabilities": main_metrics['class1_probabilities'],
        "euclidean_thresh_95": main_metrics['euclidean_thresh_95'],
        "euclidean_thresh_90": main_metrics['euclidean_thresh_90'],
        "euclidean_thresh_85": main_metrics['euclidean_thresh_85'],
        "euclidean_thresh_80": main_metrics['euclidean_thresh_80'],
        "cosine_thresh_5": main_metrics['cosine_thresh_5'],
        "cosine_thresh_10": main_metrics['cosine_thresh_10'],
        "cosine_thresh_15": main_metrics['cosine_thresh_15'],
        "cosine_thresh_20": main_metrics['cosine_thresh_20'],
        "mahalanobis_thresh_95_percentile": main_metrics['mahalanobis_thresh_95_percentile'],
        "mahalanobis_thresh_90_percentile": main_metrics['mahalanobis_thresh_90_percentile'],
        "mahalanobis_thresh_85_percentile": main_metrics['mahalanobis_thresh_85_percentile'],
        "mahalanobis_thresh_80_percentile": main_metrics['mahalanobis_thresh_80_percentile'],
        "mahalanobis_thresh_chi2": main_metrics['mahalanobis_thresh_chi2'],
        "class0_thresh_95": main_metrics['class0_thresh_95'],
        "class0_thresh_90": main_metrics['class0_thresh_90'],
        "class0_thresh_85": main_metrics['class0_thresh_85'],
        "class0_thresh_80": main_metrics['class0_thresh_80'],
        "class1_thresh_5": main_metrics['class1_thresh_5'],
        "class1_thresh_10": main_metrics['class1_thresh_10'],
        "class1_thresh_15": main_metrics['class1_thresh_15'],
        "class1_thresh_20": main_metrics['class1_thresh_20']
    }
    
    return results_main
