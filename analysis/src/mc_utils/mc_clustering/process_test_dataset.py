# src/mc_utils/mc_clustering/process_test_dataset.py

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from mc_utils.mc_clustering import (
    process_test_fold,
    create_dataloader,
    compute_mc_statistics,
    compute_uncertainty,
    find_uncertainty_threshold,
    evaluate_model
)

def process_test_dataset(
    test_indices: np.ndarray,
    test_labels: np.ndarray,
    transform: Optional[Any],
    processor: Optional[Any],
    for_vision: bool,
    device: torch.device,
    config: Any,
    retrained: bool = False
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
    mc_softmax_probabilities_list: List[np.ndarray] = []
    for fold_idx in range(1, num_folds + 1):
        try:
            logging.info(f"Processing test dataset - Fold {fold_idx} started.")
            dataloader = create_dataloader(remaining_test_indices, transform, processor, for_vision, config)
            fold_mc_softmax_probabilities = process_test_fold(fold_idx, remaining_test_indices, dataloader, device, config,retrained)
            
            if fold_mc_softmax_probabilities is not None:
                mc_softmax_probabilities_list.append(fold_mc_softmax_probabilities)
            logging.info(f"Processing test dataset - Fold {fold_idx} completed.")
        except Exception as e:
            logging.error(f"Error processing test dataset fold {fold_idx}: {e}")


    if not mc_softmax_probabilities_list:
        logging.error("No MC softmax probabilities were extracted for test across folds.")
        return None

    
    # Concatenate softmax probabilities from all folds (the result has shape (5MC, test_dataset, 2)).
    aggregated_mc_softmax_probabilities = np.concatenate(mc_softmax_probabilities_list, axis=0)
    
    # Extract class 1 probabilities (second column) (the result has shape (5MC, test_dataset, )).
    mc_softmax_class_1_probabilities = aggregated_mc_softmax_probabilities[:,:,1]
    
    # Compute per-image MC softmax probabilities mean.
    softmax_probabilities_mean, _, _, _ = compute_mc_statistics(aggregated_mc_softmax_probabilities)
    
    # Extract per-image class 1 probabilities (second column) (the result has shape (test_dataset, )).
    mc_softmax_class_1_probabilities_mean = softmax_probabilities_mean[:, 1]
    
    #Compute Uncertainty for each image (take shape (5MC, test_dataset, 2) and returns shape (test_dataset,))
    uncertainty_vector = compute_uncertainty (aggregated_mc_softmax_probabilities)
    
    #evaluate model (input of shape (MC,N,2))
    eval_metrics = evaluate_model(aggregated_mc_softmax_probabilities,remaining_test_labels)
    
    #find Uncertainty Threshold 
    uncertainty_thresh,threshold_candidates, misclass_ratio, num_uncertain, uncertainty_tpr, uncertainty_fpr = find_uncertainty_threshold(remaining_test_labels, softmax_probabilities_mean, uncertainty_vector)

    results_test: Dict[str, Any] = {
        "eval_auc": eval_metrics['auc'],
        "eval_accuracy": eval_metrics['accuracy'],
        "eval_sensitivity": eval_metrics['sensitivity'],
        "eval_specificity": eval_metrics['specificity'],
        "eval_fpr": eval_metrics['fpr'],
        "eval_tpr": eval_metrics['tpr'],
        "uncertainty": uncertainty_vector,
        "uncertainty_thresh": uncertainty_thresh,
        "mc_softmax_class_1_probabilities": mc_softmax_class_1_probabilities,
        "mc_softmax_class_1_probabilities_mean": mc_softmax_class_1_probabilities_mean,
        "threshold_candidates": threshold_candidates,
        "misclass_ratio": misclass_ratio,
        "num_uncertain": num_uncertain,
        "uncertainty_tpr": uncertainty_tpr,
        "uncertainty_fpr": uncertainty_fpr
    }
    
    
    return results_test
