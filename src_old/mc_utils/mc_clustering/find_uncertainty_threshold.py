# src/mc_utils/mc_clustering/find_uncertainty_threshold.py


import numpy as np
import logging
from sklearn.metrics import roc_curve

def find_uncertainty_threshold(labels, softmax_probs, uncertainty):
    """
    Find an uncertainty threshold that separates misclassified predictions.
    
    Parameters:
        labels (numpy.ndarray): Array of shape (N,) containing the ground truth labels.
        softmax_probs (numpy.ndarray): Array of shape (N, 2) with the softmax probabilities.
        uncertainty (numpy.ndarray): Array of shape (N,) with the uncertainty measure.
        
        
    Returns:
        threshold_candidate (float): Suggested uncertainty threshold.
        threshold_candidates, misclass_ratio, num_uncertain, tpr, fpr
    """
    # Compute predicted labels
    predicted = np.argmax(softmax_probs, axis=1)
    misclassified = predicted != labels
    misclassified = misclassified.astype(int)  # binary: 1 for misclassified, 0 for correct
    
    logging.info("Computing ROC curve for misclassification detection based on uncertainty.")
    # Using uncertainty as a "score" to detect misclassification:
    fpr, tpr, thresholds = roc_curve(misclassified, uncertainty)
    
    # Youden's index (tpr - fpr) can be used to select the threshold.
    youdens_index = tpr - fpr
    best_idx = np.argmax(youdens_index)
    threshold_candidate = thresholds[best_idx]
    logging.info(f"Suggested uncertainty threshold based on Youden's index: {threshold_candidate:.4f}")
    
    # Additionally, examine performance over a range of threshold candidates
    threshold_candidates = np.linspace(np.min(uncertainty), np.max(uncertainty), 50)
    misclass_ratio = []
    num_uncertain = []
    
    for t in threshold_candidates:
        idx = uncertainty >= t
        if np.sum(idx) > 0:
            ratio = np.sum(misclassified[idx]) / np.sum(idx)
        else:
            ratio = np.nan
        misclass_ratio.append(ratio)
        num_uncertain.append(np.sum(idx))
    
    return threshold_candidate, threshold_candidates, misclass_ratio, num_uncertain, tpr, fpr


