# src/mc_utils/mc_clustering/compute_uncertainty.py

import numpy as np
import logging

def compute_uncertainty(S, epsilon=1e-10):
    """
    Compute the predictive entropy uncertainty for each image.
    
    Parameters:
        S (numpy.ndarray): Array of shape (MC, N, 2) containing the class probabilities
                           from MC dropout iterations for N images and 2 classes.
        epsilon (float): Small value added to probabilities to avoid log(0).
        
    Returns:
        entropy (numpy.ndarray): Array of shape (N,) containing the predictive entropy
                                 for each image.
    """
    logging.info("Computing mean probabilities over MC iterations.")
    # Average over MC iterations to get the aggregated probability for each image
    mean_probs = np.mean(S, axis=0)  # shape (N, 2)
    
    logging.info("Computing predictive entropy for each image.")
    # Compute the predictive entropy for each image
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)  # shape (N,)
    
    logging.info("Entropy computation complete.")
    return entropy