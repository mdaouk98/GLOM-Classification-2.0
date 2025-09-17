# src/mc_utils/mc_clustering/compute_statistics.py

import numpy as np

def compute_statistics(features):
    """
    Compute mean, standard deviation, variance, and covariance for an array of features.
    
    Parameters:
        features (np.ndarray): Input array of shape (N, D), where N is the number of samples
                               and D is the dimensionality of each feature.
    
    Returns:
        tuple: A tuple containing:
            - mean (np.ndarray): Array of shape (D,), representing the mean of features along axis 0.
            -std (np.ndarray): Array of shape (D,), representing the standard deviation along axis 0.
            -var (np.ndarray): Array of shape (D,), representing the variance along axis 0.
            - cov (np.ndarray or None): Covariance matrix of shape (D, D) if N > 1, otherwise None.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    var = np.var(features, axis=0)
    cov = np.cov(features, rowvar=False) if features.shape[0] > 1 else None
    return mean, std, var, cov


