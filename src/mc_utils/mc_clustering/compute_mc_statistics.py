# src/mc_utils/mc_clustering/compute_mc_statistics.py

import numpy as np



def compute_mc_statistics(mc_features):
    """
    Compute per-image MC statistics from an array of shape (MC, N, D) where MC is the number of MC iterations.
    For each image, compute its mean, std, variance, and covariance matrix.
    
    Returns:
        means: np.ndarray of shape (N, D)
        stds: np.ndarray of shape (N, D)
        vars: np.ndarray of shape (N, D)
        covs: np.ndarray of shape (N, D, D)
    """
    mc_means = np.mean(mc_features, axis=0)   # shape (N, D)
    mc_stds = np.std(mc_features, axis=0)       # shape (N, D)
    mc_vars = np.var(mc_features, axis=0)       # shape (N, D)
    
    N, D = mc_means.shape
    cov_matrices = []
    for i in range(N):
        sample_features = mc_features[:, i, :]  # shape (MC, D)
        if sample_features.shape[0] > 1:
            cov = np.cov(sample_features, rowvar=False)  # shape (D, D)
        else:
            cov = np.zeros((D, D))
        cov_matrices.append(cov)
    cov_matrices = np.array(cov_matrices)  # shape (N, D, D)
    
    return mc_means, mc_stds, mc_vars, cov_matrices


