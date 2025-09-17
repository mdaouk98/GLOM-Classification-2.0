# src/mc_utils/mc_clustering/compute_distance_metrics.py

import numpy as np

def compute_distance_metrics(new_means, main_mean, main_cov):
    """
    Compute Euclidean Distance, Cosine Similarity, and Mahalanobis Distance between each new image feature vector
    (represented by its MC mean) and the main dataset (represented by its aggregated mean and covariance).
    
    Args:
        new_means (np.ndarray): Array of shape (N, D) containing the mean feature vector for each new image.
        main_mean (np.ndarray): Array of shape (D,) containing the aggregated mean of the main dataset.
        main_cov (np.ndarray): Array of shape (D, D) containing the aggregated covariance matrix of the main dataset.
    
    Returns:
        euclidean_dists (np.ndarray): Euclidean distances for each new image (shape: (N,)).
        cosine_sims (np.ndarray): Cosine similarities for each new image (shape: (N,)).
        mahalanobis_dists (np.ndarray): Mahalanobis distances for each new image (shape: (N,)).
    """
    # Euclidean distance
    euclidean_dists = np.linalg.norm(new_means - main_mean, axis=1)
    
    # Cosine similarity: (a dot b) / (||a|| * ||b||)
    main_norm = np.linalg.norm(main_mean)
    cosine_sims = []
    for vec in new_means:
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0 or main_norm == 0:
            cosine_sim = 0
        else:
            cosine_sim = np.dot(vec, main_mean) / (vec_norm * main_norm)
        cosine_sims.append(cosine_sim)
    cosine_sims = np.array(cosine_sims)
    
    # Mahalanobis distance: sqrt((x - mean)^T * cov_inv * (x - mean))
    try:
        cov_inv = np.linalg.inv(main_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(main_cov)
    
    mahalanobis_dists = []
    for vec in new_means:
        diff = vec - main_mean
        dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
        mahalanobis_dists.append(dist)
    mahalanobis_dists = np.array(mahalanobis_dists)
    
    return euclidean_dists, cosine_sims, mahalanobis_dists