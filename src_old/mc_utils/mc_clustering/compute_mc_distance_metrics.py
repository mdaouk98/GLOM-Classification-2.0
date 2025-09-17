# src/mc_utils/mc_clustering/compute_mc_distance_metrics.py

import numpy as np

def compute_mc_distance_metrics(aggregated_mc_features, main_mean, main_cov):
    """
    Compute Euclidean Distance, Cosine Similarity, and Mahalanobis Distance between each aggregated
    feature vector (MC samples per new image) and the main dataset's aggregated mean and covariance.

    Args:
        aggregated_feature_vector (np.ndarray): Array of shape (MC, N, D) containing the MC feature vectors for each new image.
        main_mean (np.ndarray): Array of shape (D,) containing the aggregated mean of the main dataset.
        main_cov (np.ndarray): Array of shape (D, D) containing the aggregated covariance matrix of the main dataset.

    Returns:
        euclidean_dists (np.ndarray): Euclidean distances for each MC sample of each new image (shape: (MC, N)).
        cosine_sims (np.ndarray): Cosine similarities for each MC sample of each new image (shape: (MC, N)).
        mahalanobis_dists (np.ndarray): Mahalanobis distances for each MC sample of each new image (shape: (MC, N)).
        mean_euclidean_dists (np.ndarray): Mean Euclidean distances over MC for each new image (shape: (N,)).
        mean_cosine_sims (np.ndarray): Mean Cosine similarities over MC for each new image (shape: (N,)).
        mean_mahalanobis_dists (np.ndarray): Mean Mahalanobis distances over MC for each new image (shape: (N,)).
    """
    # Euclidean distance: subtract main_mean (broadcasting over MC and N), then compute L2 norm over D.
    diff = aggregated_mc_features - main_mean  # shape: (MC, N, D)
    euclidean_dists = np.linalg.norm(diff, axis=2)  # shape: (MC, N)
    
    # Cosine similarity: compute dot product divided by product of norms.
    # Compute norms for aggregated_mc_features along D.
    new_norms = np.linalg.norm(aggregated_mc_features, axis=2)  # shape: (MC, N)
    main_norm = np.linalg.norm(main_mean)  # scalar
    # Compute dot products: sum over D (broadcast main_mean automatically)
    dot_products = np.sum(aggregated_mc_features * main_mean, axis=2)  # shape: (MC, N)
    
    # To avoid division by zero, set similarity to 0 if either norm is 0.
    # Create a denominator array:
    denom = new_norms * main_norm  # shape: (MC, N)
    # Use np.where to handle potential zeros.
    cosine_sims = np.where(denom == 0, 0, dot_products / denom)
    
    # Mahalanobis distance: first, compute the inverse covariance matrix.
    try:
        cov_inv = np.linalg.inv(main_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(main_cov)
    
    # Use einsum to vectorize the computation.
    # For each feature vector x, compute: sqrt((x - mean)^T * cov_inv * (x - mean))
    mahalanobis_sq = np.einsum('mnd,dc,mnc->mn', diff, cov_inv, diff)
    mahalanobis_dists = np.sqrt(mahalanobis_sq)
    
    # Compute the mean distances/similarities over the MC dimension (axis=0)
    mean_euclidean_dists = np.mean(euclidean_dists, axis=0)   # shape: (N,)
    mean_cosine_sims = np.mean(cosine_sims, axis=0)             # shape: (N,)
    mean_mahalanobis_dists = np.mean(mahalanobis_dists, axis=0) # shape: (N,)
    
    return euclidean_dists, cosine_sims, mahalanobis_dists, mean_euclidean_dists, mean_cosine_sims, mean_mahalanobis_dists
