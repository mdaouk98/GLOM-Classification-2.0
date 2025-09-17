# src/mc_utils/mc_clustering/compute_main_dataset_metrics.py 

import numpy as np
from scipy.stats import chi2

def compute_main_dataset_metrics(main_features, aggregated_mean, aggregated_cov, class0_softmax, class1_softmax):
    """
    Computes Euclidean distance, Cosine similarity, and Mahalanobis distance
    for each sample in main_features relative to aggregated_mean.
    
    Args:
        main_features (np.ndarray): Shape (N, D) for N samples and D features.
        aggregated_mean (np.ndarray): Aggregated mean vector of shape (D,).
        aggregated_cov (np.ndarray): Aggregated covariance matrix of shape (D, D).
        
    Returns:
        A dictionary containing:
            - 'euclidean_dists': np.ndarray of shape (N,)
            - 'cosine_sims': np.ndarray of shape (N,)
            - 'mahalanobis_dists': np.ndarray of shape (N,)
            - 'class0_probabilities': np.ndarray of shape (N,)
            - 'class1_probabilities': np.ndarray of shape (N,)
            - 'euclidean_thres_N': Nth percentile of Euclidean distances
            - 'cosine_thresh_N': Nth percentile of cosine similarities
            - 'mahalanobis_thresh_N_percentile': Nth percentile of Mahalanobis distances
            - 'mahalanobis_thresh_chi2': Chi-square based threshold for Mahalanobis distance
            - 'class0_thresh_N': Nth percentile of class 0's class 1's probabilities distribution
            - 'class1_thresh_N': Nth percentile of class 1's class 1's probabilities distribution
    """
    N, D = main_features.shape
    
    # Euclidean distance for each sample:
    euclidean_dists = np.linalg.norm(main_features - aggregated_mean, axis=1)
    
    # Cosine similarity for each sample:
    aggregated_norm = np.linalg.norm(aggregated_mean)
    cosine_sims = []
    for sample in main_features:
        sample_norm = np.linalg.norm(sample)
        if sample_norm == 0 or aggregated_norm == 0:
            cosine_sim = 0
        else:
            cosine_sim = np.dot(sample, aggregated_mean) / (sample_norm * aggregated_norm)
        cosine_sims.append(cosine_sim)
    cosine_sims = np.array(cosine_sims)
    
    # Mahalanobis distance for each sample:
    try:
        cov_inv = np.linalg.inv(aggregated_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(aggregated_cov)
        
    mahalanobis_dists = []
    for sample in main_features:
        diff = sample - aggregated_mean
        dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
        mahalanobis_dists.append(dist)
    mahalanobis_dists = np.array(mahalanobis_dists)
    
    # Extract class 1 probabilities (second column) for class 0 class 1
    class0_probabilities= class0_softmax[:, 1]
    class1_probabilities= class1_softmax[:, 1]
    
    # Compute thresholds at the Nth percentile:
    euclidean_thresh_95_percentile = np.percentile(euclidean_dists, 95)
    cosine_thresh_5_percentile = np.percentile(cosine_sims, 5)  # Note: high cosine means high similarity.
    mahalanobis_thresh_95_percentile = np.percentile(mahalanobis_dists, 95)
    class0_thresh_95_percentile = np.percentile(class0_probabilities, 95)
    class1_thresh_5_percentile = np.percentile(class1_probabilities, 5)
    
    euclidean_thresh_90_percentile = np.percentile(euclidean_dists, 90)
    cosine_thresh_10_percentile = np.percentile(cosine_sims, 10)  # Note: high cosine means high similarity.
    mahalanobis_thresh_90_percentile = np.percentile(mahalanobis_dists, 90)
    class0_thresh_90_percentile = np.percentile(class0_probabilities, 90)
    class1_thresh_10_percentile = np.percentile(class1_probabilities, 10)
    
    euclidean_thresh_85_percentile = np.percentile(euclidean_dists, 85)
    cosine_thresh_15_percentile = np.percentile(cosine_sims, 15)  # Note: high cosine means high similarity.
    mahalanobis_thresh_85_percentile = np.percentile(mahalanobis_dists, 85)
    class0_thresh_85_percentile = np.percentile(class0_probabilities, 85)
    class1_thresh_15_percentile = np.percentile(class1_probabilities, 15)
    
    euclidean_thresh_80_percentile = np.percentile(euclidean_dists, 80)
    cosine_thresh_20_percentile = np.percentile(cosine_sims, 20)  # Note: high cosine means high similarity.
    mahalanobis_thresh_80_percentile = np.percentile(mahalanobis_dists, 80)
    class0_thresh_80_percentile = np.percentile(class0_probabilities, 80)
    class1_thresh_20_percentile = np.percentile(class1_probabilities, 20) 
    
    # Compute chi-square threshold for Mahalanobis distance:
    # For multivariate Gaussian data, the squared Mahalanobis distance follows a chi-square distribution with D degrees of freedom.
    chi2_threshold = chi2.ppf(0.95, df=D)
    mahalanobis_thresh_chi2 = np.sqrt(chi2_threshold)
    
    
    
    return {
        'euclidean_dists': euclidean_dists,
        'cosine_sims': cosine_sims,
        'mahalanobis_dists': mahalanobis_dists,
        'class0_probabilities': class0_probabilities,
        'class1_probabilities': class1_probabilities,
        'euclidean_thresh_95': euclidean_thresh_95_percentile,
        'euclidean_thresh_90': euclidean_thresh_90_percentile,
        'euclidean_thresh_85': euclidean_thresh_85_percentile,
        'euclidean_thresh_80': euclidean_thresh_80_percentile,
        'cosine_thresh_5': cosine_thresh_5_percentile,
        'cosine_thresh_10': cosine_thresh_10_percentile,
        'cosine_thresh_15': cosine_thresh_15_percentile,
        'cosine_thresh_20': cosine_thresh_20_percentile,
        'mahalanobis_thresh_95_percentile': mahalanobis_thresh_95_percentile,
        'mahalanobis_thresh_90_percentile': mahalanobis_thresh_90_percentile,
        'mahalanobis_thresh_85_percentile': mahalanobis_thresh_85_percentile,
        'mahalanobis_thresh_80_percentile': mahalanobis_thresh_80_percentile,
        'mahalanobis_thresh_chi2': mahalanobis_thresh_chi2,
        'class0_thresh_95': class0_thresh_95_percentile,
        'class0_thresh_90': class0_thresh_90_percentile,
        'class0_thresh_85': class0_thresh_85_percentile,
        'class0_thresh_80': class0_thresh_80_percentile,
        'class1_thresh_5': class1_thresh_5_percentile,
        'class1_thresh_10': class1_thresh_10_percentile,
        'class1_thresh_15': class1_thresh_15_percentile,
        'class1_thresh_20': class1_thresh_20_percentile
    }