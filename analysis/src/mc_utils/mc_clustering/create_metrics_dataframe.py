# src/mc_utils/mc_clustering/create_metrics_dataframe.py

import numpy as np
import pandas as pd

def create_metrics_dataframe(
    indices: np.ndarray,
    euclidean_dists: np.ndarray,
    cosine_sims: np.ndarray,
    mahalanobis_dists: np.ndarray,
    main_softmax_class_1_probabilities: np.ndarray,
    euclidean_thresh_95: float,
    euclidean_thresh_90: float,
    euclidean_thresh_85: float,
    euclidean_thresh_80: float,
    cosine_thresh_5: float,
    cosine_thresh_10: float,
    cosine_thresh_15: float,
    cosine_thresh_20: float,
    mahalanobis_thresh_95_percentile: float,
    mahalanobis_thresh_90_percentile: float,
    mahalanobis_thresh_85_percentile: float,
    mahalanobis_thresh_80_percentile: float,
    mahalanobis_thresh_chi2: float,
    class0_thresh_95: float,
    class0_thresh_90: float,
    class0_thresh_85: float,
    class0_thresh_80: float,
    class1_thresh_5: float,
    class1_thresh_10: float,
    class1_thresh_15: float,
    class1_thresh_20: float
) -> pd.DataFrame:
    """
    Create a DataFrame comparing various distance metrics for each image against their thresholds.
    
    Args:
        indices (np.ndarray): Array of image indices.
        euclidean_dists (np.ndarray): Computed Euclidean distances.
        cosine_sims (np.ndarray): Computed cosine similarities.
        mahalanobis_dists (np.ndarray): Computed Mahalanobis distances.
        euclidean_thresh (float): Threshold value for Euclidean distance.
        cosine_thresh (float): Threshold value for cosine similarity.
        mahalanobis_thresh_percentile (float): Percentile-based threshold for Mahalanobis distance.
        mahalanobis_thresh_chi2 (float): Chi-squared threshold for Mahalanobis distance.
    
    Returns:
        pd.DataFrame: A DataFrame that includes the computed metrics, thresholds, and flags indicating
                      whether each metric exceeds its threshold.
    """
    num = indices.shape[0]
    df_dist = pd.DataFrame({
        "Image_Index": indices,
        "Euclidean_Distance": euclidean_dists,
        "euclidean_thresh_95": [euclidean_thresh_95] * num,
        "Euclidean_Flag_95": euclidean_dists > euclidean_thresh_95,
        "euclidean_thresh_90": [euclidean_thresh_90] * num,
        "Euclidean_Flag_90": euclidean_dists > euclidean_thresh_90,
        "euclidean_thresh_85": [euclidean_thresh_85] * num,
        "Euclidean_Flag_85": euclidean_dists > euclidean_thresh_85,
        "euclidean_thresh_80": [euclidean_thresh_80] * num,
        "Euclidean_Flag_80": euclidean_dists > euclidean_thresh_80,
        "Cosine_Similarity": cosine_sims,
        "cosine_thresh_5": [cosine_thresh_5] * num,
        "Cosine_Flag_5": cosine_sims < cosine_thresh_5,
        "cosine_thresh_10": [cosine_thresh_10] * num,
        "Cosine_Flag_10": cosine_sims < cosine_thresh_10,
        "cosine_thresh_15": [cosine_thresh_15] * num,
        "Cosine_Flag_15": cosine_sims < cosine_thresh_15,
        "cosine_thresh_20": [cosine_thresh_20] * num,
        "Cosine_Flag_20": cosine_sims < cosine_thresh_20,
        "Mahalanobis_Distance": mahalanobis_dists,
        "mahalanobis_thresh_95_percentile": [mahalanobis_thresh_95_percentile] * num,
        "Mahalanobis_Flag_95_Percentile": mahalanobis_dists > mahalanobis_thresh_95_percentile,
        "mahalanobis_thresh_90_percentile": [mahalanobis_thresh_90_percentile] * num,
        "Mahalanobis_Flag_90_Percentile": mahalanobis_dists > mahalanobis_thresh_90_percentile,
        "mahalanobis_thresh_85_percentile": [mahalanobis_thresh_85_percentile] * num,
        "Mahalanobis_Flag_85_Percentile": mahalanobis_dists > mahalanobis_thresh_85_percentile,
        "mahalanobis_thresh_80_percentile": [mahalanobis_thresh_80_percentile] * num,
        "Mahalanobis_Flag_80_Percentile": mahalanobis_dists > mahalanobis_thresh_80_percentile,
        "Mahalanobis_Threshold_Chi2": [mahalanobis_thresh_chi2] * num,
        "Mahalanobis_Flag_Chi2": mahalanobis_dists > mahalanobis_thresh_chi2,
        "main_softmax_class_1_probabilities": main_softmax_class_1_probabilities,
        "class0_thresh_95": [class0_thresh_95] * num,
        "class0_Flag_95": main_softmax_class_1_probabilities > class0_thresh_95,
        "class0_thresh_90": [class0_thresh_90] * num,
        "class0_Flag_90": main_softmax_class_1_probabilities > class0_thresh_90,
        "class0_thresh_85": [class0_thresh_85] * num,
        "class0_Flag_85": main_softmax_class_1_probabilities > class0_thresh_85,
        "clas0_thresh_80": [class0_thresh_80] * num,
        "class0_Flag_80": main_softmax_class_1_probabilities > class0_thresh_80,
        "class1_thresh_5": [class1_thresh_5] * num,
        "class1_Flag_5": main_softmax_class_1_probabilities < class1_thresh_5,
        "class1_thresh_10": [class1_thresh_10] * num,
        "class1_Flag_10": main_softmax_class_1_probabilities < class1_thresh_10,
        "class1_thresh_15": [class1_thresh_15] * num,
        "class1_Flag_15": main_softmax_class_1_probabilities < class1_thresh_15,
        "class1_thresh_20": [class1_thresh_20] * num,
        "class1_Flag_20": main_softmax_class_1_probabilities < class1_thresh_20
    })
    return df_dist

