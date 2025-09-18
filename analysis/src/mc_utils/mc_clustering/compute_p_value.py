# src/mc_utils/mc_clustering/compute_p_value.py

import logging
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

def compute_p_value(sample, class0_probabilities, class1_probabilities,comparing: str = None, alpha: float = 0.05) -> dict:
    """
    For each column in new_images_results['mc_softmax_class_1_probabilities'] (shape: (MC, N)),
    perform a Shapiro-Wilk test for normality. If the column data is normally distributed,
    perform a t-test comparing the MC samples against:
        - main_results['class0_probabilities'] (shape: (N1,))
        - main_results['class1_probabilities'] (shape: (N2,))
    Otherwise, perform a Mann-Whitney U-test for both comparisons.
    
    Only the p-values are collected and logged.
    
    Args:
        main_results (dict): Must contain 'class0_probabilities' (shape: (N,)) and 'class1_probabilities' (shape: (M,)).
        new_images_results (dict): Must contain 'mc_softmax_class_1_probabilities' (shape: (MC, N)).
        alpha (float): Significance level for the Shapiro test (default: 0.05).
        
    Returns:
        dict: Dictionary mapping each column index to a dictionary with the following keys:
              - 'shapiro_p': p-value from the Shapiro-Wilk test.
              - 'test_p_class0': p-value from either the t-est or Mann-Whitney U test vs class0.
              - 'test_p_class1': p-value from either the t-test or Mann-hitney U test vs class1.
              - 'test_used': Either 't-test' or 'mannwhitney' indicating which test was used.
    """
    # Extract the required arrays.
    class0_probs = class0_probabilities
    class1_probs = class1_probabilities
    mc_softmax = sample

    # Determine shape: A samples and B columns.
    A, B = mc_softmax.shape
    A, B = mc_softmax.shape
    results = {
        'new_image_index': list(range(B)),
        f'{comparing} test_p_class0': [],
        f'{comparing} test_p_class1': [],
        f'{comparing} test_used': []
    }
    for i in range(B):
        sample_column = mc_softmax[:, i]
        
        # Perform Shapiro-Wilk test for normality.
        _, shapiro_p = shapiro(sample_column)
        #results[i] = {'shapiro_p': shapiro_p}
        
        # Currently assuming that they are not normally distributed. Can modify later. 
        shapiro_p = 0.01
        # Depending on the normality, perform the appropriate test.
        if shapiro_p > alpha:
            # Data appears normally distributed: use independent t-tests.
            _, t_p0 = ttest_ind(sample_column, class0_probabilities, equal_var=False)
            _, t_p1 = ttest_ind(sample_column, class1_probabilities, equal_var=False)
            results[f'{comparing} test_p_class0'].append(t_p0)
            results[f'{comparing} test_p_class1'].append(t_p1)
            results[f'{comparing} test_used'].append('t-test')
        else:
            # Data is not normally distributed: use Mann–Whitney U-tests.
            _, u_p0 = mannwhitneyu(sample_column, class0_probabilities, alternative='two-sided')
            _, u_p1 = mannwhitneyu(sample_column, class1_probabilities, alternative='two-sided')
            results[f'{comparing} test_p_class0'].append(u_p0)
            results[f'{comparing} test_p_class1'].append(u_p1)
            results[f'{comparing} test_used'].append('mannwhitney')
        
        # Log the p-values for this column.
        logging.info(
            #f"Column {i}: Shapiro p-value = {results[i]['shapiro_p']:.4f}, "
            f"Currently assuming that they are not normally distributed. Can modify later."
            #f"Test used = {results[i]['test_used']}, "
            #f"p-value vs class0 = {results[i]['test_p_class0']:.4f}, "
            #f"p-value vs class1 = {results[i]['test_p_class1']:.4f}"
        )
    
    return results
