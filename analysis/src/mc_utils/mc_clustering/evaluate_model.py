# src/mc_utils/mc_clustering/evaluate_model.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

def evaluate_model(S, labels):
    """
    Evaluate a binary classification model given MC dropout predictions and true labels.
    
    Parameters:
        S (np.ndarray): Array of shape (MC, N, 2) containing softmax probabilities from MC dropout.
        labels (np.ndarray): Array of shape (N,) containing true labels (0 or 1).
        
    Returns:
        metrics (dict): Dictionary containing AUC, accuracy, sensitivity, specificity, fpr, tpr, thresholds.
    """
    # Average the MC predictions over iterations
    softmax_mean = np.mean(S, axis=0)  # shape: (N, 2)
    
    # Predicted labels (using argmax on softmax probabilities)
    preds = np.argmax(softmax_mean, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Compute confusion matrix: confusion_matrix returns in order: [ [TN, FP], [FN, TP] ]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # Sensitivity (Recall for positive class): TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Compute ROC curve and AUC using probability for positive class
    probs_positive = softmax_mean[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, probs_positive)
    roc_auc = auc(fpr, tpr)
    
    # Pack metrics into a dictionary
    metrics = {
        'auc': roc_auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    return metrics

