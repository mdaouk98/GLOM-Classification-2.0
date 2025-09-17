import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import entropy
from scipy.special import softmax

def compute_uncertainty(softmax_array):
    """Calculate predictive entropy (uncertainty)."""
    mean_softmax = np.mean(softmax_array, axis=0)  # shape: (N, C)
    return np.mean(entropy(mean_softmax.T))        # average entropy across all samples

results = []
detailed_metrics = []
metrics_list = []

# Iterate over the different hyperparameters

for model_index, model in enumerate(['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'Densenet121', 'Densenet169', 'Densenet201','efficientnet_b0', 'efficientnet_b1',
                                      'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4','efficientnet_b5',
                                     'regnety_008', 'regnety_016','resnext101_32x8d', 'Vision']):
    model_title = ['R18', 'R34', 'R50', 'R101', 'R152','D121', 'D169', 'D201', 'Eff_b0', 'Eff_b1', 'Eff_b2', 'Eff_b3', 'Eff_b4', 'Eff_b5',
                   'RGy8', 'RGy16', 'RX101_32x8d', 'Vit']
    for rmv_inf_grad in [True,False]:
        # Build the base path using the current hyperparameters
        base_path = (f"metrics/rmv_inf_grad/rmv_{rmv_inf_grad}/model_{model}")
        for fold in range(1, 6):
            metric_path = (
                f"{base_path}/training_dict_fold{fold}.json"
            )
            if not os.path.exists(metric_path):
                print(f"File not found: {metric_path}. Skipping.")
                continue
    
            with open(metric_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON in {metric_path}: {e}")
                    continue
    
            # ? here: load logits instead of softmax iterations
            logits_iters = np.array(data['all_testing_dict']['all_logits'])
            # ? convert logits to probabilities along class axis
            softmax_iters = softmax(logits_iters, axis=2)
    
            labels = np.array(data['all_testing_dict']['all_labels'])
            num_epochs_trained = data['num_epochs_trained']
    
            # Compute mean softmax predictions and predicted labels
            mean_softmax = np.mean(softmax_iters, axis=0)
            preds = np.argmax(mean_softmax, axis=1)
    
            # Compute performance metrics
            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average='macro', zero_division=0)
            recall = recall_score(labels, preds, average='macro', zero_division=0)
            f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
            # Compute AUC for binary classification (if applicable)
            positive_probs = mean_softmax[:, 1]
            try:
                auc = roc_auc_score(labels, positive_probs)
            except ValueError as e:
                print(f"Error computing AUC for {loss_function}, {model}, fold {fold}: {e}")
                auc = np.nan
    
            # Compute uncertainty on the softmax probabilities
            uncertainty = compute_uncertainty(softmax_iters)
            
            # 1) build confusion matrix and per-class accuracy
            from sklearn.metrics import confusion_matrix
            # ensure classes = 0,1,...,n_classes-1
            cm = confusion_matrix(labels, preds, labels=np.arange(2))
            # cm[i,i] is # correct in class i; cm.sum(axis=1)[i] is total true class-i
            class_acc = cm.diagonal() / cm.sum(axis=1).astype(float)
    
            detailed_metrics.append({
                'Model': model_title[model_index],
                'rmv_inf_grad': rmv_inf_grad,
                'Fold': fold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Uncertainty': uncertainty,
                'num_epochs_trained': num_epochs_trained,
                'Acc_class_0': class_acc[0],
                'Acc_class_1': class_acc[1]
            })
    
            metrics_list.append({
                'Fold': fold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Uncertainty': uncertainty,
                'Acc_class_0': class_acc[0],
                'Acc_class_1': class_acc[1]
            })
    
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            summary = {
                'Model': model,
                'rmv_inf_grad': rmv_inf_grad,
                'Accuracy_mean': df['Accuracy'].mean(),
                'Accuracy_std': df['Accuracy'].std(),
                'Accuracy_var': df['Accuracy'].var(),
                'Precision_mean': df['Precision'].mean(),
                'Precision_std': df['Precision'].std(),
                'Precision_var': df['Precision'].var(),
                'Recall_mean': df['Recall'].mean(),
                'Recall_std': df['Recall'].std(),
                'Recall_var': df['Recall'].var(),
                'F1_mean': df['F1'].mean(),
                'F1_std': df['F1'].std(),
                'F1_var': df['F1'].var(),
                'AUC_mean': df['AUC'].mean(),
                'AUC_std': df['AUC'].std(),
                'AUC_var': df['AUC'].var(),
                'Uncertainty_mean': df['Uncertainty'].mean(),
                'Uncertainty_std': df['Uncertainty'].std(),
                'Uncertainty_var': df['Uncertainty'].var(),
                'Acc_class_0_mean': df['Acc_class_0'].mean(),
                'Acc_class_1_mean': df['Acc_class_1'].mean()
            }
            results.append(summary)


# Create DataFrames for summary and detailed metrics
results_df = pd.DataFrame(results)
detailed_df = pd.DataFrame(detailed_metrics)

# Optionally save the summary to a CSV file
results_df.to_csv('analysis/tables/rmv_inf_grad_model_evaluation_summary.csv', index=False)
detailed_df.to_csv('analysis/tables/rmv_inf_grad_model_evaluation_detailed.csv', index=False)

print(results_df)
