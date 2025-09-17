import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import entropy
from scipy.special import softmax  # ? added

def compute_uncertainty(softmax_array):
    """Calculate predictive entropy (uncertainty)."""
    mean_softmax = np.mean(softmax_array, axis=0)  # shape: (N, C)
    return np.mean(entropy(mean_softmax.T))        # average entropy across all samples

results = []
detailed_metrics = []

# Iterate over the different hyperparameters
for image_input in [224]:
    for augmentation_type in ['none', 'basic', 'advanced', 'mixup', 'cutmix']:
        for scheduler_type in ['reduce_on_plateau']:
            for criterion_weight in ['None']:
                for optimizer_type in ['Adam']:
                    for optimizer_lr in [0.0001]:
                        base_path = (
                            f"metrics/stain/image_size_{image_input}/augmentation_{augmentation_type}/"
                            f"scheduler_type_{scheduler_type}/criterion_weight_{criterion_weight}/"
                            f"optimizer_type_{optimizer_type}/optimizer_lr_{optimizer_lr}"
                        )
                        for loss_function_index, loss_function in enumerate([
                            'CrossEntropyLoss',
                            'TotalCrossEntropyLoss', 'FocalLoss'
                        ]):
                            loss_function_title = ['CEL', 'TCEL', 'FL']
                            for model_index, model in enumerate([
                                'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152',
                                'Densenet121', 'Densenet169', 'Densenet201', 'Vision'
                            ]):
                                model_title = ['R18', 'R34', 'R50', 'R101', 'R152', 'D121', 'D169', 'D201', 'Vit']
                                metrics_list = []
                                for fold in range(1, 6):
                                    metric_path = (
                                        f"{base_path}/loss_function_{loss_function}/"
                                        f"model_{model}/training_dict_fold{fold}.json"
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

                                    detailed_metrics.append({
                                        'ImageInput': image_input,
                                        'Augmentation': augmentation_type,
                                        'Scheduler': scheduler_type,
                                        'CriterionWeight': criterion_weight,
                                        'OptimizerType': optimizer_type,
                                        'OptimizerLR': optimizer_lr,
                                        'LossFunction': loss_function_title[loss_function_index],
                                        'Model': model_title[model_index],
                                        'Fold': fold,
                                        'Accuracy': accuracy,
                                        'Precision': precision,
                                        'Recall': recall,
                                        'F1': f1,
                                        'AUC': auc,
                                        'Uncertainty': uncertainty,
                                        'num_epochs_trained': num_epochs_trained
                                    })

                                    metrics_list.append({
                                        'Fold': fold,
                                        'Accuracy': accuracy,
                                        'Precision': precision,
                                        'Recall': recall,
                                        'F1': f1,
                                        'AUC': auc,
                                        'Uncertainty': uncertainty
                                    })

                                if metrics_list:
                                    df = pd.DataFrame(metrics_list)
                                    summary = {
                                        'ImageInput': image_input,
                                        'Augmentation': augmentation_type,
                                        'Scheduler': scheduler_type,
                                        'CriterionWeight': criterion_weight,
                                        'OptimizerType': optimizer_type,
                                        'OptimizerLR': optimizer_lr,
                                        'LossFunction': loss_function,
                                        'Model': model,
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
                                        'Uncertainty_var': df['Uncertainty'].var()
                                    }
                                    results.append(summary)

# Create DataFrames for summary and detailed metrics
results_df = pd.DataFrame(results)
detailed_df = pd.DataFrame(detailed_metrics)

# Optionally save the summary to a CSV file
results_df.to_csv('analysis/stain_model_evaluation_summary.csv', index=False)
detailed_df.to_csv('analysis/stain_model_evaluation_detailed.csv', index=False)

print(results_df)
