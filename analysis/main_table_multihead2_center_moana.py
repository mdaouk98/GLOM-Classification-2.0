import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.stats import entropy
from scipy.special import softmax


def compute_uncertainty(softmax_array):
    """Calculate predictive entropy (uncertainty)."""
    mean_softmax = np.mean(softmax_array, axis=0)  # shape: (N, C)
    return np.mean(entropy(mean_softmax.T))        # average entropy across all samples

results = []
detailed_metrics = []
metrics_list = []

from main_utils.train_utils import load_hdf5_data, split_data, DataLoadingError
from main_utils.config import load_config
config_path = 'configs/bias2/scanner/model_Resnet18.yaml'
config = load_config(config_path)

try:
    all_indices, all_labels, all_wsis, all_stain, all_scanner, total_samples = load_hdf5_data(config)
    print(f"Total samples loaded: {total_samples}")
    print(f"All labels shape: {np.array(all_labels).shape}, unique labels: {np.unique(all_labels)}")    
    print(f"All stains shape: {np.array(all_stain).shape}, unique stains: {np.unique(all_stain)}")
    print(f"All scanners shape: {np.array(all_scanner).shape}, unique scanners: {np.unique(all_scanner)}")
except DataLoadingError as e:
    logging.error(f"[Data Loading] {e}")
    

train_val_idx, train_val_lbls, test_idx, test_lbls = split_data(
    all_indices, all_scanner, all_wsis, config
)
print(f"test labaels shape: {np.array(test_lbls).shape}, unique labels: {np.unique(test_lbls)}")

train_val_idx, train_val_lbls, test_idx, test_lbls = split_data(
    all_indices, all_labels, all_wsis, config
)
print(f"test labaels shape: {np.array(test_lbls).shape}, unique labels: {np.unique(test_lbls)}")

test_stains = np.array(all_stain)[test_idx]
unique_test_stains = np.unique(test_stains)
print(
    "Test stain distribution: ",
    {stain: np.sum(test_stains == stain) for stain in unique_test_stains}
)



# Iterate over the different hyperparameters

for model_index, model in enumerate(['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'Densenet121', 'Densenet169', 'Densenet201','efficientnet_b0', 'efficientnet_b1',
                                      'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4','efficientnet_b5',
                                     'regnety_008', 'regnety_016','resnext101_32x8d', 'Vision']):
    model_title = ['R18', 'R34', 'R50', 'R101', 'R152','D121', 'D169', 'D201', 'Eff_b0', 'Eff_b1', 'Eff_b2', 'Eff_b3', 'Eff_b4', 'Eff_b5',
                   'RGy8', 'RGy16', 'RX101_32x8d', 'Vit']
    for scale_multiplier in [0.0,-0.01,-0.05,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5]:
        for cluster in [3]:
            # Build the base path using the current hyperparameters
            base_path = (f"metrics/multihead2/scale_{scale_multiplier}/cluster_{cluster}/model_{model}")
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
                        
                
                
                dict_metrics = {}
                key_list = []
                # ? here: load logits instead of softmax iterations
                for key in data['all_testing_dict']['all_logits'].keys():
                    print(f"Processing {key} for model {model}, scale {scale_multiplier}, cluster {cluster}, fold {fold}")
                    key_list.append(key)
                    logits_iters = np.array(data['all_testing_dict']['all_logits'][key])
                    print(f"logits_iters shape: {logits_iters.shape}")
                    
                    # ? convert logits to probabilities along class axis
                    softmax_iters = softmax(logits_iters, axis=2)

                    if key == 'class':
                        labels = np.array(data['all_testing_dict']['all_labels'][key])
                    else:
                        labels = test_lbls
                    print(f"labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
                
                    
        
                    # Compute mean softmax predictions and predicted labels
                    mean_softmax = np.mean(softmax_iters, axis=0)
                    preds = np.argmax(mean_softmax, axis=1)
        
                    # Compute performance metrics
                    accuracy = accuracy_score(labels, preds)
                    precision = precision_score(labels, preds, average='macro', zero_division=0)
                    recall = recall_score(labels, preds, average='macro', zero_division=0)
                    f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
                    # Compute AUC (if applicable)
                    n_classes = mean_softmax.shape[1]
                    unique_labels = np.unique(labels)
                    
                    try:
                        if n_classes == 2 and len(unique_labels) == 2:
                            # binary case, keep as-is
                            auc = roc_auc_score(labels, mean_softmax[:, 1])
                        else:
                            # multiclass: need full probability matrix + multi_class param
                            # Option A: label-binarize by hand
                            lb = label_binarize(labels, classes=np.arange(n_classes))
                            auc = roc_auc_score(lb, mean_softmax,
                                                multi_class='ovr',
                                                average='macro')
                            # Option B (sklearn =0.23): you can also pass labels directly
                            # auc = roc_auc_score(labels, mean_softmax,
                            #                     multi_class='ovr',
                            #                     average='macro')
                    except ValueError as e:
                        print(f"Error computing AUC : {e}")
                        auc = np.nan
        
                    # Compute uncertainty on the softmax probabilities
                    uncertainty = compute_uncertainty(softmax_iters)
                    
                    
                    # 1) build confusion matrix and per-class accuracy
                    from sklearn.metrics import confusion_matrix
                    # ensure classes = 0,1,...,n_classes-1
                    cm = confusion_matrix(labels, preds, labels=np.arange(n_classes))
                    # cm[i,i] is # correct in class i; cm.sum(axis=1)[i] is total true class-i
                    class_acc = cm.diagonal() / cm.sum(axis=1).astype(float)
                
                    per_stain_metrics = {}

                    if key == 'class':
                        if len(test_stains) != labels.shape[0]:
                            print(
                                "Skipping per-stain metrics: mismatch between"
                                f" stain labels ({len(test_stains)}) and"
                                f" predictions ({labels.shape[0]})."
                            )
                        else:
                            for stain_value in unique_test_stains:
                                stain_mask = test_stains == stain_value
                                if not np.any(stain_mask):
                                    continue

                                labels_subset = labels[stain_mask]
                                preds_subset = preds[stain_mask]
                                softmax_subset = mean_softmax[stain_mask]
                                softmax_iters_subset = softmax_iters[:, stain_mask, :]

                                stain_accuracy = accuracy_score(
                                    labels_subset,
                                    preds_subset
                                )
                                stain_precision = precision_score(
                                    labels_subset,
                                    preds_subset,
                                    average='macro',
                                    zero_division=0
                                )
                                stain_recall = recall_score(
                                    labels_subset,
                                    preds_subset,
                                    average='macro',
                                    zero_division=0
                                )
                                stain_f1 = f1_score(
                                    labels_subset,
                                    preds_subset,
                                    average='macro',
                                    zero_division=0
                                )

                                try:
                                    unique_subset_labels = np.unique(labels_subset)
                                    if (
                                        n_classes == 2
                                        and len(unique_subset_labels) == 2
                                    ):
                                        stain_auc = roc_auc_score(
                                            labels_subset,
                                            softmax_subset[:, 1]
                                        )
                                    else:
                                        lb_subset = label_binarize(
                                            labels_subset,
                                            classes=np.arange(n_classes)
                                        )
                                        stain_auc = roc_auc_score(
                                            lb_subset,
                                            softmax_subset,
                                            multi_class='ovr',
                                            average='macro'
                                        )
                                except ValueError:
                                    stain_auc = np.nan

                                stain_uncertainty = compute_uncertainty(
                                    softmax_iters_subset
                                )

                                cm_subset = confusion_matrix(
                                    labels_subset,
                                    preds_subset,
                                    labels=np.arange(n_classes)
                                )
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    stain_class_acc = np.divide(
                                        cm_subset.diagonal().astype(float),
                                        cm_subset.sum(axis=1).astype(float),
                                        out=np.full(n_classes, np.nan),
                                        where=cm_subset.sum(axis=1) != 0
                                    )

                                per_stain_metrics.update({
                                    f'PerStain_{stain_value}_Accuracy': stain_accuracy,
                                    f'PerStain_{stain_value}_Precision': stain_precision,
                                    f'PerStain_{stain_value}_Recall': stain_recall,
                                    f'PerStain_{stain_value}_F1': stain_f1,
                                    f'PerStain_{stain_value}_AUC': stain_auc,
                                    f'PerStain_{stain_value}_Uncertainty': stain_uncertainty,
                                })

                                for class_index, class_value in enumerate(stain_class_acc):
                                    per_stain_metrics[
                                        f'PerStain_{stain_value}_Acc_class_{class_index}'
                                    ] = class_value

                    # 2) stash everything in your dict_metrics

                    dict_metrics[key]=({
                        'Model': model,
                        'Scale': scale_multiplier,
                        'Cluster': cluster,
                        'Fold': fold,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'AUC': auc,
                        'Uncertainty': uncertainty,
                        # now append one field per class:
                        **{ f'Acc_class_{i}': class_acc[i]
                            for i in range(n_classes) },
                        **per_stain_metrics
                            })
                num_epochs_trained = data['num_epochs_trained']
        
                entry = {
                    'Model': model,
                    'Scale': scale_multiplier,
                    'Cluster': cluster,
                    'Fold':  fold,
                }
                
                for key in key_list:
                    for metric_name, metric_value in dict_metrics[key].items():
                        # skip the Scale/Fold since those are at the top level
                        if metric_name in ('Model','Scale','Cluster','Fold'): continue
                        entry[f'{key}_{metric_name}'] = metric_value
                
                metrics_list.append(entry)
                entry['num_epochs_trained'] = num_epochs_trained
                detailed_metrics.append(entry)
                
                if metrics_list:
                    pass
                    df = pd.DataFrame(metrics_list)
                    summary = {
                        'Model': model,
                        'Scale': scale_multiplier,
                        'Cluster': cluster,
                        'Class_Accuracy_mean': df['class_Accuracy'].mean(),
                        'Class_Accuracy_std': df['class_Accuracy'].std(),
                        'Class_Accuracy_var': df['class_Accuracy'].var(),
                        'Class_Precision_mean': df['class_Precision'].mean(),
                        'Class_Precision_std': df['class_Precision'].std(),
                        'Class_Precision_var': df['class_Precision'].var(),
                        'Class_Recall_mean': df['class_Recall'].mean(),
                        'Class_Recall_std': df['class_Recall'].std(),
                        'Class_Recall_var': df['class_Recall'].var(),
                        'Class_F1_mean': df['class_F1'].mean(),
                        'Class_F1_std': df['class_F1'].std(),
                        'Class_F1_var': df['class_F1'].var(),
                        'Class_AUC_mean': df['class_AUC'].mean(),
                        'Class_AUC_std': df['class_AUC'].std(),
                        'Class_AUC_var': df['class_AUC'].var(),
                        'Class_Uncertainty_mean': df['class_Uncertainty'].mean(),
                        'Class_Uncertainty_std': df['class_Uncertainty'].std(),
                        'Class_Uncertainty_var': df['class_Uncertainty'].var(),
                        'Class_Acc_class_0_mean': df['class_Acc_class_0'].mean(),
                        'Class_Acc_class_1_mean': df['class_Acc_class_1'].mean(),
                        'Stain_Accuracy_mean': df['stain_Accuracy'].mean(),
                        'Stain_Accuracy_std': df['stain_Accuracy'].std(),
                        'Stain_Accuracy_var': df['stain_Accuracy'].var(),
                        'Stain_Precision_mean': df['stain_Precision'].mean(),
                        'Stain_Precision_std': df['stain_Precision'].std(),
                        'Stain_Precision_var': df['stain_Precision'].var(),
                        'Stain_Recall_mean': df['stain_Recall'].mean(),
                        'Stain_Recall_std': df['stain_Recall'].std(),
                        'Stain_Recall_var': df['stain_Recall'].var(),
                        'Stain_F1_mean': df['stain_F1'].mean(),
                        'Stain_F1_std': df['stain_F1'].std(),
                        'Stain_F1_var': df['stain_F1'].var(),
                        'Stain_AUC_mean': df['stain_AUC'].mean(),
                        'Stain_AUC_std': df['stain_AUC'].std(),
                        'Stain_AUC_var': df['stain_AUC'].var(),
                        'Stain_Uncertainty_mean': df['stain_Uncertainty'].mean(),
                        'Stain_Uncertainty_std': df['stain_Uncertainty'].std(),
                        'Stain_Uncertainty_var': df['stain_Uncertainty'].var(),
                        'Stain_Acc_class_0_mean': df['stain_Acc_class_0'].mean(),
                        'Stain_Acc_class_1_mean': df['stain_Acc_class_1'].mean(),
                    }
                    results.append(summary)


# Create DataFrames for summary and detailed metrics
#results_df = pd.DataFrame(results)
detailed_df = pd.DataFrame(detailed_metrics)

# Optionally save the summary to a CSV file
#results_df.to_csv('analysis/tables/multihead2_model_evaluation_summary.csv', index=False)
detailed_df.to_csv('analysis/tables/multihead3_center_model_evaluation_detailed.csv', index=False)

#print(results_df)
