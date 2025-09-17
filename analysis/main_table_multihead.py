import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.stats import entropy
from scipy.special import softmax

parser = argparse.ArgumentParser(description='Analyze Bayesian Models.')
parser.add_argument('--multihead_type', type=str, default=None,
                    help='')
parser.add_argument('--loss_function', type=str, default=None,
                    help='')
parser.add_argument('--model', type=str, default=None,
                    help='')
args = parser.parse_args()           
                    

def compute_uncertainty(softmax_array):
    """Calculate predictive entropy (uncertainty)."""
    mean_softmax = np.mean(softmax_array, axis=0)  # shape: (N, C)
    return np.mean(entropy(mean_softmax.T))        # average entropy across all samples

results = []
metrics    = ['Accuracy','Precision','Recall','F1','AUC','Uncertainty']
detailed_metrics = []

multihead_type = args.multihead_type
loss_function = args.loss_function
model = args.model

# Iterate over the different hyperparameters
# Build the base path using the current hyperparameters

if multihead_type.endswith('_t'):
    scale_order = [0,'n_1','n_5','n1','n1_5','n2','n2_5','n5','n10']
else:
    scale_order = ['n4','n3','n2','n1',0,'p1','p2','p3','p4']
base_path = (f"metrics/multihead_{multihead_type}/{loss_function}/{model}")
for scale_index, scale in enumerate(scale_order):
    model_title = ['n4','n3','n2','n1',0,'p1','p2','p3','p4']
    metrics_list = []
    for fold in range(1, 6):
        metric_path = (
            f"{base_path}/scale_{scale}/training_dict_fold{fold}.json"
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
        
        print('data:',data.keys())
        print('testing_dict',data['all_testing_dict'].keys())
        print('all_logits',data['all_testing_dict']['all_logits'].keys())
        for key in data['all_testing_dict']['all_logits'].keys():
            print(key)
        
        dict_metrics = {}
        key_list = []
        # ? here: load logits instead of softmax iterations
        for key in data['all_testing_dict']['all_logits'].keys():
            key_list.append(key)
            logits_iters = np.array(data['all_testing_dict']['all_logits'][key])
            # ? convert logits to probabilities along class axis
            softmax_iters = softmax(logits_iters, axis=2)

            labels = np.array(data['all_testing_dict']['all_labels'][key])

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
        
            # 2) stash everything in your dict_metrics
            
            dict_metrics[key]=({
                'Scale': scale,
                'Fold': fold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Uncertainty': uncertainty,
                # now append one field per class:
                **{ f'Acc_class_{i}': class_acc[i]
                    for i in range(n_classes) }
                    })
        num_epochs_trained = data['num_epochs_trained']

        entry = {
            'Scale': scale,
            'Fold':  fold,
        }
        
        for key in key_list:
            for metric_name, metric_value in dict_metrics[key].items():
                # skip the Scale/Fold since those are at the top level
                if metric_name in ('Scale','Fold'): continue
                entry[f'{key}_{metric_name}'] = metric_value
        
        metrics_list.append(entry)
        entry['num_epochs_trained'] = num_epochs_trained
        detailed_metrics.append(entry)

        

#    if metrics_list:
#        df = pd.DataFrame(metrics_list)
#        summary = {
#            'Scale': scale,
#            f'{key_list[0]}_Accuracy_mean': df[f'{key_list[0]}_Accuracy'].mean(),
#            f'{key_list[0]}_Accuracy_std': df[f'{key_list[0]}_Accuracy'].std(),
#            'Accuracy_var': df['Accuracy'].var(),
#            'Precision_mean': df['Precision'].mean(),
#            'Precision_std': df['Precision'].std(),
#            'Precision_var': df['Precision'].var(),
#            'Recall_mean': df['Recall'].mean(),
#            'Recall_std': df['Recall'].std(),
#            'Recall_var': df['Recall'].var(),
#            'F1_mean': df['F1'].mean(),
#            'F1_std': df['F1'].std(),
#            'F1_var': df['F1'].var(),
#            'AUC_mean': df['AUC'].mean(),
#            'AUC_std': df['AUC'].std(),
#            'AUC_var': df['AUC'].var(),
#            'Uncertainty_mean': df['Uncertainty'].mean(),
#            'Uncertainty_std': df['Uncertainty'].std(),
#            'Uncertainty_var': df['Uncertainty'].var()
#        }
#        results.append(summary)


# Create DataFrames for summary and detailed metrics
#results_df = pd.DataFrame(results)
detailed_df = pd.DataFrame(detailed_metrics)

# Optionally save the summary to a CSV file
#results_df.to_csv(f'analysis/multihead_{multihead_path_type}_evaluation_summary.csv', index=False)
detailed_df.to_csv(f'analysis/tables/multihead_{multihead_type}_{loss_function}_{model}_evaluation_detailed.csv', index=False)

#print(results_df)
