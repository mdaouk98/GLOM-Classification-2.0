import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import entropy

# Define loss functions and base path
loss_functions = ['CrossEntropyLoss', 'ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss']
base_path = "metrics/image_size_224/augmentation_none/scheduler_type_reduce_on_plateau/criterion_weight_None/optimizer_type_Adam/optimizer_lr_0.01"

def compute_uncertainty(softmax_array):
    """Calculate predictive entropy (uncertainty)."""
    mean_softmax = np.mean(softmax_array, axis=0)  # shape: (N, C)
    return np.mean(entropy(mean_softmax.T))        # average entropy across all samples

results = []
detailed_metrics = []  # This will store per-fold detailed metrics

for loss_function in loss_functions:
    metrics_list = []
    
    for fold in range(1, 6):
        metric_path = f"{base_path}/loss_function_{loss_function}/model_Resnet18/training_dict_fold{fold}.json"
        with open(metric_path, 'r') as f:
            data = json.load(f)
            
        softmax_iters = np.array(data['all_testing_dict']['all_softmax_iterations'])  # shape: (50, N, C)
        labels = np.array(data['all_testing_dict']['all_labels'])                    # shape: (N,)
        
        # Compute mean softmax prediction over MC Dropout iterations (shape: (N, C))
        mean_softmax = np.mean(softmax_iters, axis=0)
        preds = np.argmax(mean_softmax, axis=1)
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
        # For binary classification, extract positive class probability
        positive_probs = mean_softmax[:, 1]
        try:
            auc = roc_auc_score(labels, positive_probs)
        except ValueError as e:
            print(f"Error computing AUC for loss_function {loss_function}, fold {fold}: {e}")
            auc = np.nan
        
        uncertainty = compute_uncertainty(softmax_iters)
        
        # Store the detailed metrics for this fold
        detailed_metrics.append({
            'LossFunction': loss_function,
            'Fold': fold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'Uncertainty': uncertainty
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
    
    # Create the summary as before (if needed)
    df = pd.DataFrame(metrics_list)
    summary = {
        'LossFunction': loss_function,
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

# Create DataFrames for summary and detailed data
results_df = pd.DataFrame(results)
detailed_df = pd.DataFrame(detailed_metrics)

# Save summary if desired
results_df.to_csv('model_evaluation_summary.csv', index=False)

print(results_df)


##########################################
# Create Radar Chart with Varying Dot Sizes
##########################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

def create_radar_polygon_plot(detailed_df, metrics=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Uncertainty']):
    """
    Create a radar (polar) plot where for each loss function a polygon is drawn
    connecting the mean metric values, with a filled band showing mean +/- std (variance).
    
    Parameters:
      detailed_df: DataFrame with columns: 'LossFunction', 'Fold', and each metric.
      metrics: List of metrics to plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    N = len(metrics)
    # Compute angles for each metric axis (make sure to close the polygon)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # repeat the first angle at end for closure

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    loss_functions = detailed_df['LossFunction'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_functions)))
    
    for i, lf in enumerate(loss_functions):
        group = detailed_df[detailed_df['LossFunction'] == lf]
        mean_vals = []
        std_vals = []
        for metric in metrics:
            mean_vals.append(group[metric].mean())
            std_vals.append(group[metric].std())
        # Close the polygon for the mean and std arrays
        mean_vals += mean_vals[:1]
        std_vals += std_vals[:1]
        
        # Compute outer and inner boundaries: mean +/- std
        outer_boundary = [m + s for m, s in zip(mean_vals, std_vals)]
        inner_boundary = [m - s for m, s in zip(mean_vals, std_vals)]
        
        # Create a closed polygon by concatenating the outer boundary and the inner boundary reversed.
        polygon_angles = angles + angles[::-1]
        polygon_radii = outer_boundary + inner_boundary[::-1]
        
        # Fill the area between mean + std and mean - std
        ax.fill(polygon_angles, polygon_radii, color=colors[i], alpha=0.3, label=f'{lf} +/- std')
        # Plot the mean polygon
        ax.plot(angles, mean_vals, color=colors[i], marker='o', linewidth=2, label=f'{lf} Mean')
    
    # Set the labels for each metric axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_rlabel_position(30)
    ax.yaxis.grid(True, color='grey', linestyle='dashed', alpha=0.5)
    ax.set_title("Polygon Radar Plot with Variance Width", size=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


# Example usage with your detailed_df:
# Assuming detailed_df is already defined (see your code snippet above)
create_radar_polygon_plot(detailed_df)


