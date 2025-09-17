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
    """
    Compute uncertainty (predictive entropy). If softmax_array is 3D (multiple iterations),
    we average the predictions over iterations first. If 2D (a single iteration), we compute entropy directly.
    """
    if softmax_array.ndim == 3:  # Multiple MC iterations: shape (iterations, N, C)
        mean_softmax = np.mean(softmax_array, axis=0)
        return np.mean(entropy(mean_softmax.T))
    elif softmax_array.ndim == 2:  # Single MC iteration: shape (N, C)
        return np.mean(entropy(softmax_array.T))
    else:
        raise ValueError("softmax_array must be 2D or 3D")

# --- Compute per-MC iteration metrics (50 iterations per fold, total 250 iterations per loss function) ---
detailed_metrics_mc = []  # To store metrics per MC iteration

for loss_function in loss_functions:
    for fold in range(1, 6):
        metric_path = f"{base_path}/loss_function_{loss_function}/model_Resnet18/training_dict_fold{fold}.json"
        with open(metric_path, 'r') as f:
            data = json.load(f)
            
        # Shape: (50, N, C) -- 50 MC iterations per fold.
        softmax_iters = np.array(data['all_testing_dict']['all_softmax_iterations'])
        labels = np.array(data['all_testing_dict']['all_labels'])
        
        # For each MC iteration, compute the metrics using the iteration's softmax output.
        num_iterations = softmax_iters.shape[0]
        for mc in range(num_iterations):
            # Get the softmax predictions for this MC iteration (shape: (N, C))
            softmax = softmax_iters[mc, :, :]
            preds = np.argmax(softmax, axis=1)
            
            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average='macro', zero_division=0)
            recall = recall_score(labels, preds, average='macro', zero_division=0)
            f1 = f1_score(labels, preds, average='macro', zero_division=0)
            
            # For binary classification, use the positive class probability.
            positive_probs = softmax[:, 1]
            try:
                auc = roc_auc_score(labels, positive_probs)
            except ValueError as e:
                print(f"Error computing AUC for loss_function {loss_function}, fold {fold}, MC iteration {mc+1}: {e}")
                auc = np.nan
            
            # Compute uncertainty for the single MC iteration
            uncertainty = compute_uncertainty(softmax)
            
            detailed_metrics_mc.append({
                'LossFunction': loss_function,
                'Fold': fold,
                'MC_Iteration': mc + 1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Uncertainty': uncertainty
            })

# Create DataFrame with all 250 MC iterations per loss function.
detailed_df_mc = pd.DataFrame(detailed_metrics_mc)

# (Optional) Save the detailed MC iteration metrics to a CSV.
detailed_df_mc.to_csv('model_evaluation_detailed_MC.csv', index=False)
print(detailed_df_mc.head())

# --- Radar Polygon Plot Using Variance Across 250 MC Iterations ---
def create_radar_polygon_plot(detailed_df, metrics=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Uncertainty']):
    """
    Create a radar (polar) plot where, for each loss function, a polygon is drawn connecting the mean metric values,
    with a filled band showing mean +/- std computed over all MC iterations.
    
    Parameters:
      detailed_df: DataFrame with columns: 'LossFunction', 'Fold', 'MC_Iteration', and each metric.
      metrics: List of metric names to plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    N = len(metrics)
    # Compute base angles for each metric (close the polygon by repeating the first angle).
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

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
        # Close the polygon for mean and std values.
        mean_vals += mean_vals[:1]
        std_vals += std_vals[:1]
        
        # Compute outer and inner boundaries: mean +/- std.
        outer_boundary = [m + s for m, s in zip(mean_vals, std_vals)]
        inner_boundary = [m - s for m, s in zip(mean_vals, std_vals)]
        
        # Create a closed polygon by concatenating outer and reversed inner boundaries.
        polygon_angles = angles + angles[::-1]
        polygon_radii = outer_boundary + inner_boundary[::-1]
        
        # Fill the area between (mean + std) and (mean - std)
        ax.fill(polygon_angles, polygon_radii, color=colors[i], alpha=0.3, label=f'{lf} +/- std')
        # Plot the mean polygon
        ax.plot(angles, mean_vals, color=colors[i], marker='o', linewidth=2, label=f'{lf} Mean')
    
    # Set labels on each metric axis.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_rlabel_position(30)
    ax.yaxis.grid(True, color='grey', linestyle='dashed', alpha=0.5)
    ax.set_title("Polygon Radar Plot with Variance over 250 MC Iterations", size=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# Plot the radar polygon using the MC iteration detailed metrics.
create_radar_polygon_plot(detailed_df_mc)
