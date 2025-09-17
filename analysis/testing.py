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

path = '/data/mdaouk/Repo2/metrics/augmentation1/image_size_224/augmentation_none/scheduler_type_reduce_on_plateau/criterion_weight_None/optimizer_type_Adam/optimizer_lr_0.0001/loss_function_CrossEntropyLoss/model_Resnet18/'

metrics_list = []
for fold in range(1, 6):
    metric_path = (
        f"{path}training_dict_fold{fold}.json"
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

print(len(data['all_testing_dict']['all_logits']))
for i in range(50):
  print(data['all_testing_dict']['all_logits'][i][100])        
                                    