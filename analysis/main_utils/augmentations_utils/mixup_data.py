# src/main_utils/augmentations_utils/mixup_data.py


import numpy as np
import torch

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Apply MixUp augmentation to a batch of inputs and labels.

    Args:
        x (Tensor): Input tensor of shape (batch_size, ...).
        y (Tensor): Corresponding labels (e.g., class indices or one-hot vectors).
        alpha (float): MixUp Beta distribution parameter. If <= 0, no mixing is applied.
        device (str or torch.device): Device on which to perform index shuffling.

    Returns:
        mixed_x (Tensor): Batch of mixed inputs.
        y_a (Tensor): Original labels for each mixed sample.
        y_b (Tensor): Labels of the paired samples for mixing.
        lam (float): Mixing coefficient ? sampled from Beta(alpha, alpha).
    """

    # 1. Sample mixing coefficient ? from Beta(alpha, alpha) if alpha > 0
    #    Otherwise fallback to no mixing (? = 1.0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # 2. Determine batch size from the first dimension of x
    batch_size = x.size(0)

    # 3. Create a random permutation of the batch indices on the specified device
    index = torch.randperm(batch_size).to(device)

    # 4. Compute the mixed inputs:
    #    mixed_x = ? * x_i + (1 - ?) * x_j,
    #    where j = index[i]
    mixed_x = lam * x + (1.0 - lam) * x[index]

    # 5. Prepare paired labels:
    #    y_a: original labels
    #    y_b: labels of the shuffled inputs
    y_a, y_b = y, y[index]

    # 6. Return mixed inputs, both sets of labels, and the mixing ratio
    return mixed_x, y_a, y_b, lam


