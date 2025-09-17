# src/main_utils/augmentations_utils/cutmix_data.py

import numpy as np
import torch

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    Apply CutMix augmentation to a batch of images and labels.

    Args:
        x (Tensor): Input image batch of shape (batch_size, channels, height, width).
        y (Tensor): Corresponding labels of shape (batch_size, ...).
        alpha (float): Parameter for sampling from the Beta distribution.
                       Higher alpha ? more diverse mixing. If <= 0, no mixing.
        device (str or torch.device): Device on which to perform the operation.

    Returns:
        x_mixed (Tensor): Augmented image batch.
        y_a (Tensor): Original labels for the first part of each mixed sample.
        y_b (Tensor): Labels of the randomly paired samples.
        lam (float): Mixing ratio used for the batch (after clipping).
    """

    # 1. Sample mixing coefficient ? from Beta(alpha, alpha) if alpha > 0
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0  # No mixing when alpha <= 0

    # 2. Get batch dimensions
    batch_size, channels, height, width = x.size()

    # 3. Generate a random permutation of the batch indices
    index = torch.randperm(batch_size).to(device)

    # 4. Prepare paired labels: y_a is original, y_b from shuffled batch
    y_a, y_b = y, y[index]

    # 5. Compute the dimensions of the patch to cut and mix
    #    cut_ratio defines the proportion of the area to be replaced
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    # 6. Randomly choose the center point (cx, cy) of the patch
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    # 7. Compute the bounding box coordinates (x1, y1) to (x2, y2)
    #    Use np.clip to ensure they lie within image boundaries
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    # 8. Replace the patch in each image with the corresponding patch from the shuffled batch
    #    This modifies x in-place
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 9. Recompute ? based on the actual area of the patch (in case clipping occurred)
    patch_area = (x2 - x1) * (y2 - y1)
    total_area = width * height
    lam = 1.0 - (patch_area / total_area)

    return x, y_a, y_b, lam

