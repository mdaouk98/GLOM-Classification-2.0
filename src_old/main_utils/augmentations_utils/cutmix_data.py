# src/main_utils/augmentations_utils/cutmix_data.py

import numpy as np
import torch

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]

    # Compute the bounding box for cut
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Uniformly sample cutout region
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lam based on the clipped region
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

    return x, y_a, y_b, lam
