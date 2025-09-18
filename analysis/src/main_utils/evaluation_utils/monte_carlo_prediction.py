# src/main_utils/evaluation_utils/monte_carlo_prediction.py

import logging
from typing import Dict, Optional
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def monte_carlo_prediction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_iter: int = 10,
    writer: Optional[SummaryWriter] = None,
    fold: int = 1,
    heads: Optional[Dict[str, int]] = None
) -> Dict[str, Dict]:
    """
    Perform Monte Carlo (MC) prediction by enabling MC Dropout on the final layer
    and collecting **logits** across multiple stochastic forward passes.

    Returns (single-head):
        {
          'all_logits': np.ndarray of shape (n_iter, N, C),
          'all_labels': np.ndarray of shape (N,)
        }

    Returns (multi-head):
        {
          'all_logits': { head_name: np.ndarray of shape (n_iter, N, num_classes) },
          'all_labels': { head_name: np.ndarray of shape (N,) }
        }
    """

    # --------------------------------------------------
    # 1) Prepare model for MC dropout
    # --------------------------------------------------
    model.eval()             # disable BatchNorm stats updates, etc.
    model.to(device)         # move to target device
    # Re-enable dropout layers for stochasticity if accessible
    if hasattr(model, 'dropout') and isinstance(model.dropout, torch.nn.Dropout):
        model.dropout.train()

    # --------------------------------------------------
    # 2) Collect ground-truth labels from the dataloader
    # --------------------------------------------------
    multi_head = (heads is not None) and (len(heads) > 1)
    if multi_head:
        all_labels: Dict[str, list] = {}
    else:
        all_labels: list = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Collecting Labels', leave=False):
            if multi_head:
                # accumulate labels for each head separately
                for head_name in heads:
                    all_labels.setdefault(head_name, []).extend(labels[head_name].cpu().numpy())
            else:
                # single-head: append to a flat list
                all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays and determine total sample count
    if multi_head:
        total_samples = len(next(iter(all_labels.values())))
        for head_name, vals in all_labels.items():
            all_labels[head_name] = np.array(vals, dtype=np.int64)
    else:
        all_labels = np.array(all_labels, dtype=np.int64)
        total_samples = all_labels.shape[0]

    if total_samples == 0:
        raise ValueError("The provided DataLoader is empty.")

    # --------------------------------------------------
    # 3) Infer number of classes per head if not provided
    # --------------------------------------------------
    if not multi_head:
        if heads is None or 'default' not in heads:
            # run one forward pass to get output dimension
            sample_inputs, _ = next(iter(dataloader))
            sample_inputs = sample_inputs.to(device)
            with torch.no_grad():
                sample_logits = model(sample_inputs, mc_dropout=False)
            # handle scalar vs vector outputs
            num_classes = sample_logits.dim() == 1 and 1 or sample_logits.size(1)
            heads = {'default': num_classes}

    # --------------------------------------------------
    # 4) Validate labels within [0, num_classes)
    # --------------------------------------------------
    #if multi_head:
    #    for head_name, label_array in all_labels.items():
    #        C = heads[head_name]
    #        if not np.all((label_array >= 0) & (label_array < C)):
                #raise ValueError(f"Labels for head '{head_name}' exceed its num_classes={C}")
    #else:
    #    C = heads['default']
    #    if not np.all((all_labels >= 0) & (all_labels < C)):
    #        raise ValueError(f"Single-head labels exceed num_classes={C}")

    # --------------------------------------------------
    # 5) Allocate storage for logits across iterations
    # --------------------------------------------------
    if multi_head:
        all_logits: Dict[str, np.ndarray] = {
            head_name: np.zeros((n_iter, total_samples, num_classes), dtype=np.float32)
            for head_name, num_classes in heads.items()
        }
    else:
        C = heads['default']
        all_logits = np.zeros((n_iter, total_samples, C), dtype=np.float32)

    # --------------------------------------------------
    # 6) Monte Carlo sampling loop
    # --------------------------------------------------
    for iter_idx in range(n_iter):
        sample_offset = 0
        pbar = tqdm(
            dataloader,
            desc=f'Fold {fold} MC Iter {iter_idx+1}/{n_iter}',
            leave=False
        )
        for inputs, _labels in pbar:
            inputs = inputs.to(device)
            with torch.no_grad():
                logits = model(inputs, mc_dropout=True)

            batch_size = inputs.size(0)
            if multi_head:
                # distribute each head's logits
                for head_name, logit_tensor in logits.items():
                    arr = logit_tensor.cpu().numpy()
                    all_logits[head_name][iter_idx, sample_offset:sample_offset+batch_size, :] = arr
            else:
                # single head: flatten if necessary
                arr = logits.dim() == 1 and logits.cpu().numpy().reshape(-1, 1) or logits.cpu().numpy()
                all_logits[iter_idx, sample_offset:sample_offset+batch_size] = arr

            sample_offset += batch_size

        # Ensure we saw exactly all samples this iteration
        if sample_offset != total_samples:
            raise RuntimeError(
                f"Iteration {iter_idx+1}: expected {total_samples} samples but got {sample_offset}"
            )

    # --------------------------------------------------
    # 7) Optional TensorBoard logging
    # --------------------------------------------------
    if writer is not None:
        writer.add_scalar(f'Fold{fold}/MC_Iterations', n_iter, fold)
        writer.add_scalar(f'Fold{fold}/Num_Samples', total_samples, fold)

    return {
        'all_logits': all_logits,
        'all_labels': all_labels
    }
