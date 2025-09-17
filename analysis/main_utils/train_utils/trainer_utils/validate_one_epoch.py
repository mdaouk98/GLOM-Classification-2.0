# src/main_utils/train_utils/trainer_utils/validate_one_epoch.py

import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Tuple, Dict, List
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from main_utils.helpers_utils import timeit

# Type alias for clarity: per-head metrics dictionary
Metrics = Dict[str, Dict[str, float]]

def validate_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    criterions: Dict[str, torch.nn.Module],  # head_name -> loss module
    loss_weights: Dict[str, float],           # head_name -> weight scalar
    config: Any,
    fold: int,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, Metrics, Dict[str, int], Dict[str, float]]:
    """
    Validate the model for one epoch, collecting weighted loss and per-head metrics.

    Returns:
      - avg_val_loss (float): weighted loss averaged over all samples
      - metrics (Metrics): per-head metrics dict with keys:
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix'
      - counts (Dict[str,int]): number of samples seen per head
      - avg_head_losses (Dict[str,float]): unweighted average loss per head
    """
    model.eval()

    # --------------------------------------------------------------------------
    # 1) Initialize accumulators
    # --------------------------------------------------------------------------
    total_weighted_loss = 0.0
    total_samples       = 0
    head_loss_accum     = {h: 0.0 for h in criterions}
    all_labels, all_preds, all_probs = {
        h: [] for h in criterions
    }, {
        h: [] for h in criterions
    }, {
        h: [] for h in criterions
    }
    correct_counts = {h: 0 for h in criterions}
    total_counts   = {h: 0 for h in criterions}

    batch_load_times, batch_proc_times = [], []
    skipped_batches: List[int] = []

    pbar = tqdm(
        val_loader,
        desc=f'Fold {fold} | Epoch {epoch} - Validation',
        leave=False
    )

    # --------------------------------------------------------------------------
    # 2) Loop over validation batches
    # --------------------------------------------------------------------------
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(pbar):
            # 2a) Move data to device, time load
            try:
                with timeit() as t0:
                    imgs = imgs.to(device, non_blocking=True)
                    if isinstance(labels, dict):
                        # multi-head: move each label tensor
                        labels = {h: labels[h].to(device, non_blocking=True)
                                  for h in criterions}
                    else:
                        # single-head: wrap in dict for uniformity
                        head = next(iter(criterions))
                        labels = {head: labels.to(device, non_blocking=True)}
                batch_load_times.append(t0())
            except Exception as e:
                logging.error(f"[Validation] batch {batch_idx} load failed: {e}", exc_info=True)
                skipped_batches.append(batch_idx)
                continue

            # 2b) Forward pass and loss computation, time processing
            try:
                with timeit() as t1:
                    # optionally use mixed precision
                    if scaler and device.type == 'cuda':
                        with autocast(device_type='cuda'):
                            raw_outputs = model(imgs, mc_dropout=False)
                    else:
                        raw_outputs = model(imgs, mc_dropout=False)

                    # unify outputs to dict[head ? logits]
                    outputs = (raw_outputs if isinstance(raw_outputs, dict)
                               else {next(iter(criterions)): raw_outputs})

                    # compute weighted loss sum over heads
                    loss = 0.0
                    if scaler is not None and device.type == 'cuda':
                        # force entire loss into FP32
                        with autocast(enabled=False,device_type='cuda'):
                            for h, crit in criterions.items():
                                preds = outputs[h].float()      # up-cast to float32
                                tgt   = labels[h]
                                try:
                                    lh = crit(preds, tgt)
                                except TypeError:
                                    lh = crit(preds)
                                loss += loss_weights[h] * lh
                    else:
                        for h, crit in criterions.items():
                            preds = outputs[h]
                            # if it’s still half, cast to float
                            if preds.dtype == torch.float16:
                                preds = preds.float()
                            tgt = labels[h]
                            try:
                                lh = crit(preds, tgt)
                            except TypeError:
                                lh = crit(preds)
                            loss += loss_weights[h] * lh
                batch_proc_times.append(t1())
            except Exception as e:
                logging.error(f"[Validation] batch {batch_idx} forward/loss failed: {e}", exc_info=True)
                skipped_batches.append(batch_idx)
                continue

            # Skip batches with non-finite loss
            if not torch.isfinite(loss):
                logging.warning(f"[Validation] batch {batch_idx} non-finite loss={loss.item():.6f}, skipping")
                continue

            # 2c) Accumulate overall weighted loss and sample count
            bs = imgs.size(0)
            total_weighted_loss += loss.item() * bs
            total_samples       += bs

            # 2d) Accumulate per-head raw loss and collect predictions/labels/probs
            for h, crit in criterions.items():
                preds = outputs[h]
                # up-cast any FP16 logits to FP32 before calling the loss
                if preds.dtype == torch.float16:
                    preds = preds.float()
                tgt   = labels[h]
                # raw (unweighted) per-head loss in FP32
                try:
                    single_loss = crit(preds, tgt)
                except TypeError:
                    single_loss = crit(preds)
                head_loss_accum[h] += single_loss.item() * bs

                # gather numpy arrays for metrics
                pred_labels = preds.argmax(dim=1).cpu().numpy()
                true_labels = tgt.cpu().numpy()
                probs       = torch.softmax(preds, dim=1).cpu().numpy()

                all_preds[h].append(pred_labels)
                all_labels[h].append(true_labels)
                # for binary, store positive-class prob; else full vector
                all_probs[h].append(probs[:, 1] if probs.shape[1] == 2 else probs)

                correct_counts[h] += (pred_labels == true_labels).sum()
                total_counts[h]   += bs

            # 2e) Update progress bar with interim loss & accuracy
            interim_loss = total_weighted_loss / total_samples
            postfix = {'ValLoss': f"{interim_loss:.4f}"}
            for h in criterions:
                acc = 100.0 * correct_counts[h] / total_counts[h] if total_counts[h] else 0.0
                postfix[f"Acc_{h}"] = f"{acc:.2f}%"
            pbar.set_postfix(postfix)

    # --------------------------------------------------------------------------
    # 3) Log timing and skipped batches
    # --------------------------------------------------------------------------
    if skipped_batches:
        logging.warning(f"[Validation] skipped batches: {skipped_batches}")
    if batch_load_times:
        logging.info(f"[Validation] avg load time: {np.mean(batch_load_times):.4f}s/batch")
    if batch_proc_times:
        logging.info(f"[Validation] avg proc time: {np.mean(batch_proc_times):.4f}s/batch")

    # --------------------------------------------------------------------------
    # 4) Compute final metrics
    # --------------------------------------------------------------------------
    avg_val_loss = (total_weighted_loss / total_samples) if total_samples else float('nan')
    avg_head_losses = {
        h: (head_loss_accum[h] / total_samples) if total_samples else float('nan')
        for h in criterions
    }

    metrics: Metrics = {}
    counts: Dict[str, int] = {}
    for h in criterions:
        y_true = np.concatenate(all_labels[h], axis=0)
        y_pred = np.concatenate(all_preds[h],  axis=0)
        y_prob = np.concatenate(all_probs[h],  axis=0)

        # accuracy
        acc = 100.0 * correct_counts[h] / total_counts[h] if total_counts[h] else float('nan')
        # precision, recall, f1
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        # AUC (handle multiclass vs binary)
        try:
            auc = (roc_auc_score(y_true, y_prob, multi_class='ovr')
                   if y_prob.ndim > 1 else roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float('nan')
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics[h] = {
            'accuracy': acc,
            'precision': p,
            'recall':    r,
            'f1':        f1,
            'auc':       auc,
            'confusion_matrix': cm
        }
        counts[h] = total_counts[h]

        # Optionally log to TensorBoard
        if writer:
            writer.add_scalar(f'Fold{fold}/{h}_Val_Acc',        acc, epoch)
            writer.add_scalar(f'Fold{fold}/{h}_Val_Precision', p,   epoch)
            writer.add_scalar(f'Fold{fold}/{h}_Val_Recall',    r,   epoch)
            writer.add_scalar(f'Fold{fold}/{h}_Val_F1',        f1,  epoch)
            writer.add_scalar(f'Fold{fold}/{h}_Val_AUC',       auc, epoch)

    return avg_val_loss, metrics, counts, avg_head_losses

