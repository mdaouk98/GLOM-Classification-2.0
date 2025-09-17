# src/trainer/validate_one_epoch.py

import logging
import torch
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Optional, Tuple, Any, List
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

def validate_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    config: Any,
    fold: int,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[List[List[int]]]]:
    """
    Validate model for one epoch.
    Returns:
        avg_val_loss (float): Average validation loss.
        val_accuracy (Optional[float]): Validation accuracy if applicable.
        precision (Optional[float]): Precision score.
        recall (Optional[float]): Recall score.
        f1 (Optional[float]): F1 score.
        auc (Optional[float]): AUC score.
        confusion_mat (Optional[list]): Confusion matrix.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    # Lists to store timing measurements for validation.
    batch_load_times = []
    batch_process_times = []
    prev_time = time.time()

    progress_bar = tqdm(
        val_loader,
        desc=f'Fold {fold} | Epoch {epoch} - Validation',
        leave=False
    )

    requires_targets = config.loss_function.lower() not in ['reversecrossentropyloss']

    with torch.no_grad():
        for inputs, labels in progress_bar:
            current_time = time.time()
            load_time = current_time - prev_time
            batch_load_times.append(load_time)
            process_start_time = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs, mc_dropout=False)
                loss = criterion(outputs, labels) if requires_targets else criterion(outputs)

            process_end_time = time.time()
            process_time = process_end_time - process_start_time
            batch_process_times.append(process_time)

            val_loss += loss.item() * inputs.size(0)
            if requires_targets:
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1].to(torch.float32)  # Assuming binary classification
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                current_val_loss = val_loss / val_total
                current_val_acc = 100.0 * val_correct / val_total
                progress_bar.set_postfix({
                    'Val Loss': f'{current_val_loss:.4f}', 
                    'Val Acc': f'{current_val_acc:.2f}%',
                    'LoadT (s)': f'{load_time:.3f}',
                    'ProcT (s)': f'{process_time:.3f}'
                })
            else:
                val_total += inputs.size(0)
                progress_bar.set_postfix({
                    'Val Loss': f'{val_loss / val_total:.4f}',
                    'LoadT (s)': f'{load_time:.3f}',
                    'ProcT (s)': f'{process_time:.3f}'
                })

            prev_time = time.time()

    avg_load_time = sum(batch_load_times) / len(batch_load_times) if batch_load_times else 0.0
    avg_process_time = sum(batch_process_times) / len(batch_process_times) if batch_process_times else 0.0
    logging.info(f"Fold {fold} Epoch {epoch} - Average DataLoader retrieval time (Validation): {avg_load_time:.4f} sec per batch")
    logging.info(f"Fold {fold} Epoch {epoch} - Average processing time (Validation): {avg_process_time:.4f} sec per batch")

    avg_val_loss = val_loss / len(val_loader.dataset)
    if requires_targets:
        val_accuracy = 100.0 * val_correct / val_total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float('nan')
        confusion_mat = confusion_matrix(all_labels, all_preds).tolist()
    else:
        val_accuracy = None
        precision = recall = f1 = auc = None
        confusion_mat = None

    return avg_val_loss, val_accuracy, precision, recall, f1, auc, confusion_mat
