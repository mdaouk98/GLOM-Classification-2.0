# src/main_utils/train_utils/trainer_utils/validate_one_epoch.py

import logging
import torch
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Optional, Tuple, Any, List, Union
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from main_utils.helpers_utils import move_to_device, timeit

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
    Validate the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be validated.
        device (torch.device): The device to run validation on.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        config (Any): Configuration object with training parameters.
        fold (int): Current fold number.
        epoch (int): Current epoch number.
        scaler (Optional[torch.cuda.amp.GradScaler]): Gradient scaler for mixed precision, if used.
        writer (Optional[SummaryWriter]): TensorBoard writer for logging.

    Returns:
        Tuple containing:
            - avg_val_loss (float): Average validation loss.
            - val_accuracy (Optional[float]): Validation accuracy, if applicable.
            - precision (Optional[float]): Precision score.
            - recall (Optional[float]): Recall score.
            - f1 (Optional[float]): F1 score.
            - auc (Optional[float]): AUC score.
            - confusion_mat (Optional[List[List[int]]]): Confusion matrix.
    """
    model.eval()
    val_loss: float = 0.0
    val_correct: int = 0
    val_total: int = 0
    all_labels: List[Union[int, float]] = []
    all_preds: List[Union[int, float]] = []
    all_probs: List[Union[float, List[float]]] = []  # For storing probability estimates

    # Lists to store timing measurements for validation.
    batch_load_times: List[float] = []
    batch_process_times: List[float] = []

    progress_bar = tqdm(
        val_loader,
        desc=f'Fold {fold} | Epoch {epoch} - Validation',
        leave=False
    )

    # Determine if target-based metrics are required.
    requires_targets: bool = config.loss_function.lower() not in ['reversecrossentropyloss']
    # Get the number of classes; default to 2 (binary classification) if not specified.
    num_classes: int = getattr(config.training, 'num_classes', 2)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            try:
                # Measure DataLoader retrieval time using a context manager.
                with timeit() as get_load_time:
                    inputs, labels = move_to_device((inputs, labels), device)
                load_time: float = get_load_time()
                batch_load_times.append(load_time)

                # Measure processing time for the forward pass and loss calculation.
                with timeit() as get_process_time:
                    if scaler is not None and device.type == 'cuda':
                        # Use mixed precision if scaler is provided.
                        with autocast('cuda'):
                            outputs = model(inputs, mc_dropout=False)
                            loss = criterion(outputs, labels) if requires_targets else criterion(outputs)
                    else:
                        outputs = model(inputs, mc_dropout=False)
                        loss = criterion(outputs, labels) if requires_targets else criterion(outputs)
                process_time: float = get_process_time()
                batch_process_times.append(process_time)

                # Accumulate loss.
                val_loss += loss.item() * inputs.size(0)

                if requires_targets:
                    # Get predictions and probability estimates.
                    _, predicted = torch.max(outputs, 1)
                    if num_classes > 2:
                        # Multi-class classification: store full probability distribution.
                        probabilities = torch.softmax(outputs, dim=1)
                        all_probs.extend(probabilities.cpu().numpy().tolist())
                    else:
                        # Binary classification: store probability for the positive class.
                        probabilities = torch.softmax(outputs, dim=1)[:, 1].to(torch.float32)
                        all_probs.extend(probabilities.cpu().numpy())

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    current_val_loss: float = val_loss / val_total
                    current_val_acc: float = 100.0 * val_correct / val_total

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
            except Exception as e:
                logging.error(
                    f"Error processing validation batch {batch_idx} in epoch {epoch} on fold {fold}: {e}",
                    exc_info=True
                )
                continue

    # Compute average times per batch.
    avg_load_time: float = sum(batch_load_times) / len(batch_load_times) if batch_load_times else 0.0
    avg_process_time: float = sum(batch_process_times) / len(batch_process_times) if batch_process_times else 0.0
    logging.info(f"Fold {fold} Epoch {epoch} - Average DataLoader retrieval time (Validation): {avg_load_time:.4f} sec per batch")
    logging.info(f"Fold {fold} Epoch {epoch} - Average processing time (Validation): {avg_process_time:.4f} sec per batch")

    avg_val_loss: float = val_loss / len(val_loader.dataset)
    if requires_targets and val_total > 0:
        val_accuracy: float = 100.0 * val_correct / val_total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        # Attempt to calculate AUC, adapting for binary or multi-class cases.
        try:
            all_probs_np: np.ndarray = np.array(all_probs)
            if num_classes > 2:
                auc: float = roc_auc_score(all_labels, all_probs_np, multi_class='ovr')
            else:
                auc: float = roc_auc_score(all_labels, all_probs_np)
        except Exception as e:
            logging.warning(f"Could not compute AUC: {e}")
            auc = float('nan')
        confusion_mat: List[List[int]] = confusion_matrix(all_labels, all_preds).tolist()
    else:
        val_accuracy = None
        precision = None
        recall = None
        f1 = None
        auc = None
        confusion_mat = None

    return avg_val_loss, val_accuracy, precision, recall, f1, auc, confusion_mat
