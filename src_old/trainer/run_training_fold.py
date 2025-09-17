# src/trainer/run_training_fold.py

import logging
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
from trainer import train_one_epoch, validate_one_epoch

def run_training_fold(
    config: Any,
    fold: int,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    writer: Optional[SummaryWriter],
    output_path_model: str
) -> Dict[str, Any]:
    """
    Run training for a single fold.
    Returns:
        Dict[str, Any]: Collected metrics and information from the training fold.
    """
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        delta=config.training.early_stopping_delta
    )
    best_val_loss = float('inf')

    # Initialize metric containers with an added key for learning rates.
    fold_metrics = {
        'fold_loss': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'aucs': [],
        'confusion_matrices': [],
        'gradient_norms': [],
        'epoch_times': [],
        'num_epochs_trained': 0,
        'learning_rates': []  # New key to track the learning rate per epoch.
    }

    for epoch in range(1, config.training.epochs + 1):
        epoch_start_time = time.time()
        logging.info(f'Fold {fold}, Epoch {epoch}/{config.training.epochs}')

        # Training Phase.
        train_loss, train_accuracy, avg_grad_norm = train_one_epoch(
            model, device, train_loader, criterion, optimizer, config, fold, epoch, scaler=scaler, writer=writer
        )
        fold_metrics['fold_loss'].append(train_loss)
        fold_metrics['gradient_norms'].append(avg_grad_norm)
        if train_accuracy is not None:
            fold_metrics['train_accuracies'].append(train_accuracy)

        logging.info(
            f"Train Loss: {train_loss:.4f}" +
            (f", Train Accuracy: {train_accuracy:.2f}%" if train_accuracy is not None else "")
        )
        if writer is not None:
            writer.add_scalar(f'Fold{fold}/Train_Loss', train_loss, epoch)
            if train_accuracy is not None:
                writer.add_scalar(f'Fold{fold}/Train_Accuracy', train_accuracy, epoch)
            writer.add_scalar(f'Fold{fold}/Gradient_Norm', avg_grad_norm, epoch)

        # Validation Phase.
        avg_val_loss, val_accuracy, precision, recall, f1, auc, confusion_mat = validate_one_epoch(
            model, device, val_loader, criterion, config, fold, epoch, scaler=scaler, writer=writer
        )
        fold_metrics['val_losses'].append(avg_val_loss)
        if val_accuracy is not None:
            fold_metrics['val_accuracies'].append(val_accuracy)
            fold_metrics['precisions'].append(precision)
            fold_metrics['recalls'].append(recall)
            fold_metrics['f1_scores'].append(f1)
            fold_metrics['aucs'].append(auc)
            fold_metrics['confusion_matrices'].append(confusion_mat)

        logging.info(
            f"Validation Loss: {avg_val_loss:.4f}" +
            (f", Validation Accuracy: {val_accuracy:.2f}%" if val_accuracy is not None else "")
        )
        if val_accuracy is not None:
            logging.info(
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                f"F1-Score: {f1:.4f}, AUC: {auc:.4f}"
            )
        if writer is not None:
            writer.add_scalar(f'Fold{fold}/Val_Loss', avg_val_loss, epoch)
            if val_accuracy is not None:
                writer.add_scalar(f'Fold{fold}/Val_Accuracy', val_accuracy, epoch)
                writer.add_scalar(f'Fold{fold}/Precision', precision, epoch)
                writer.add_scalar(f'Fold{fold}/Recall', recall, epoch)
                writer.add_scalar(f'Fold{fold}/F1_Score', f1, epoch)
                writer.add_scalar(f'Fold{fold}/AUC', auc, epoch)

        # Scheduler step.
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar(f'Fold{fold}/Learning_Rate', current_lr, epoch)
        logging.info(f"Current learning rate: {current_lr}")
        # Append the current learning rate to the metrics dictionary.
        fold_metrics['learning_rates'].append(current_lr)

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            fold_metrics['num_epochs_trained'] = epoch
            break
        else:
            fold_metrics['num_epochs_trained'] = epoch

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        fold_metrics['epoch_times'].append(epoch_duration)
        if writer is not None:
            writer.add_scalar(f'Fold{fold}/Epoch_Time', epoch_duration, epoch)
        logging.info(f"Epoch Time: {epoch_duration:.2f} seconds")

    return fold_metrics
