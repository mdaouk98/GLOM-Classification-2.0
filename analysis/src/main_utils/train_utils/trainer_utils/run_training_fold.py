# src/main_utils/train_utils/trainer_utils/run_training_fold.py

import logging
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from typing import Any, Dict, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from main_utils.train_utils.trainer_utils import EarlyStopping, train_one_epoch, validate_one_epoch

def run_training_fold(
    config: Any,
    fold: int,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterions: Dict[str, torch.nn.Module],
    loss_weights: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    writer: Optional[SummaryWriter],
    output_path_model: str
) -> Dict[str, Any]:
    """
    Run training and validation for a single cross-validation fold.

    Args:
        config:            Configuration with training hyperparameters.
        fold:              Fold index.
        model:             Neural network to train.
        device:            Computation device (CPU/GPU).
        train_loader:      DataLoader for training data.
        val_loader:        DataLoader for validation data.
        criterions:        Dict mapping head names -> loss modules.
        loss_weights:      Dict mapping head names -> weight scalars.
        optimizer:         Optimizer instance.
        scheduler:         LR scheduler (can be ReduceLROnPlateau).
        scaler:            GradScaler for mixed precision.
        writer:            TensorBoard writer (optional).
        output_path_model: Path to save the best model (not used here but could be).
    
    Returns:
        fold_metrics: Dict of training/validation metrics collected per epoch.
    """
    # -------------------------------
    # 1) Setup early stopping and metrics storage
    # -------------------------------
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        delta=config.training.early_stopping_delta
    )
    fold_metrics = {
        'fold_loss': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_metrics_per_epoch': [],
        'gradient_norms': [],
        'epoch_times': [],
        'learning_rates': [],
        'num_epochs_trained': 0,
    }

    # -------------------------------
    # 2) Training loop over epochs
    # -------------------------------
    for epoch in range(1, config.training.epochs + 1):
        epoch_start = time.time()
        logging.info(f"Fold {fold} - Epoch {epoch}/{config.training.epochs}")

        # ----- 2a) Train one epoch -----
        try:
            train_loss, train_accs, grad_norm = train_one_epoch(
                model, device, train_loader,
                criterions, loss_weights,
                optimizer, config, fold, epoch,
                scaler=scaler, writer=writer
            )
        except Exception as e:
            logging.error(f"Training error at epoch {epoch}: {e}", exc_info=True)
            break

        # Record training metrics
        fold_metrics['fold_loss'].append(train_loss)
        fold_metrics['gradient_norms'].append(grad_norm)
        fold_metrics['train_accuracies'].append(train_accs)
        logging.info(
            f" Train Loss: {train_loss:.4f} | " +
            ", ".join(f"{h}={train_accs[h]:.2f}%" for h in train_accs)
        )
        if writer:
            writer.add_scalar(f'Fold{fold}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'Fold{fold}/Gradient_Norm', grad_norm, epoch)
            for head, acc in train_accs.items():
                writer.add_scalar(f'Fold{fold}/{head}_Train_Acc', acc, epoch)

        # ----- 2b) Validate one epoch -----
        try:
            val_loss, val_metrics, val_counts, head_losses = validate_one_epoch(
                model, device, val_loader,
                criterions, loss_weights,
                config, fold, epoch,
                scaler=scaler, writer=writer
            )
        except Exception as e:
            logging.error(f"Validation error at epoch {epoch}: {e}", exc_info=True)
            break

        # Record validation metrics
        fold_metrics['val_losses'].append(val_loss)
        fold_metrics['val_metrics_per_epoch'].append(val_metrics)
        logging.info(f" Validation Loss: {val_loss:.4f} | " +
                     " | ".join(f"{h} Acc={val_metrics[h]['accuracy']:.2f}%"
                                for h in val_metrics))
        if writer:
            writer.add_scalar(f'Fold{fold}/Val_Loss', val_loss, epoch)
            for head, m in val_metrics.items():
                writer.add_scalar(f'Fold{fold}/{head}_Val_Acc', m['accuracy'], epoch)
                writer.add_scalar(f'Fold{fold}/{head}_Precision', m['precision'], epoch)
                writer.add_scalar(f'Fold{fold}/{head}_Recall', m['recall'], epoch)
                writer.add_scalar(f'Fold{fold}/{head}_F1', m['f1'], epoch)
                writer.add_scalar(f'Fold{fold}/{head}_AUC', m['auc'], epoch)

        # ----- 2c) Scheduler step -----
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        fold_metrics['learning_rates'].append(current_lr)
        logging.info(f" Learning rate: {current_lr:.6f}")
        if writer:
            writer.add_scalar(f'Fold{fold}/Learning_Rate', current_lr, epoch)

        # ----- 2d) Early stopping check -----
        # Choose primary head for early stopping
        primary_head = (
            config.multihead.labels[0]
            if getattr(config.model, 'multihead', False)
            else config.model.label
        )
        primary_loss = head_losses[primary_head]
        early_stopping(primary_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            fold_metrics['num_epochs_trained'] = epoch
            break

        # ----- 2e) Record epoch duration -----
        epoch_duration = time.time() - epoch_start
        fold_metrics['epoch_times'].append(epoch_duration)
        fold_metrics['num_epochs_trained'] = epoch
        logging.info(f" Epoch duration: {epoch_duration:.2f}s")
        if writer:
            writer.add_scalar(f'Fold{fold}/Epoch_Time', epoch_duration, epoch)

    return fold_metrics

