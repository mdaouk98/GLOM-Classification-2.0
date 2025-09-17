# src/main_utils/train_utils/save_metrics_and_checkpoint.py

import json
import logging
from os import makedirs
from os.path import join
import torch
from main_utils.helpers_utils import convert_to_serializable
from typing import Any, Optional, Dict
from torch.amp import GradScaler

def save_metrics_and_checkpoint(
    config: Any,
    fold: int,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Optional[GradScaler],
    metrics_for_fold: Dict[str, Any],
    checkpoints_dir: str,
    metrics_dir: str
) -> None:
    """
    Save training/validation metrics to JSON and (optionally) checkpoint model state.

    Args:
        config:            Configuration object with path settings.
        fold:              Current fold index.
        model:             Trained PyTorch model.
        optimizer:         Optimizer used during training.
        scheduler:         LR scheduler used.
        scaler:            GradScaler for mixed precision (optional).
        metrics_for_fold:  Dictionary of recorded metrics for this fold.
        checkpoints_dir:   Directory in which to save model checkpoints.
        metrics_dir:       Directory in which to save metrics JSON files.
    """
    # ----------------------------------------
    # 1) Prepare subset of metrics to serialize
    # ----------------------------------------
    output_dict = {
        'fold_loss':          metrics_for_fold.get('fold_loss'),
        'train_accuracies':   metrics_for_fold.get('train_accuracies'),
        'val_losses':         metrics_for_fold.get('val_losses'),
        'val_accuracies':     metrics_for_fold.get('val_accuracies'),
        'precisions':         metrics_for_fold.get('precisions'),
        'recalls':            metrics_for_fold.get('recalls'),
        'f1_scores':          metrics_for_fold.get('f1_scores'),
        'aucs':               metrics_for_fold.get('aucs'),
        'confusion_matrices': metrics_for_fold.get('confusion_matrices'),
        'gradient_norms':     metrics_for_fold.get('gradient_norms'),
        'epoch_times':        metrics_for_fold.get('epoch_times'),
        'num_epochs_trained': metrics_for_fold.get('num_epochs_trained'),
        'all_testing_dict':   metrics_for_fold.get('all_testing_dict')
    }

    # Convert NumPy arrays and other non-serializable objects to Python types
    serializable_metrics = convert_to_serializable(output_dict)

    # ----------------------------------------
    # 2) Save metrics JSON
    # ----------------------------------------
    try:
        # Ensure the metrics directory exists
        makedirs(metrics_dir, exist_ok=True)

        metrics_path = join(
            metrics_dir,
            f"{config.paths.output_path_dict}_fold{fold}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logging.info(f"[Fold {fold}] Metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"[Fold {fold}] Failed to save metrics JSON: {e}", exc_info=True)
        raise

    # ----------------------------------------
    # 3) (Optional) Save model checkpoint
    # ----------------------------------------
#    try:
#        # Ensure the checkpoints directory exists
#        makedirs(checkpoints_dir, exist_ok=True)
#
#        checkpoint_path = join(checkpoints_dir, f"model_fold{fold}.pt")
#        # Build checkpoint state dict
#        ckpt = {
#            'model_state':     model.state_dict(),
#            'optimizer_state': optimizer.state_dict(),
#            'scheduler_state': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
#            'scaler_state':    scaler.state_dict()    if scaler is not None else None,
#            'fold':            fold,
#            'config':          config.model_dump()    # serialize config if Pydantic
#        }
#        torch.save(ckpt, checkpoint_path)
#        logging.info(f"[Fold {fold}] Checkpoint saved to {checkpoint_path}")
#    except Exception as e:
#        logging.error(f"[Fold {fold}] Failed to save checkpoint: {e}", exc_info=True)
#        raise

    # Save model state.
#    model_save_path = join(checkpoints_dir, f"{config.paths.output_path_model}_fold{fold}.pth")
#    makedirs(dirname(model_save_path), exist_ok=True)
#    try:
#        torch.save(model.state_dict(), model_save_path)
#        logging.info(f"[Fold {fold}] Model saved at {model_save_path}")
#    except Exception as e:
#        logging.error(f"[Fold {fold}] Error saving model: {e}")
#        raise

#    final_checkpoint_path = join(checkpoints_dir, f"{config.paths.output_path_model}_fold{fold}_final.pth")
#    try:
#        torch.save({
#            'fold': fold,
#            'epoch': metrics_for_fold.get('num_epochs_trained'),
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'scheduler_state_dict': scheduler.state_dict(),
#            'scaler_state_dict': scaler.state_dict() if scaler else None,
#            'metrics': output_dict_per_fold
#        }, final_checkpoint_path)
#        logging.info(f"[Fold {fold}] Final checkpoint saved at {final_checkpoint_path}")
#    except Exception as e:
#        logging.error(f"[Fold {fold}] Error saving final checkpoint: {e}")
#        raise
