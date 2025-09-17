# src/train_utils/save_metrics_and_checkpoint.py

import json
import logging
from os import makedirs
from os.path import join, dirname
import torch
from main_utils.helpers_utils import convert_to_serializable
from typing import Any, Optional, Tuple, Dict
from torch.amp import GradScaler

def save_metrics_and_checkpoint(config: Any, fold: int, model: torch.nn.Module, optimizer: Any, scheduler: Any,
                                scaler: Optional[GradScaler], metrics_for_fold: Dict[str, Any],
                                checkpoints_dir: str, metrics_dir: str) -> None:
    """
    Save metrics to JSON and model checkpoints.

    Args:
        config (Any): Configuration object.
        fold (int): Current fold number.
        model (torch.nn.Module): Trained model.
        optimizer (Any): Optimizer used.
        scheduler (Any): Learning rate scheduler.
        scaler (Optional[GradScaler]): GradScaler for mixed precision.
        metrics_for_fold (Dict[str, Any]): Dictionary of metrics.
        checkpoints_dir (str): Directory to save model checkpoints.
        metrics_dir (str): Directory to save metrics JSON.
    """
    output_dict_per_fold = {
        'fold_loss': metrics_for_fold.get('fold_loss'),
        'train_accuracies': metrics_for_fold.get('train_accuracies'),
        'val_losses': metrics_for_fold.get('val_losses'),
        'val_accuracies': metrics_for_fold.get('val_accuracies'),
        'precisions': metrics_for_fold.get('precisions'),
        'recalls': metrics_for_fold.get('recalls'),
        'f1_scores': metrics_for_fold.get('f1_scores'),
        'aucs': metrics_for_fold.get('aucs'),
        'confusion_matrices': metrics_for_fold.get('confusion_matrices'),
        'gradient_norms': metrics_for_fold.get('gradient_norms'),
        'epoch_times': metrics_for_fold.get('epoch_times'),
        'num_epochs_trained': metrics_for_fold.get('num_epochs_trained'),
        'all_testing_dict': metrics_for_fold.get('all_testing_dict')
    }
    serializable_output_dict = convert_to_serializable(output_dict_per_fold)
    try:
        metrics_save_path = join(metrics_dir, f"{config.paths.output_path_dict}_fold{fold}.json")
        with open(metrics_save_path, 'w') as f:
            json.dump(serializable_output_dict, f, indent=4)
        logging.info(f"[Fold {fold}] Metrics saved to {metrics_save_path}")
    except Exception as e:
        logging.error(f"[Fold {fold}] Failed to save metrics to JSON: {e}")
        raise

    # Save model state.
    model_save_path = join(checkpoints_dir, f"{config.paths.output_path_model}_fold{fold}.pth")
    makedirs(dirname(model_save_path), exist_ok=True)
    try:
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"[Fold {fold}] Model saved at {model_save_path}")
    except Exception as e:
        logging.error(f"[Fold {fold}] Error saving model: {e}")
        raise

    final_checkpoint_path = join(checkpoints_dir, f"{config.paths.output_path_model}_fold{fold}_final.pth")
    try:
        torch.save({
            'fold': fold,
            'epoch': metrics_for_fold.get('num_epochs_trained'),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': output_dict_per_fold
        }, final_checkpoint_path)
        logging.info(f"[Fold {fold}] Final checkpoint saved at {final_checkpoint_path}")
    except Exception as e:
        logging.error(f"[Fold {fold}] Error saving final checkpoint: {e}")
        raise
