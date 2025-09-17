# src/main_utils/train_utils/train_and_evaluate_fold.py

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from os.path import join
from sklearn.model_selection import StratifiedKFold
from main_utils.models_utils import get_bayesian_model
from main_utils.hyperparameters_utils import initialize_optimizer, initialize_scheduler, initialize_loss_function
from main_utils.train_utils.trainer_utils import run_training_fold
from main_utils.evaluation_utils import monte_carlo_prediction
from main_utils.train_utils.dataloader_utils import create_dataloader
from typing import Any, Optional, Tuple, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from torch.utils.data import DataLoader

def train_and_evaluate_fold(config: Any, fold: int, train_val_indices: np.ndarray, train_val_labels: np.ndarray,
                            device: torch.device, transform_for_train: Optional[Any], transform_for_val: Optional[Any],
                            processor: Optional[Any], for_vision: bool, checkpoint_data: Optional[Dict[str, Any]],
                            checkpoints_dir: str, writer: Optional[SummaryWriter], scaler: Optional[GradScaler],
                            test_loader: DataLoader) -> Tuple[nn.Module, Any, Any, Dict[str, Any]]:
    """
    Train and evaluate a single fold using cross-validation.

    Args:
        config (Any): Configuration object.
        fold (int): Current fold number.
        train_val_indices (np.ndarray): Indices for training/validation.
        train_val_labels (np.ndarray): Corresponding labels.
        device (torch.device): Device to run training on.
        transform_for_train (Optional[Any]): Training data augmentation.
        transform_for_val (Optional[Any]): Validation data augmentation.
        processor (Optional[Any]): Vision processor if needed.
        for_vision (bool): Flag for vision model.
        checkpoint_data (Optional[Dict[str, Any]]): Checkpoint data if resuming.
        checkpoints_dir (str): Directory to save checkpoints.
        writer (Optional[SummaryWriter]): TensorBoard writer.
        scaler (Optional[GradScaler]): For mixed precision training.
        test_loader (DataLoader): DataLoader for the test set.

    Returns:
        Tuple containing:
          - model: The trained model.
          - optimizer: The optimizer used.
          - scheduler: The learning rate scheduler.
          - metrics_for_fold: Dictionary with training and evaluation metrics.
    """
    # Create StratifiedKFold splits and select current fold indices
    skf = StratifiedKFold(n_splits=config.training.folds, shuffle=True, random_state=config.training.seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = list(skf.split(train_val_indices, train_val_labels))
    try:
        current_train_idx, current_val_idx = splits[fold - 1]
    except IndexError as ie:
        logging.error(f"[Fold {fold}] IndexError when accessing fold splits: {ie}")
        raise

    current_train_indices = train_val_indices[current_train_idx]
    current_val_indices = train_val_indices[current_val_idx]

    # Create DataLoaders for training and validation.
    train_loader = create_dataloader(config.paths.hdf5_path, current_train_indices, transform_for_train,
                                     processor, for_vision, config)
    val_loader = create_dataloader(config.paths.hdf5_path, current_val_indices, transform_for_val,
                                   processor, for_vision, config)

    # Initialize the model.
    model = get_bayesian_model(
        config.model.name,
        num_classes=config.model.num_classes,
        input_size=config.model.input_size,
        use_checkpointing=config.model.use_checkpointing,
        dropout_p=config.model.dropout_p
    ).to(device)
    criterion = initialize_loss_function(config, device)
    optimizer = initialize_optimizer(config, model, device)
    scheduler = initialize_scheduler(config, optimizer, train_loader)
    logging.info(f"[Fold {fold}] Initialized model, optimizer, scheduler, and loss function.")

    # Resume from checkpoint if available and applicable.
    if checkpoint_data and (fold == checkpoint_data.get('fold', 0) + 1):
        try:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            if scaler and checkpoint_data.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            logging.info(f"[Fold {fold}] Loaded checkpoint states successfully.")
        except Exception as e:
            logging.error(f"[Fold {fold}] Failed to load checkpoint states: {e}")
            raise

    # Train for this fold.
    fold_start_time = time.time()
    try:
        metrics_for_fold = run_training_fold(
            config,
            fold,
            model,
            device,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            writer,
            join(checkpoints_dir, config.paths.output_path_model)
        )
    except Exception as e:
        logging.error(f"[Fold {fold}] Error during training: {e}")
        raise

    fold_duration = time.time() - fold_start_time
    logging.info(f"[Fold {fold}] Training completed in {fold_duration / 60:.2f} minutes")

    # Evaluate with Monte Carlo predictions on the test set.
    try:
        testing_results = monte_carlo_prediction(
            model,
            test_loader,
            device,
            n_iter=config.training.mc_iterations,
            writer=writer,
            fold=fold
        )
    except Exception as e:
        logging.error(f"[Fold {fold}] Error during Monte Carlo prediction: {e}")
        raise

    metrics_for_fold['all_testing_dict'] = testing_results

    return model, optimizer, scheduler, metrics_for_fold
