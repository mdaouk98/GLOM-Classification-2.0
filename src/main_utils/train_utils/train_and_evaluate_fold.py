# src/main_utils/train_utils/train_and_evaluate_fold.py

import time
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from os.path import join
from sklearn.model_selection import StratifiedKFold
from typing import Any, Optional, Tuple, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from main_utils.models_utils import get_bayesian_model
from main_utils.hyperparameters_utils import (
    initialize_optimizer,
    initialize_scheduler,
    initialize_loss_function
)
from main_utils.train_utils.trainer_utils import run_training_fold
from main_utils.evaluation_utils import monte_carlo_prediction
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset

def train_and_evaluate_fold(
    config: Any,
    fold: int,
    train_val_indices: np.ndarray,
    train_val_labels: np.ndarray,
    device: torch.device,
    transform_for_train: Optional[Any],
    transform_for_val: Optional[Any],
    processor: Optional[Any],
    for_vision: bool,
    checkpoint_data: Optional[Dict[str, Any]],
    checkpoints_dir: str,
    writer: Optional[SummaryWriter],
    scaler: Optional[GradScaler],
    test_loader: DataLoader,
    full_dataset: HDF5Dataset
) -> Tuple[nn.Module, Any, Any, Dict[str, Any]]:
    """
    Train and evaluate a single cross-validation fold.

    Returns:
        - model: Trained model.
        - optimizer: Optimizer instance (with updated state).
        - scheduler: LR scheduler instance.
        - metrics_for_fold: Dictionary of train/val/test metrics.
    """
    # --------------------------------------------------------------------------
    # 1) Split train/validation via StratifiedKFold
    # --------------------------------------------------------------------------
    # Choose the array for stratification: primary head if multi-head, else labels
    if getattr(config.model, 'multihead', False):
        primary_head = config.multihead.labels[0]
        stratify_y = train_val_labels[primary_head]
    else:
        stratify_y = train_val_labels

    skf = StratifiedKFold(
        n_splits=config.training.folds,
        shuffle=True,
        random_state=config.training.seed
    )
    splits = list(skf.split(train_val_indices, stratify_y))

    try:
        train_idx, val_idx = splits[fold - 1]
    except IndexError as e:
        logging.error(f"[Fold {fold}] Invalid fold index: {e}")
        raise

    current_train_indices = train_val_indices[train_idx]
    current_val_indices   = train_val_indices[val_idx]

    # --------------------------------------------------------------------------
    # 2) Build DataLoaders for this fold
    # --------------------------------------------------------------------------
    train_loader = create_dataloader_from_dataset(
        dataset=full_dataset,
        indices=current_train_indices,
        config=config,
        loader_type='train',
        dataset_transform=transform_for_train,
        processor=processor,
        for_vision=for_vision
    )
    val_loader = create_dataloader_from_dataset(
        dataset=full_dataset,
        indices=current_val_indices,
        config=config,
        loader_type='validate',
        dataset_transform=transform_for_val,
        processor=processor,
        for_vision=for_vision
    )

    # --------------------------------------------------------------------------
    # 3) Determine model heads and initialize model
    # --------------------------------------------------------------------------
    if getattr(config.model, 'multihead', False):
        # multi-head: map each head name to its number of classes
        heads = dict(zip(config.multihead.labels, config.multihead.num_classes))
    else:
        # single-head: one head mapping
        head_name = config.model.label
        heads = {head_name: config.model.num_classes}

    # Instantiate Bayesian model and move to device
    model = get_bayesian_model(
        model_name=config.model.name,
        heads=heads,
        input_size=config.model.input_size,
        use_checkpointing=config.model.use_checkpointing,
        dropout_p=config.model.dropout_p,
        device=device
    ).to(device, memory_format=torch.channels_last)

    # --------------------------------------------------------------------------
    # 4) Initialize per-head loss functions and weights
    # --------------------------------------------------------------------------
    criterions: Dict[str, nn.Module] = {}
    loss_weights: Dict[str, float]  = {}

    if getattr(config.model, 'multihead', False):
        # Ensure consistency across lists
        labels, loss_funcs, weights = (
            config.multihead.labels,
            config.multihead.loss_functions,
            config.multihead.loss_weights
        )
        if not (len(labels) == len(loss_funcs) == len(weights)):
            raise ValueError("multihead.labels, loss_functions, and loss_weights lengths must match.")

        for head, lf_name, w in zip(labels, loss_funcs, weights):
            # copy config to set per-head loss_function field
            tmp_cfg = copy.copy(config)
            tmp_cfg.loss_function = lf_name
            criterions[head]   = initialize_loss_function(tmp_cfg, device)
            loss_weights[head] = w
    else:
        # single-head
        head_name = next(iter(heads))
        criterions[head_name]   = initialize_loss_function(config, device)
        loss_weights[head_name] = 1.0

    # --------------------------------------------------------------------------
    # 5) Initialize optimizer and scheduler
    # --------------------------------------------------------------------------
    optimizer = initialize_optimizer(config, model, device)
    scheduler = initialize_scheduler(config, optimizer, train_loader)
    logging.info(f"[Fold {fold}] Model, optimizer, scheduler, and losses initialized.")

    # --------------------------------------------------------------------------
    # 6) Optionally resume from checkpoint
    # --------------------------------------------------------------------------
    if checkpoint_data and fold == checkpoint_data.get('fold', 0) + 1:
        try:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            if scaler and checkpoint_data.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            logging.info(f"[Fold {fold}] Resumed from checkpoint.")
        except Exception as e:
            logging.error(f"[Fold {fold}] Checkpoint resume failed: {e}")
            raise

    # --------------------------------------------------------------------------
    # 7) Train for this fold
    # --------------------------------------------------------------------------
    start_time = time.time()
    try:
        metrics_for_fold = run_training_fold(
            config=config,
            fold=fold,
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterions=criterions,
            loss_weights=loss_weights,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            writer=writer,
            output_path_model=join(checkpoints_dir, config.paths.output_path_model)
        )
    except Exception as e:
        logging.error(f"[Fold {fold}] Training failed: {e}")
        raise
    
    # If no epochs completed (e.g. first epoch crashed), abort before evaluating
    if metrics_for_fold.get('num_epochs_trained', 0) == 0:
        logging.error(f"[Fold {fold}] No training epochs completed; aborting evaluation.")
        raise RuntimeError(f"[Fold {fold}] Training did not complete any epochs.")

    elapsed_min = (time.time() - start_time) / 60
    logging.info(f"[Fold {fold}] Training completed in {elapsed_min:.2f} minutes.")

    # --------------------------------------------------------------------------
    # 8) Evaluate on test set via Monte Carlo sampling
    # --------------------------------------------------------------------------
    try:
        test_results = monte_carlo_prediction(
            model=model,
            dataloader=test_loader,
            device=device,
            n_iter=config.training.mc_iterations,
            writer=writer,
            fold=fold,
            heads=heads
        )
        metrics_for_fold['all_testing_dict'] = test_results
    except Exception as e:
        logging.error(f"[Fold {fold}] Monte Carlo evaluation failed: {e}")
        raise

    return model, optimizer, scheduler, metrics_for_fold

