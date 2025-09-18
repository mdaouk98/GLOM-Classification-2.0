# src/mc_utils/mc_clustering/retrain_main.py

import argparse
import logging
from os import makedirs
from os.path import join, isfile, dirname
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import cuda
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from h5py import File
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
import json
import traceback
import gc

from main_utils.config import load_config
from main_utils.hyperparameters_utils import initialize_optimizer, initialize_scheduler, initialize_loss_function
from main_utils.helpers_utils import (construct_directory_structure, save_config_snapshot, set_seed, convert_to_serializable)
from main_utils.augmentations_utils import get_augmentation_pipeline
from main_utils.datasets_utils import HDF5Dataset
from mc_utils.mc_clustering import load_model
from main_utils.train_utils.trainer_utils import run_training_fold


def retrain_main(config, train_val_indices, train_val_labels):

    logging.info(f"Loaded CUDA device from config: {config.misc.cuda}")

    dirs = construct_directory_structure(config)
    checkpoints_dir = dirs['checkpoints_dir']
    metrics_dir = dirs['metrics_dir']
    logs_dir = dirs['logs_dir']
    runs_dir = dirs['runs_dir']
    dir_name = dirs['dir_name']

    makedirs(checkpoints_dir, exist_ok=True)

    # --- Revised Logging Setup: Clear Existing Handlers and Configure Logging ---
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    log_file = join(logs_dir, 'retraining.log')
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Loaded model config: {config.model}")
    logging.info(f"Loaded optimizer config: {config.optimizer}")
    logging.info(f"Loaded scheduler config: {config.scheduler}")
    logging.info(f"Loaded HDF5Dataset config: {config.HDF5Dataset}")
    logging.info(f"Loaded augmentation config: {config.augmentation}")
    logging.info(f"Loaded training config: {config.training}")
    logging.info(f"Loaded profiler config: {config.profiler}")
    logging.info(f"Loaded tensorboard config: {config.tensorboard}")
    logging.info(f"Loaded paths config: {config.paths}")
    logging.info(f"Loaded misc config: {config.misc}")
    logging.info(f"Loaded data config: {config.data}")
    logging.info(f"Loaded loss function config: {config.loss_function}")
    logging.info(f"Checkpoints directory set to: {checkpoints_dir}")
    save_config_snapshot(config, metrics_dir)
    logging.info("Starting retraining script...")

    # Initialize TensorBoard SummaryWriter if activated in config.
    writer = SummaryWriter(log_dir=runs_dir) if config.tensorboard.activation else None

    # Device and CUDA setup
    device = torch.device(f'cuda:{config.misc.cuda}' if cuda.is_available() else 'cpu')
    print("Number of GPUs available:", torch.cuda.device_count())
    if cuda.is_available():
        logging.info(f'CUDA device {config.misc.cuda} is available')
        cuda.set_device(device)
        cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    logging.info(f'Using device: {device}')
    set_seed(config.training.seed)

    # Data transformations (unchanged)
    augmentation_type = config.augmentation.type.lower()
    if augmentation_type not in ['mixup', 'cutmix']:
        transform_for_train = get_augmentation_pipeline(augmentation_type, config.model.input_size)
    else:
        transform_for_train = get_augmentation_pipeline('basic', config.model.input_size)
    transform_for_val = get_augmentation_pipeline('none', config.model.input_size)

    processor = None
    for_vision = False
    if config.model.name.lower() == 'vision':
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        for_vision = True
        # For vision models, rely on processor instead of augmentation transforms
        transform_for_train = None
        transform_for_val = None



    # Set up mixed precision if enabled.
    scaler = GradScaler('cuda') if cuda.is_available() and config.training.use_mixed_precision else None

    skf = StratifiedKFold(
        n_splits=config.training.folds,
        shuffle=True,
        random_state=config.training.seed
    )

    start_fold = 1
    checkpoint_data = None


    # Set up the PyTorch profiler if activated in config.
    profiler = None
    if config.profiler.activation:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=config.profiler.wait_steps,
                warmup=config.profiler.warmup_steps,
                active=config.profiler.active_steps,
                repeat=config.profiler.repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(join(runs_dir, 'profiler')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()

    try:
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_val_indices, train_val_labels),
            start=start_fold
        ):
            if fold > config.training.folds:
                logging.info("All folds processed.")
                break

            logging.info(f'--- Fold {fold}/{config.training.folds} ---')
            current_train_indices = train_val_indices[train_idx]
            current_val_indices = train_val_indices[val_idx]

            # Create training and validation datasets with optional caching.
            train_subset = HDF5Dataset(
                hdf5_file=config.paths.hdf5_path,
                indices=current_train_indices,
                transform=transform_for_train,
                processor=processor,
                for_vision=for_vision,
                cache_in_memory=config.HDF5Dataset.cache_in_memory
            )
            val_subset = HDF5Dataset(
                hdf5_file=config.paths.hdf5_path,
                indices=current_val_indices,
                transform=transform_for_val,
                processor=processor,
                for_vision=for_vision,
                cache_in_memory=config.HDF5Dataset.cache_in_memory
            )

            # Create DataLoaders with an increased prefetch_factor.
            train_loader = DataLoader(
                train_subset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
                prefetch_factor=config.data.prefetch_factor
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
                prefetch_factor=config.data.prefetch_factor
            )

            # Initialize the model.
            model = load_model(fold, device, config, retrained = False)

            # Initialize loss, optimizer, and scheduler.
            criterion = initialize_loss_function(config, device)
            optimizer = initialize_optimizer(config, model, device)
            scheduler = initialize_scheduler(config, optimizer, train_loader)

            

            # Train for this fold.
            fold_start_time = time.time()
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
            
            # Save metrics to JSON.
            output_dict_per_fold = {
                'fold_loss': metrics_for_fold['fold_loss'],
                'train_accuracies': metrics_for_fold['train_accuracies'],
                'val_losses': metrics_for_fold['val_losses'],
                'val_accuracies': metrics_for_fold['val_accuracies'],
                'precisions': metrics_for_fold['precisions'],
                'recalls': metrics_for_fold['recalls'],
                'f1_scores': metrics_for_fold['f1_scores'],
                'aucs': metrics_for_fold['aucs'],
                'confusion_matrices': metrics_for_fold['confusion_matrices'],
                'gradient_norms': metrics_for_fold['gradient_norms'],
                'epoch_times': metrics_for_fold['epoch_times'],
                'num_epochs_trained': metrics_for_fold['num_epochs_trained'],
            }
            
            fold_end_time = time.time()
            fold_duration = fold_end_time - fold_start_time
            logging.info(f"Fold {fold} training completed in {fold_duration / 60:.2f} minutes")


            # Save the best and final models.
            model_save_path = join(checkpoints_dir, f"{config.paths.output_path_model}_fold{fold}_retrained.pth")
            makedirs(dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved at {model_save_path}")

            final_checkpoint_path = join(checkpoints_dir, f"{config.paths.output_path_model}_fold{fold}_final_retrained.pth")
            torch.save({
                'fold': fold,
                'epoch': metrics_for_fold['num_epochs_trained'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'metrics': output_dict_per_fold
            }, final_checkpoint_path)
            logging.info(f"Final checkpoint saved at {final_checkpoint_path}")
            
            # Empty the GPU cache after this fold
            torch.cuda.empty_cache()
            logging.info(f"Cleared GPU cache after fold {fold}")

            # Step the profiler if it is active.
            if profiler is not None:
                profiler.step()
            
            # Explicitly free fold-specific objects to avoid memory buildup
            del model, optimizer, scheduler, train_loader, val_loader, train_subset, val_subset, metrics_for_fold, output_dict_per_fold
            gc.collect()
            logging.info(f"Cleaned up memory after fold {fold}")

    except Exception as e:
        logging.error(f"Training script terminated with an error: {e}")
        logging.error(traceback.format_exc())
    finally:
        if profiler is not None:
            profiler.stop()

    if writer is not None:
        writer.close()

    # --- Write Profiler Summary to Text File if profiler is active ---
    if profiler is not None:
        try:
            profiler_summary = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
            summary_path = join(runs_dir, 'profiler_summary.txt')
            with open(summary_path, "w") as f:
                f.write(profiler_summary)
            logging.info(f"Profiler summary written to {summary_path}")
        except Exception as e:
            logging.error(f"Failed to write profiler summary: {e}")


