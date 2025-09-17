# src/main_utils/main_train.py


import logging
import time
import gc
import traceback
from os import makedirs
from os.path import join, isfile
import torch
from torch import cuda
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from main_utils.config import load_config
from main_utils.helpers_utils import (
    construct_directory_structure,
    save_config_snapshot,
    set_seed
)
from main_utils.train_utils import (
    setup_logging,
    setup_device,
    setup_transforms_and_processor,
    load_hdf5_data,
    DataLoadingError,
    split_data,
    train_and_evaluate_fold,
    save_metrics_and_checkpoint
)
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset


def main(config_path: str, resume_checkpoint: str, full_dataset: HDF5Dataset) -> None:
    """
    Entry point for cross-validated Bayesian model training.

    Steps:
      1) Load YAML config ? Pydantic model.
      2) Create output directories (checkpoints, metrics, logs, runs).
      3) Initialize logging, TensorBoard, and compute device.
      4) Set random seed for reproducibility.
      5) Prepare data transforms or vision processor.
      6) Load dataset indices/labels and split into train/val/test.
      7) Build test DataLoader.
      8) Optionally resume from a checkpoint.
      9) Optionally start profiler.
     10) Loop over CV folds:
           a) Train & validate model.
           b) Evaluate on test set (MC dropout).
           c) Save fold metrics and checkpoint.
           d) Cleanup GPU memory.
     11) Stop profiler and write summary.
     12) Close TensorBoard writer.
    """
    # --- 1) Load config & prepare directory structure ---
    config = load_config(config_path)
    dirs = construct_directory_structure(config)
    checkpoints_dir = dirs['checkpoints_dir']
    metrics_dir     = dirs['metrics_dir']
    logs_dir        = dirs['logs_dir']
    runs_dir        = dirs['runs_dir']

    # Ensure checkpoint directory exists
    makedirs(checkpoints_dir, exist_ok=True)

    # --- 2) Setup logging and snapshot config ---
    setup_logging(logs_dir)
    logging.info(f"[Config] Loaded config: {config}")
    save_config_snapshot(config, metrics_dir)

    # --- 3) Setup TensorBoard writer (optional) ---
    writer = SummaryWriter(log_dir=runs_dir) if config.tensorboard.activation else None

    # --- 4) Setup compute device & random seed ---
    device = setup_device(config)
    set_seed(config.training.seed)

    # --- 5) Prepare transforms / processor for data ---
    transform_train, transform_val, processor, for_vision = setup_transforms_and_processor(config)

    # --- 6) Load HDF5 indices & labels, then split data ---
    try:
        all_indices, all_labels, all_wsis, _, _, total_samples = load_hdf5_data(config)
    except DataLoadingError as e:
        logging.error(f"[Data Loading] {e}")
        return

    train_val_idx, train_val_lbls, test_idx, test_lbls = split_data(
        all_indices, all_labels, all_wsis, config
    )

    # --- 7) Build test DataLoader ---
    test_loader = create_dataloader_from_dataset(
        full_dataset, test_idx, config, 'test',
        transform_val, processor, for_vision
    )

    # --- 8) Mixed precision scaler (optional) ---
    scaler = GradScaler() if cuda.is_available() and config.training.use_mixed_precision else None

    # --- 9) Resume checkpoint (if given) ---
    start_fold = 1
    checkpoint_data = None
    if resume_checkpoint:
        if not isfile(resume_checkpoint):
            logging.error(f"[Checkpoint] Not found: {resume_checkpoint}")
            raise FileNotFoundError(resume_checkpoint)
        checkpoint_data = torch.load(resume_checkpoint, map_location=device)
        start_fold = checkpoint_data.get('fold', 0) + 1
        logging.info(f"[Checkpoint] Resuming from fold {checkpoint_data.get('fold')}")

    # --- 10) (Optional) Start profiler ---
    profiler = None
    if config.profiler.activation:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
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
        logging.info("[Profiler] Started")

    # --- 11) Cross-validation loop ---
    try:
        for fold in range(start_fold, config.training.folds + 1):
            logging.info(f"--- Fold {fold}/{config.training.folds} ---")

            # Train & validate this fold
            model, optimizer, scheduler, metrics = train_and_evaluate_fold(
                config, fold,
                train_val_idx, train_val_lbls,
                device,
                transform_train, transform_val,
                processor, for_vision,
                checkpoint_data,
                checkpoints_dir,
                writer,
                scaler,
                test_loader,
                full_dataset
            )

            # Save fold metrics + model checkpoint
            save_metrics_and_checkpoint(
                config, fold,
                model, optimizer, scheduler, scaler,
                metrics,
                checkpoints_dir, metrics_dir
            )

            # Cleanup GPU memory between folds
            torch.cuda.empty_cache()
            logging.info(f"[Fold {fold}] Cleared GPU cache")
            if profiler:
                profiler.step()
            del model, optimizer, scheduler, metrics
            gc.collect()
            logging.info(f"[Fold {fold}] Memory cleaned up")

    except Exception as e:
        logging.error(f"[Main] Error in training loop: {e}")
        logging.error(traceback.format_exc())

    finally:
        # --- 12) Stop profiler & write summary ---
        if profiler:
            profiler.stop()
            logging.info("[Profiler] Stopped")
            try:
                summary = profiler.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=50
                )
                summary_path = join(runs_dir, 'profiler_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write(summary)
                logging.info(f"[Profiler] Summary saved to {summary_path}")
            except Exception as e:
                logging.error(f"[Profiler] Summary write failed: {e}")

        # Close TensorBoard writer
        if writer:
            writer.close()
            logging.info("[Main] TensorBoard writer closed")




