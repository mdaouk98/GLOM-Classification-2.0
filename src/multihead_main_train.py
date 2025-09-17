# src/speed_main_train.py


import logging
import torch
import gc
import traceback
from os import makedirs
from os.path import join, isfile
from torch import cuda
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from main_utils.config import load_config
from main_utils.helpers_utils import construct_directory_structure, save_config_snapshot, set_seed

from main_utils.train_utils import setup_logging, setup_device, setup_transforms_and_processor, load_hdf5_data, DataLoadingError, split_data, train_and_evaluate_fold, save_metrics_and_checkpoint
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset

def main(config_path, resume_checkpoint, full_dataset) -> None:
    """
    Main entry point for training the Bayesian Model with cross-validation.
    
    This function performs the following steps:
      1. Parses command-line arguments to obtain the configuration file and (optionally) a checkpoint to resume from.
      2. Loads the training configuration and constructs the required directory structure.
      3. Sets up logging, TensorBoard, and the computing device (CPU/GPU) based on the configuration.
      4. Sets the random seed for reproducibility.
      5. Initializes data augmentation pipelines and (if needed) a vision processor.
      6. Loads dataset indices and labels from an HDF5 file, and splits the data into training/validation and test sets.
      7. Creates DataLoaders for training/validation and testing.
      8. Optionally resumes from a checkpoint to continue training.
      9. If configured, starts a profiler to collect performance metrics.
      10. Performs cross-validation training:
             - For each fold, trains the model, evaluates it using Monte Carlo predictions, and saves checkpoints and metrics.
      11. Cleans up GPU memory and stops the profiler (if active).
      12. Closes TensorBoard and writes the profiler summary (if active).

    This modular design ensures that each training fold is processed independently,
    and the training progress can be resumed from a checkpoint if required.
    """

    # Load configuration and set up directories.
    config = load_config(config_path)
    dirs = construct_directory_structure(config)
    checkpoints_dir = dirs['checkpoints_dir']
    metrics_dir = dirs['metrics_dir']
    logs_dir = dirs['logs_dir']
    runs_dir = dirs['runs_dir']
    dir_name = dirs['dir_name']

    makedirs(checkpoints_dir, exist_ok=True)
    setup_logging(logs_dir)
    logging.info("[Config] Loaded configurations:")
    logging.info(f"Model config: {config.model}")
    logging.info(f"Optimizer config: {config.optimizer}")
    logging.info(f"Scheduler config: {config.scheduler}")
    logging.info(f"HDF5Dataset config: {config.HDF5Dataset}")
    logging.info(f"Augmentation config: {config.augmentation}")
    logging.info(f"Training config: {config.training}")
    logging.info(f"Profiler config: {config.profiler}")
    logging.info(f"TensorBoard config: {config.tensorboard}")
    logging.info(f"Paths config: {config.paths}")
    logging.info(f"Misc config: {config.misc}")
    logging.info(f"Data config: {config.data}")
    logging.info(f"Loss function config: {config.loss_function}")
    logging.info(f"Checkpoints directory set to: {checkpoints_dir}")
    if config.model.multihead:
        logging.info(f"Multihead config: {config.multihead}")

    save_config_snapshot(config, metrics_dir)
    logging.info("[Main] Starting training script...")

    # Initialize TensorBoard writer if activated.
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=runs_dir) if config.tensorboard.activation else None

    # Setup device and seed.
    device = setup_device(config)
    set_seed(config.training.seed)

    # Setup augmentation transforms and processor.
    transform_for_train, transform_for_val, processor, for_vision = setup_transforms_and_processor(config)

    # Load HDF5 data and perform train/validation/test split.
    try:
        all_indices, all_labels, total_samples = load_hdf5_data(config)
    except DataLoadingError as dle:
        logging.error(f"[Main] Data loading failed: {dle}")
        return

    train_val_indices, train_val_labels, test_indices, test_labels = split_data(all_indices, all_labels, config)

    # Create test DataLoader.
    test_loader = create_dataloader_from_dataset(full_dataset, test_indices, config, 'test', transform_for_val, processor, for_vision)
    #######test_loader = initialize_test_loader(config, test_indices, transform_for_val, processor, for_vision)

    # Set up mixed precision scaler if enabled.
    scaler: Optional[GradScaler] = GradScaler('cuda') if cuda.is_available() and config.training.use_mixed_precision else None

    # Load checkpoint data if provided.
    start_fold: int = 1
    checkpoint_data: Optional[Dict[str, Any]] = None
    if resume_checkpoint is not None:
        if not isfile(resume_checkpoint):
            logging.error(f"[Checkpoint] File not found: {resume_checkpoint}")
            raise FileNotFoundError(f"Checkpoint file not found: {resume_checkpoint}")
        try:
            checkpoint_data = torch.load(resume_checkpoint, map_location=device)
            start_fold = checkpoint_data['fold'] + 1
            logging.info(f"[Checkpoint] Resumed training from fold {checkpoint_data['fold']} and epoch {checkpoint_data['epoch']}")
        except Exception as e:
            logging.error(f"[Checkpoint] Failed to load checkpoint: {e}")
            raise

    # Setup profiler if activated.
    profiler: Optional[torch.profiler.profile] = None
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
        logging.info("[Profiler] Profiler started.")

    # Begin cross-validation training.
    try:
        for fold in range(start_fold, config.training.folds + 1):
            logging.info(f"--- [Fold {fold}] Starting training for fold {fold}/{config.training.folds} ---")
            # Train and evaluate this fold.
            model, optimizer, scheduler, metrics_for_fold = train_and_evaluate_fold(
                config,
                fold,
                train_val_indices,
                train_val_labels,
                device,
                transform_for_train,
                transform_for_val,
                processor,
                for_vision,
                checkpoint_data,
                checkpoints_dir,
                writer,
                scaler,
                test_loader,
                full_dataset
            )

            # Save metrics and checkpoint for the fold.
            save_metrics_and_checkpoint(config, fold, model, optimizer, scheduler, scaler,
                                        metrics_for_fold, checkpoints_dir, metrics_dir)

            # Clear GPU cache and cleanup.
            torch.cuda.empty_cache()
            logging.info(f"[Fold {fold}] Cleared GPU cache after fold {fold}")
            if profiler is not None:
                profiler.step()
            del model, optimizer, scheduler, metrics_for_fold
            gc.collect()
            logging.info(f"[Fold {fold}] Cleaned up memory after fold {fold}")

    except Exception as e:
        logging.error(f"[Main] Training script terminated with an error: {e}")
        logging.error(traceback.format_exc())
    finally:
        if profiler is not None:
            profiler.stop()
            logging.info("[Profiler] Profiler stopped.")

    if writer is not None:
        writer.close()
        logging.info("[Main] TensorBoard writer closed.")

    # Write profiler summary if profiler is active.
    if profiler is not None:
        try:
            profiler_summary = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
            summary_path = join(runs_dir, 'profiler_summary.txt')
            with open(summary_path, "w") as f:
                f.write(profiler_summary)
            logging.info(f"[Profiler] Profiler summary written to {summary_path}")
        except Exception as e:
            logging.error(f"[Profiler] Failed to write profiler summary: {e}")



