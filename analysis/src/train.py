# src/train.py

import argparse
import os
import sys
from time import time
from logging import error, info

from main_utils.config import load_config
from main_utils.helpers_utils import set_seed
from main_utils.train_utils import (
    load_hdf5_data,
    setup_transforms_and_processor,
    setup_device,
    DataLoadingError
)
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset
from main_utils.main_train import main

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a Bayesian Model.')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to YAML configuration file.'
    )
    parser.add_argument(
        '--resume_checkpoint', '-r',
        type=str,
        default=None,
        help='Path to a checkpoint file to resume training.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --- 1) Load config & set up device/seed ---
    config = load_config(args.config)
    device = setup_device(config)
    set_seed(config.training.seed)
    info(f"[Startup] Using device: {device}")
    info(f"[Startup] Loaded config from: {args.config}")

    # --- 2) Prepare data transforms / vision processor ---
    transform_train, transform_val, processor, for_vision = setup_transforms_and_processor(config)
    info(f"[Startup] Data transforms ready (vision={for_vision}).")

    # --- 3) Load HDF5 indices & labels, split test/train-val later ---
    try:
        all_idxs, all_lbls, all_wsis, _, _, total = load_hdf5_data(config)
        info(f"[Startup] Loaded {total} samples from HDF5.")
    except DataLoadingError as e:
        error(f"[Error] Data loading failed: {e}")
        sys.exit(1)

    # --- 4) Instantiate full dataset (lazy transforms applied per-split) ---
    label_names = config.multihead.labels if config.model.multihead else [config.model.label]
    full_dataset = HDF5Dataset(
        hdf5_file=config.paths.hdf5_path,
        indices=all_idxs,            # initially load all samples
        transform=None,              # wrap later with TransformWrapper
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=config.HDF5Dataset.cache_in_memory,
        label_names=label_names
    )
    info(f"[Startup] HDF5Dataset created with {len(all_idxs)} indices.")

    # --- 5) Run the main training routine ---
    start = time()
    try:
        main(
            config_path=args.config,
            resume_checkpoint=args.resume_checkpoint,
            full_dataset=full_dataset
        )
    except Exception as exc:
        error(f"[Fatal] Training terminated: {exc}")
        error(sys.exc_info()[2])
        sys.exit(1)
    finally:
        elapsed = (time() - start) / 60
        info(f"[Shutdown] Total runtime: {elapsed:.2f} minutes")
