# src/iterate_train_config.py

import argparse
import os
import torch
import gc
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
     '--config', '-m',
     nargs='+',                      # <-- one or more values
     default=['configs/default_config.yaml'],           # <-- now default is a list
     help='One or more config'
 )
    parser.add_argument(
        '--resume_checkpoint', '-r',
        type=str,
        default=None,
        help='Path to a checkpoint file to resume training.'
    )
    return parser.parse_args()
    
args = parse_args()

print(args)

# Load configuration and set up directories.
config_path = f"{args.config[0]}"
config = load_config(config_path)

# Setup device and seed.
device = setup_device(config)
set_seed(config.training.seed)

# Setup augmentation transforms and processor.
transform_for_train, transform_for_val, processor, for_vision = setup_transforms_and_processor(config)

# Load HDF5 data and perform train/validation/test split.
try:
    all_idxs, all_lbls, all_wsis, _, _, total = load_hdf5_data(config)
    info(f"[Startup] Loaded {total} samples from HDF5.")
except DataLoadingError as e:
    error(f"[Error] Data loading failed: {e}")
    sys.exit(1)

# Load the dataset once.
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
info(f"[full dataset] Loaded HDF5Dataset with all samples.")
    
for c in args.config:
    config_path = c
    print(f" Processing {config_path}")
      
    from main_utils.main_train import main
    
    if __name__ == "__main__":
        start_time = time()
        try:
            main(config_path, args.resume_checkpoint, full_dataset)
        except Exception as e:
            error(f"Training script terminated with an error: {e}")
            error(format_exc())  # Log full stack trace
        end_time = time()
        elapsed_time = end_time - start_time
        info(f"Script completed in {elapsed_time / 60:.2f} minutes")
        
    # --- Dismantle every reference to the old model / optimizer / buffers ---
    # so that torch's caching allocator can really free all of that VRAM:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # and force Python to drop any lingering object graphs:
    gc.collect()