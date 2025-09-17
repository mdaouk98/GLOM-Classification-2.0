# src/iterate_train.py

import argparse
import os
import sys
from time import time
from logging import error, info
from traceback import format_exc

from main_utils.config import load_config
from main_utils.helpers_utils import set_seed
from main_utils.train_utils import load_hdf5_data, setup_transforms_and_processor, setup_device
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset

parser = argparse.ArgumentParser(description='Train a Bayesian Model.')
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to resume training from a checkpoint.')
args = parser.parse_args()


# Load configuration and set up directories.
config_path = f"configs/multihead_scanner/scale_n4.yaml"
config = load_config(config_path)

# Setup device and seed.
device = setup_device(config)
set_seed(config.training.seed)

# Setup augmentation transforms and processor.
transform_for_train, transform_for_val, processor, for_vision = setup_transforms_and_processor(config)

# Load HDF5 data and perform train/validation/test split.
try:
    all_indices, all_labels, all_wsis, _, _, total_samples = load_hdf5_data(config)
except DataLoadingError as dle:
    error(f"[Main] Data loading failed: {dle}")
    sys.exit(1)

# Load the dataset once.
full_dataset = HDF5Dataset(
    hdf5_file=config.paths.hdf5_path,
    indices=all_indices,  # Load all indices initially.
    transform=None,  # You can set transform later using wrapper if needed.
    processor=processor, 
    for_vision=for_vision,
    cache_in_memory=config.HDF5Dataset.cache_in_memory,
    label_names=(config.multihead.labels if config.model.multihead else [config.model.label])
)
info(f"[full dataset] Loaded HDF5Dataset with all samples.")
for scale in ['n4','n3','n2','n1',0,'p1','p2','p3','p4']:
    path = f"scale_{scale}"
    config_path = f"configs/multihead_scanner/{path}.yaml"
    print(f" Processing {config_path}")
    metric_num = 0
    for metric_index in range(5):
      metric_path = f"metrics/multihead_scanner/{path}/training_dict_fold{metric_index+1}.json"
      if os.path.exists(metric_path):
        metric_num += 1
    if metric_num == 5:
            print(f""" 
            
            
            {config_path} is already trained.
            Passing into next configuration
            
            
            """)
            pass
    else:
      
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