# src/iterate_train_rmv_inf_grad.py

import os
# allow large allocations to be carved into smaller segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from time import time
from logging import error, info
from traceback import format_exc
import gc
import torch

from main_utils.config import load_config
from main_utils.helpers_utils import set_seed
from main_utils.train_utils import load_hdf5_data, setup_transforms_and_processor, setup_device, DataLoadingError
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset


parser = argparse.ArgumentParser(description='Train a Bayesian Model.')
parser.add_argument(
     '--model', '-m',
     nargs='+',                      # <-- one or more values
     default=['Resnet18'],           # <-- now default is a list
     help='One or more model types, e.g. --model Resnet18 VGG16'
 )

parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to resume training from a checkpoint.')
args = parser.parse_args()


# Load configuration and set up directories.
config_path = f"configs/multihead2/scale_0.0/cluster_2/model_{args.model[0]}.yaml"
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

for scale_multiplier in [0.0,-0.01,-0.05,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5]:
    for cluster in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        for model in args.model:
            path = f"scale_{scale_multiplier}/cluster_{cluster}/model_{model}"
            config_path = f"configs/multihead2/{path}.yaml"
            print(f" Processing {config_path}")
            metric_num = 0
            for metric_index in range(5):
              metric_path = f"metrics/multihead2/{path}/training_dict_fold{metric_index+1}.json"
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
                  
              # --- Dismantle every reference to the old model / optimizer / buffers ---
              # so that torch's caching allocator can really free all of that VRAM:
              torch.cuda.empty_cache()
              torch.cuda.reset_peak_memory_stats()
              # and force Python to drop any lingering object graphs:
              gc.collect()
                  
