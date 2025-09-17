# src/speed_iterate_train.py

import argparse
import os
from time import time
from logging import error, info
from traceback import format_exc

from main_utils.config import load_config
from main_utils.helpers_utils import set_seed
from main_utils.train_utils import load_hdf5_data, setup_transforms_and_processor, setup_device, DataLoadingError
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset
from main_utils.datasets_utils import HDF5Dataset


parser = argparse.ArgumentParser(description='Train a Bayesian Model.')
parser.add_argument('--model', type=str, default='Resnet18',
                    help='Iterating based on model type.')
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to resume training from a checkpoint.')
args = parser.parse_args()

for image_input in [224]:
  index = 0
  for augmentation_type in ['none', 'basic', 'advanced','mixup','cutmix']:
    # Load configuration and set up directories.
    config_path = f"configs/augmentation1/image_size_{image_input}/augmentation_{augmentation_type}/scheduler_type_reduce_on_plateau/criterion_weight_None/optimizer_type_Adam/optimizer_lr_0.0001/loss_function_CrossEntropyLoss/model_{args.model}.yaml"
    config = load_config(config_path)
    
    # Setup device and seed.
    device = setup_device(config)
    set_seed(config.training.seed)
    
    # Setup augmentation transforms and processor.
    transform_for_train, transform_for_val, processor, for_vision = setup_transforms_and_processor(config)
    
    # Load HDF5 data and perform train/validation/test split.
    try:
        all_indices, all_labels, total_samples = load_hdf5_data(config)
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
        cache_in_memory=config.HDF5Dataset.cache_in_memory
    )
    info(f"[full dataset] Loaded HDF5Dataset with all samples.")
    for scheduler_type in ['reduce_on_plateau']:
      for criterion_weight in ['None', 'equal_weight', 'weight10']:
        for optimizer_type in ['Adam', 'AdamW']:
          for optimizer_lr in [0.0001]:
            for loss_function in ['CrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss']:
              index += 1
              print(f"Processing model {index} out of 90")
              path = f"image_size_{image_input}/augmentation_{augmentation_type}/scheduler_type_{scheduler_type}/criterion_weight_{criterion_weight}/optimizer_type_{optimizer_type}/optimizer_lr_{optimizer_lr}/loss_function_{loss_function}/model_{args.model}"
              config_path = f"configs/augmentation1/{path}.yaml"
              print(f" Processing {config_path}")
              metric_num = 0
              for metric_index in range(5):
                metric_path = f"metrics/augmentation1/{path}/training_dict_fold{metric_index+1}.json"
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
