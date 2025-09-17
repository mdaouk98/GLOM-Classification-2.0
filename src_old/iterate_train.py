# src/iterate_train.py

import argparse
from time import time
from logging import error, info
from traceback import format_exc

from main_train import main

parser = argparse.ArgumentParser(description='Train a Bayesian Model.')
parser.add_argument('--model', type=str, default='Resnet18',
                    help='Iterating based on model type.')
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to resume training from a checkpoint.')
args = parser.parse_args()

for image_input in [224,512,1024]:
  for augmentation_type in ['none', 'basic', 'advanced','mixup','cutmix']:
    for scheduler_type in ['reduce_on_plateau', 'cosine_warm_up']:
      for criterion_weight in ['None', 'equal_weight', 'weight10']:
        for optimizer_type in ['Adam', 'AdamW']:
          for optimizer_lr in [0.01,0.001,0.0001]:
            for loss_function in ['CrossEntropyLoss','ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss']:
              config_path = f"configs/image_size_{image_input}/augmentation_{augmentation_type}/scheduler_type_{scheduler_type}/criterion_weight_{criterion_weight}/optimizer_type_{optimizer_type}/optimizer_lr_{optimizer_lr}/loss_function_{loss_function}/model_{args.model}.yaml"
              print(f" Processing {config_path}")
              if __name__ == "__main__":
                  start_time = time()
                  try:
                      main(config_path, args.resume_checkpoint)
                  except Exception as e:
                      error(f"Training script terminated with an error: {e}")
                      error(format_exc())  # Log full stack trace
                  end_time = time()
                  elapsed_time = end_time - start_time
                  info(f"Script completed in {elapsed_time / 60:.2f} minutes")
