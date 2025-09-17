# src/train.py

import argparse
from time import time
from logging import error, info
from traceback import format_exc

from main_train import main

parser = argparse.ArgumentParser(description='Train a Bayesian Model.')
parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                    help='Path to configuration YAML file.')
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to resume training from a checkpoint.')
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time()
    try:
        main(args.config, args.resume_checkpoint)
    except Exception as e:
        error(f"Training script terminated with an error: {e}")
        error(format_exc())  # Log full stack trace
    end_time = time()
    elapsed_time = end_time - start_time
    info(f"Script completed in {elapsed_time / 60:.2f} minutes")
