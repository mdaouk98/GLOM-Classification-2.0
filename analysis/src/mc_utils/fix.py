# src/mc_utils/mc_clustering/mc_clustering_main.py

"""
Extract intermediate feature vectors from a trained model for clustering/visualization.
This script uses the same configuration and data splitting logic as main_train.py.
It computes statistics on a main dataset (combining original train and val)
and also computes MC-dropout statistics on a subset (10%) of the test dataset,
called new_images.

Note: The statistics (and MC statistics) are computed over the entire aggregated dataset
across all folds.
"""



import argparse
import logging
import os
from os.path import join
import numpy as np
import random
import torch
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config
from mc_utils.mc_clustering import build_transform_processor, ensure_dir_exists, create_metrics_dataframe, process_main_dataset, process_new_images_dataset, process_test_dataset, compute_p_value, retrain_main


def fix() -> None:
    """
    Main function to extract intermediate feature vectors from a trained model,
    compute feature statistics, and evaluate distance metrics for both the main (train+val)
    and new_images (subset of test) datasets. The results are saved to disk.
    """
    parser = argparse.ArgumentParser(
        description="Feature extraction and statistics computation for main and new_images splits."
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration YAML file.')
    args = parser.parse_args()
    
    # Setup logging configuration.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration and select computation device.
    config = load_config(args.config)
    device = torch.device(f"cuda:{config.misc.cuda}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Build transforms and processor based on model type.
    transform, processor, for_vision = build_transform_processor(config)
    
    # Read indices and labels from an HDF5 file.
    try:
        with h5py.File(config.paths.hdf5_path, 'r') as f:
            all_labels = f['labels'][:]
            total_samples = len(all_labels)
            all_indices = np.arange(total_samples)
    except Exception as e:
        logging.error(f"Error reading HDF5 file: {e}")
        return
    
    # Split the data into main (train+val) and test sets.
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices,
        all_labels,
        test_size=config.training.test_size,
        random_state=config.training.seed,
        stratify=all_labels
    )
    logging.info(f"Main dataset (train+val) samples: {len(train_val_indices)}")
    logging.info(f"Test dataset samples: {len(test_indices)}")
    
    # Ensure the output directory exists.
    output_dir = join("metrics", config.paths.run_name)
    ensure_dir_exists(output_dir)
    
    
    # Retrain model (Step 3)
    random.seed(config.training.seed)  # Set a reproducible seed
    random_numbers = random.sample(range(145), 5)
    logging.info(f"Our random numbers are {random_numbers}")
    for i in random_numbers:
      # Split out new_images from test indices (10% for new images).
      new_images_indices, remaining_test_indices, new_images_labels, remaining_test_labels = train_test_split(
          test_indices,
          test_labels,
          test_size=0.9,
          random_state=config.training.seed,
          stratify=test_labels
      )
      new_train_val_indices = train_val_indices.copy()
      new_train_val_labels = train_val_labels.copy()
      new_train_val_indices = np.append(new_train_val_indices,new_images_indices[i])
      new_train_val_labels = np.append(new_train_val_labels,new_images_labels[i])
      retrain_main(config, new_train_val_indices, new_train_val_labels)
      
      logging.info(f"new_train_val_indices is of shape{new_train_val_indices.shape} compared to train_val_indices of shape {train_val_indices}")
      
      # Process retrained test dataset (remaining subset of test) for MC-dropout statistics.
      retrained_test_results = process_test_dataset(test_indices, test_labels, transform, processor, for_vision, device, config,retrained = True)
      if retrained_test_results is None:
          logging.error("Retrained Test dataset processing failed for image {i}.")
          return
          
      # Save retrained test statistics.
      output_path_retrained_test = join(output_dir, f"retrained_test_uncertainty_stats_image{i}.npz")
      np.savez(output_path_retrained_test, **retrained_test_results)
      logging.info(f"Retrained test uncertainty for image {i} saved to {output_path_retrained_test}.")
      
      
    
