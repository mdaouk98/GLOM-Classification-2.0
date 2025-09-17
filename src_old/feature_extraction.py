# src/feature_extraction.py

"""
Extract intermediate feature vectors from a trained model for clustering/visualization.
This script uses the same configuration and data splitting logic as main_train.py.
The extracted features (along with predictions and true labels) are saved to a file
whose path is determined by the config YAML file.
"""

import argparse
import logging
import os
from os.path import join
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
import tqdm

from config import load_config
from datasets import HDF5Dataset
from augmentations import get_augmentation_pipeline
from models.get_bayesian_model import get_bayesian_model


def extract_features(model, dataloader, device, mc_dropout=False):
    """
    Run the model in evaluation mode on the provided dataloader (with optional MC dropout)
    and collect the intermediate feature vectors (using return_features=True).
    """
    model.eval()
    features_list = []
    with torch.no_grad():
        for inputs, _ in tqdm.tqdm(dataloader, desc="Extracting features", unit="batch"):
            inputs = inputs.to(device)
            # Forward pass with dropout as specified
            output = model(inputs, mc_dropout=mc_dropout, return_features=True)
            # Expecting (logits, features)
            if isinstance(output, tuple):
                _, features = output
            else:
                features = output
            if features is not None:
                features_list.append(features.cpu().numpy())
    if features_list:
        features_array = np.concatenate(features_list, axis=0)
    else:
        features_array = np.array([])
    return features_array


def compute_statistics(features):
    """
    Compute mean, standard deviation, variance, and covariance for an array
    of features with shape (N, D), where N is the number of samples and D is the feature dimension.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    var = np.var(features, axis=0)
    cov = np.cov(features, rowvar=False) if features.shape[0] > 1 else None
    return mean, std, var, cov


def compute_mc_statistics(mc_features):
    """
    Compute MC statistics from an array of shape (MC, N, D) where MC is the number of MC iterations.
    For each sample (across MC iterations) we compute its mean, std, and variance.
    Then we average these per-sample statistics across samples.
    For covariance, we compute a covariance matrix for each sample and then average these matrices.
    """
    # mc_features: shape (MC, N, D)
    mc_means = np.mean(mc_features, axis=0)   # shape (N, D)
    mc_stds = np.std(mc_features, axis=0)       # shape (N, D)
    mc_vars = np.var(mc_features, axis=0)       # shape (N, D)
    
    N, D = mc_means.shape
    cov_matrices = []
    for i in range(N):
        sample_features = mc_features[:, i, :]  # shape (MC, D)
        if sample_features.shape[0] > 1:
            cov = np.cov(sample_features, rowvar=False)  # shape (D, D)
        else:
            cov = np.zeros((D, D))
        cov_matrices.append(cov)
    cov_matrices = np.array(cov_matrices)  # shape (N, D, D)
    avg_cov = np.mean(cov_matrices, axis=0)  # shape (D, D)
    
    avg_mean = np.mean(mc_means, axis=0)
    avg_std = np.mean(mc_stds, axis=0)
    avg_var = np.mean(mc_vars, axis=0)
    
    return avg_mean, avg_std, avg_var, avg_cov


def main():
    parser = argparse.ArgumentParser(
        description="Feature extraction with statistics computation for train/val and test splits."
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration YAML file.')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True,
                        help='Data split to extract features from.')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration and device
    config = load_config(args.config)
    device = torch.device(f"cuda:{config.misc.cuda}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Setup transforms and processor based on model type
    if config.model.name.lower() == "vision":
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        for_vision = True
        transform = None
    else:
        processor = None
        for_vision = False
        # For test and val we use the "none" augmentation
        if args.split in ["test", "val"]:
            transform = get_augmentation_pipeline("none", config.model.input_size)
        elif args.split == "train":
            transform = get_augmentation_pipeline(config.augmentation.type.lower(), config.model.input_size)
    
    # Read HDF5 indices and labels
    try:
        with h5py.File(config.paths.hdf5_path, 'r') as f:
            all_labels = f['labels'][:]
            total_samples = len(all_labels)
            all_indices = np.arange(total_samples)
    except Exception as e:
        logging.error(f"Error reading HDF5 file: {e}")
        return
    
    from sklearn.model_selection import train_test_split, StratifiedKFold
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices,
        all_labels,
        test_size=config.training.test_size,
        random_state=config.training.seed,
        stratify=all_labels
    )
    
    results = {}
    
    # Process each fold over the number specified in config.training.folds
    num_folds = config.training.folds
    fold_stats = {}
    
    if args.split in ['train', 'val']:
        # For train/val, extract features once (without MC dropout) per fold.
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.training.seed)
        splits = list(skf.split(train_val_indices, train_val_labels))
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            if args.split == "train":
                chosen_indices = train_val_indices[train_idx]
            else:
                chosen_indices = train_val_indices[val_idx]
            
            logging.info(f"[Split: {args.split}] Processing fold {fold_idx} with {len(chosen_indices)} samples.")
            
            dataset = HDF5Dataset(
                hdf5_file=config.paths.hdf5_path,
                indices=chosen_indices,
                transform=transform,
                processor=processor,
                for_vision=for_vision,
                cache_in_memory=config.HDF5Dataset.cache_in_memory
            )
            dataloader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
                prefetch_factor=config.data.prefetch_factor
            )
            
            # Load the saved model checkpoint for this fold
            checkpoint_path = join("checkpoints", config.paths.run_name,
                                   f"{config.paths.output_path_model}_fold{fold_idx}_final.pth")
            if not os.path.isfile(checkpoint_path):
                logging.error(f"Checkpoint not found for fold {fold_idx} at {checkpoint_path}")
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = get_bayesian_model(
                model_name=config.model.name,
                num_classes=config.model.num_classes,
                input_size=config.model.input_size,
                use_checkpointing=config.model.use_checkpointing,
                dropout_p=config.model.dropout_p,
                device=device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Extract features (without MC dropout)
            features = extract_features(model, dataloader, device, mc_dropout=False)
            if features.size == 0:
                logging.warning(f"No features extracted for fold {fold_idx}.")
                continue
            mean, std, var, cov = compute_statistics(features)
            fold_stats[f"fold_{fold_idx}"] = {
                "mean": mean,
                "std": std,
                "var": var,
                "cov": cov
            }
            logging.info(f"Computed statistics for fold {fold_idx} on split '{args.split}'.")
        results["fold_stats"] = fold_stats
        output_path = join("metrics", config.paths.run_name, f"{args.split}_feature_stats.npz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, **results)
        logging.info(f"Feature statistics for '{args.split}' split saved to {output_path}.")
        
    elif args.split == "test":
        # For test, run MC iterations for each fold
        mc_iterations = config.training.mc_iterations
        for fold_idx in range(1, num_folds + 1):
            logging.info(f"[Test Split] Processing fold {fold_idx} with MC iterations = {mc_iterations}.")
            
            dataset = HDF5Dataset(
                hdf5_file=config.paths.hdf5_path,
                indices=test_indices,
                transform=transform,
                processor=processor,
                for_vision=for_vision,
                cache_in_memory=config.HDF5Dataset.cache_in_memory
            )
            dataloader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
                prefetch_factor=config.data.prefetch_factor
            )
            
            # Load the saved model checkpoint for this fold
            checkpoint_path = join("checkpoints", config.paths.run_name,
                                   f"{config.paths.output_path_model}_fold{fold_idx}_final.pth")
            if not os.path.isfile(checkpoint_path):
                logging.error(f"Checkpoint not found for fold {fold_idx} at {checkpoint_path}")
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = get_bayesian_model(
                model_name=config.model.name,
                num_classes=config.model.num_classes,
                input_size=config.model.input_size,
                use_checkpointing=config.model.use_checkpointing,
                dropout_p=config.model.dropout_p,
                device=device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Run MC iterations and collect features for each iteration
            mc_features_list = []
            for mc in range(mc_iterations):
                logging.info(f"Fold {fold_idx} - MC Iteration {mc+1}/{mc_iterations}")
                features = extract_features(model, dataloader, device, mc_dropout=True)
                if features.size == 0:
                    logging.warning(f"No features extracted in MC iteration {mc+1} for fold {fold_idx}.")
                    continue
                mc_features_list.append(features)
            if not mc_features_list:
                logging.error(f"No features extracted for fold {fold_idx} in any MC iteration.")
                continue
            # Stack features: shape (MC, N, D)
            mc_features_array = np.stack(mc_features_list, axis=0)
            # Compute statistics from MC features
            mean, std, var, cov = compute_mc_statistics(mc_features_array)
            fold_stats[f"fold_{fold_idx}"] = {
                "mean": mean,
                "std": std,
                "var": var,
                "cov": cov
            }
            logging.info(f"Computed MC statistics for fold {fold_idx}.")
        results["fold_stats"] = fold_stats
        output_path = join("metrics", config.paths.run_name, "test_feature_stats.npz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, **results)
        logging.info(f"Test feature statistics saved to {output_path}.")
    else:
        logging.error(f"Invalid split: {args.split}")


if __name__ == "__main__":
    main()
