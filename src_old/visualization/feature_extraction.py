# src/visualization/feature_extraction.py

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
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import tqdm

# Import the load_config function from your configuration module
from config import load_config
# Import the HDF5Dataset (used in main_train.py)
from datasets import HDF5Dataset
# Import your augmentation pipeline
from augmentations import get_augmentation_pipeline
# Import the model factory
from models.get_bayesian_model import get_bayesian_model


def main():
    parser = argparse.ArgumentParser(
        description="Extract intermediate features from a trained model for clustering/visualization."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        required=True,
        help="Which data split to extract features from (train, val, or test).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=5,
        help="Fold number for train/val splits (default: 1). Ignored if split=test.",
    )
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting feature extraction.")

    # Load configuration.
    config = load_config(args.config)
    logging.info("Configuration loaded successfully.")

    # Set device.
    device = torch.device(f"cuda:{config.misc.cuda}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set up data transforms.
    augmentation_type = config.augmentation.type.lower()
    if augmentation_type not in ["mixup", "cutmix"]:
        transform_for_train = get_augmentation_pipeline(augmentation_type, config.model.input_size)
    else:
        transform_for_train = get_augmentation_pipeline("basic", config.model.input_size)
    transform_for_val = get_augmentation_pipeline("none", config.model.input_size)

    processor = None
    for_vision = False
    if config.model.name.lower() == "vision":
        # If using a Vision Transformer, load the processor and disable transforms.
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        for_vision = True
        transform_for_train = None
        transform_for_val = None

    # Open the HDF5 file to get indices and labels.
    try:
        with h5py.File(config.paths.hdf5_path, "r") as f:
            if "images" not in f or "labels" not in f:
                raise KeyError("HDF5 file must contain 'images' and 'labels' datasets.")
            total_samples = len(f["labels"])
            all_indices = np.arange(total_samples)
            all_labels = f["labels"][:]
    except Exception as e:
        logging.error(f"Error reading HDF5 file: {e}")
        return

    logging.info(f"Total samples in dataset: {total_samples}")

    # Perform the train-test split (same as in main_train.py).
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices,
        all_labels,
        test_size=config.training.test_size,
        random_state=config.training.seed,
        stratify=all_labels,
    )

    # Select the indices and transforms based on the chosen split.
    if args.split == "test":
        chosen_indices = test_indices
        chosen_labels = test_labels
        chosen_transform = transform_for_val
        shuffle_flag = False
        logging.info("Using the test split.")
    else:
        # Use StratifiedKFold on the train_val split.
        skf = StratifiedKFold(n_splits=config.training.folds, shuffle=True, random_state=config.training.seed)
        splits = list(skf.split(train_val_indices, train_val_labels))
        if args.fold < 1 or args.fold > len(splits):
            logging.error(f"Invalid fold number {args.fold}. Must be between 1 and {len(splits)}.")
            return
        train_idx, val_idx = splits[args.fold - 1]
        if args.split == "train":
            chosen_indices = train_val_indices[train_idx]
            chosen_labels = train_val_labels[train_idx]
            chosen_transform = transform_for_train
            shuffle_flag = True
            logging.info(f"Using fold {args.fold} training split.")
        elif args.split == "val":
            chosen_indices = train_val_indices[val_idx]
            chosen_labels = train_val_labels[val_idx]
            chosen_transform = transform_for_val
            shuffle_flag = False
            logging.info(f"Using fold {args.fold} validation split.")

    logging.info(f"Number of samples in the {args.split} split: {len(chosen_indices)}")

    # Create the dataset.
    dataset = HDF5Dataset(
        hdf5_file=config.paths.hdf5_path,
        indices=chosen_indices,
        transform=chosen_transform,
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=False,
    )

    # Create the DataLoader.
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle_flag,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=config.data.prefetch_factor,
    )

    logging.info("DataLoader created successfully.")

    # Construct the output path using the configuration.
    # Here we assume that the config.paths contains run_name and output_path_dict.
    metrics_dir = join("metrics", config.paths.run_name)
    os.makedirs(metrics_dir, exist_ok=True)
    # Use the output_path_dict as a base name and append a suffix for features.
    output_path = join(metrics_dir, args.split + "_features.npz")
    logging.info(f"Features will be saved to: {output_path}")

    # Load the model using the factory function.
    model = get_bayesian_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        input_size=config.model.input_size,
        use_checkpointing=config.model.use_checkpointing,
        dropout_p=config.model.dropout_p,
        device=device,
    )
    model.eval()
    logging.info("Model loaded and set to evaluation mode.")

    # Loop through the dataloader and extract features.
    all_features = []
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            inputs, labels = batch
            inputs = inputs.to(device)
            # Run forward pass with return_features=True so the model returns (logits, features)
            outputs = model(inputs, mc_dropout=False, return_features=True)
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                logits = outputs
                features = None

            # Collect predictions and true labels.
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

            # Collect features.
            if features is not None:
                all_features.append(features.cpu().numpy())

    # Concatenate batch results.
    all_features = np.concatenate(all_features, axis=0) if len(all_features) > 0 else None
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    logging.info(f"Extracted features shape: {all_features.shape if all_features is not None else 'None'}")
    logging.info(f"Predictions shape: {all_predictions.shape}")
    logging.info(f"True labels shape: {all_true_labels.shape}")

    # Save the extracted features, predictions, and labels.
    np.savez(output_path, features=all_features, predictions=all_predictions, labels=all_true_labels)
    logging.info(f"Extracted data saved to {output_path}")


if __name__ == "__main__":
    main()
