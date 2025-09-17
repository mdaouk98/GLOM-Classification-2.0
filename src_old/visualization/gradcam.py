# src/visualization/gradcam.py

"""
This script generates a Grad-CAM visualization for a single image sample.
It uses the same configuration and data splitting logic as main_train.py,
and loads the model via the modified get_bayesian_model (which returns standard
outputs). The script attaches hooks to a specified target layer (e.g., "layer4")
to capture activations and gradients, computes the Grad-CAM heatmap, overlays it
on the original image, and saves/displays the result.
"""

import argparse
import logging
import os
from os.path import join
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import tqdm

# Import configuration and dataset/model utilities.
from config import load_config
from datasets import HDF5Dataset
from augmentations import get_augmentation_pipeline
from models.get_bayesian_model import get_bayesian_model

# Global variables to store activations and gradients.
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

def generate_gradcam(model, input_tensor, target_class=None, target_layer=None, device="cpu"):
    """
    Generates Grad-CAM for the provided input_tensor using the specified target_layer.
    If target_class is None, the predicted class is used.
    
    Returns:
        cam (numpy.ndarray): The computed Grad-CAM heatmap (normalized to [0,1]).
        target_class (int): The target class used.
    """
    global activations, gradients
    activations = None
    gradients = None

    # Register hooks on the target layer.
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass.
    input_tensor = input_tensor.to(device)
    output = model(input_tensor, mc_dropout=False, return_features=False)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass: compute gradient of the score for the target class.
    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    # Remove hooks.
    handle_forward.remove()
    handle_backward.remove()

    # Compute weights: global-average pooling of gradients.
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # shape: (1, C, 1, 1)
    cam = torch.sum(weights * activations, dim=1)  # shape: (1, H, W)
    cam = torch.relu(cam)

    # Normalize CAM between 0 and 1.
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()
    cam = cam.squeeze(0).cpu().numpy()  # shape: (H, W)
    return cam, target_class

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualization for a single image sample."
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
        default="test",
        help="Data split to use (default: test).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number for train/val splits (ignored for test).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the sample image to visualize.",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="layer4",
        help="Name of the target layer for Grad-CAM (for CNN backbones).",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="gradcam.png",
        help="Path to save the Grad-CAM overlay image.",
    )
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Grad-CAM visualization.")

    # Load configuration.
    config = load_config(args.config)
    logging.info("Configuration loaded successfully.")

    # Set device.
    device = torch.device(f"cuda:{config.misc.cuda}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set up transform for visualization (use validation transform).
    augmentation_type = config.augmentation.type.lower()
    if augmentation_type not in ["mixup", "cutmix"]:
        transform_for_val = get_augmentation_pipeline("none", config.model.input_size)
    else:
        transform_for_val = get_augmentation_pipeline("basic", config.model.input_size)

    processor = None
    for_vision = False
    if config.model.name.lower() == "vision":
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        for_vision = True
        transform_for_val = None

    # Read the HDF5 file to get indices and labels.
    try:
        import h5py
        with h5py.File(config.paths.hdf5_path, "r") as f:
            total_samples = len(f["labels"])
            all_indices = np.arange(total_samples)
            all_labels = f["labels"][:]
    except Exception as e:
        logging.error(f"Error reading HDF5 file: {e}")
        return

    # Split the data (same as in main_train.py).
    from sklearn.model_selection import train_test_split, StratifiedKFold
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices,
        all_labels,
        test_size=config.training.test_size,
        random_state=config.training.seed,
        stratify=all_labels,
    )

    if args.split == "test":
        chosen_indices = test_indices
        chosen_transform = transform_for_val
        logging.info("Using test split.")
    else:
        skf = StratifiedKFold(n_splits=config.training.folds, shuffle=True, random_state=config.training.seed)
        splits = list(skf.split(train_val_indices, train_val_labels))
        if args.fold < 1 or args.fold > len(splits):
            logging.error(f"Invalid fold number {args.fold}. Must be between 1 and {len(splits)}.")
            return
        train_idx, val_idx = splits[args.fold - 1]
        if args.split == "train":
            chosen_indices = train_val_indices[train_idx]
            chosen_transform = transform_for_val
            logging.info(f"Using fold {args.fold} training split.")
        elif args.split == "val":
            chosen_indices = train_val_indices[val_idx]
            chosen_transform = transform_for_val
            logging.info(f"Using fold {args.fold} validation split.")

    if args.index < 0 or args.index >= len(chosen_indices):
        logging.error(f"Index {args.index} is out of range for the selected split (size {len(chosen_indices)}).")
        return

    # Create the dataset and DataLoader for a single sample.
    dataset = HDF5Dataset(
        hdf5_file=config.paths.hdf5_path,
        indices=[chosen_indices[args.index]],
        transform=chosen_transform,
        processor=processor,
        for_vision=for_vision,
        cache_in_memory=True,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model.
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

    # Identify the target layer.
    target_layer = None
    # For CNN backbones, assume the backbone is stored in model.base.
    if hasattr(model, "base"):
        target_layer = getattr(model.base, args.target_layer, None)
    if target_layer is None:
        logging.error(f"Target layer '{args.target_layer}' not found in the model.")
        return

    # Get the sample image.
    for batch in dataloader:
        inputs, labels = batch
        break

    # Generate Grad-CAM.
    cam, target_class = generate_gradcam(model, inputs, target_class=None, target_layer=target_layer, device=device)
    logging.info(f"Grad-CAM generated for target class {target_class}.")

    # Retrieve the original image (from the dataset output).
    image = inputs[0].cpu().numpy()  # Assume shape (C, H, W)
    image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
    # Convert to uint8 for visualization (assuming values in [0,1] or already in [0,255]).
    if image.max() <= 1:
        image_uint8 = (image * 255).astype("uint8")
    else:
        image_uint8 = image.astype("uint8")

    # Resize CAM to match the image dimensions.
    cam_resized = cv2.resize(cam, (image_uint8.shape[1], image_uint8.shape[0]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_uint8, 0.6, heatmap, 0.4, 0)

    # Save the overlay image.
    cv2.imwrite(args.output_image, overlay)
    logging.info(f"Grad-CAM overlay image saved to {args.output_image}.")

    # Display the overlay image.
    plt.figure(figsize=(8, 8))
    # OpenCV reads in BGR; convert to RGB.
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM for target class {target_class}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
