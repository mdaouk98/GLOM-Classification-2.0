# src/main_utils/evaluation_utils/monte_carlo_prediction.py

import logging
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def monte_carlo_prediction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_iter: int = 10,
    writer: Optional[SummaryWriter] = None,
    fold: int = 1,
    num_classes: Optional[int] = None  # Optional parameter for number of classes
) -> Dict[str, np.ndarray]:
    """
    Perform Monte Carlo (MC) prediction with MC Dropout enabled and collect softmax outputs.

    This function performs multiple stochastic forward passes through the model with dropout enabled
    to estimate uncertainty by collecting softmax outputs across all MC iterations.

    Parameters:
    ----------
    model : torch.nn.Module
        The trained PyTorch model. The model's `forward` method must accept an `mc_dropout` boolean
        parameter to toggle dropout during inference.

    dataloader : torch.utils.data.DataLoader
        DataLoader for the evaluation/test dataset. It should have `shuffle=False` to ensure consistent
        ordering between label collection and prediction phases.

    device : torch.device
        The device (CPU or GPU) on which computations will be performed.

    n_iter : int, optional (default=10)
        Number of Monte Carlo iterations (forward passes) to perform.

    writer : Optional[SummaryWriter], optional (default=None)
        TensorBoard SummaryWriter for logging metrics.

    fold : int, optional (default=1)
        Fold number for cross-validation logging purposes.

    num_classes : Optional[int], optional (default=None)
        The number of classes. If not provided, it will be inferred from the model's output during the first forward pass.

    Returns:
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
            - 'all_softmax_iterations': np.ndarray of shape (n_iter, num_samples, num_classes)
                The softmax probabilities from each Monte Carlo iteration.
            - 'all_labels': np.ndarray of shape (num_samples,)
                The ground-truth labels corresponding to the softmax outputs.

    Raises:
    ------
    ValueError
        If the provided DataLoader is empty, if there's a mismatch between predicted and actual sample counts,
        or if the model's output dimension does not match the inferred or provided number of classes.
    """

    # Ensure the model is in evaluation mode
    model.eval()
    model.to(device)

    # Step 1: Collect all labels
    all_labels = []
    logging.info("Collecting labels from the DataLoader...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Collecting Labels', leave=False):
            all_labels.extend(labels.cpu().numpy())

    total_samples = len(all_labels)
    logging.info(f"Total samples in DataLoader: {total_samples}")

    if total_samples == 0:
        logging.error("No samples found in the DataLoader.")
        raise ValueError("The provided DataLoader is empty.")

    # Step 2: Determine number of classes
    if num_classes is None:
        # Attempt to infer num_classes from model's output
        logging.info("Inferring number of classes from the model's output...")
        try:
            # Get a batch of data
            dataloader_iter = iter(dataloader)
            inputs, _ = next(dataloader_iter)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs, mc_dropout=False)
                if outputs.dim() == 1:
                    inferred_num_classes = 1
                else:
                    inferred_num_classes = outputs.size(1)
            num_classes = inferred_num_classes
            logging.info(f"Inferred number of classes from model's output: {num_classes}")
        except StopIteration:
            logging.error("DataLoader is empty while trying to infer number of classes.")
            raise ValueError("Cannot infer number of classes from an empty DataLoader.")
        except TypeError as e:
            logging.error(f"Error during model's forward pass for inferring num_classes: {e}")
            raise ValueError("The model's forward method does not support inference for num_classes determination.")
    else:
        logging.info(f"Using provided number of classes: {num_classes}")

    # Validate that labels are within the expected range
    all_labels_array = np.array(all_labels)
    if not np.all((all_labels_array >= 0) & (all_labels_array < num_classes)):
        logging.error("Some labels are outside the range [0, num_classes-1].")
        raise ValueError("Label values exceed the number of classes specified.")

    # Step 3: Initialize storage for softmax outputs across MC iterations
    all_softmax_iterations = np.zeros((n_iter, total_samples, num_classes), dtype=np.float32)
    logging.debug(f"Initialized all_softmax_iterations with shape: {all_softmax_iterations.shape}")

    # Step 4: Perform MC iterations
    for iteration in range(n_iter):
        logging.info(f'Fold {fold}, MC Iteration {iteration + 1}/{n_iter}')
        progress_bar = tqdm(dataloader, desc=f'Fold {fold} MC Iteration {iteration + 1}', leave=False)

        sample_count = 0  # Reset sample count for each iteration

        for inputs, _ in progress_bar:
            inputs = inputs.to(device)
            with torch.no_grad():
                # Perform forward pass with dropout enabled
                try:
                    outputs = model(inputs, mc_dropout=True)  # Ensure model's forward supports mc_dropout
                except TypeError as e:
                    logging.error(f"Model's forward method does not support 'mc_dropout' parameter: {e}")
                    raise ValueError("The model's forward method does not support the 'mc_dropout' parameter.")
                
                if outputs.dim() == 1:
                    # Binary classification case
                    softmax_output = torch.sigmoid(outputs).cpu().numpy().reshape(-1, 1)  # Shape: (batch_size, 1)
                else:
                    softmax_output = F.softmax(outputs, dim=1).cpu().numpy()  # Shape: (batch_size, num_classes)

            current_batch_size = softmax_output.shape[0]
            logging.debug(f"Iteration {iteration + 1}, Batch size: {current_batch_size}")

            if sample_count + current_batch_size > total_samples:
                logging.error(
                    f"Sample count ({sample_count}) + current batch size ({current_batch_size}) exceeds total samples ({total_samples})."
                )
                raise ValueError("Batch size exceeds total_samples during softmax accumulation.")

            # Store softmax probabilities
            all_softmax_iterations[iteration, sample_count:sample_count + current_batch_size] = softmax_output
            sample_count += current_batch_size

    logging.info("Completed collecting softmax outputs for all MC iterations.")

    # Optional: Log basic information to TensorBoard
    if writer is not None:
        writer.add_scalar(f'Fold{fold}/MC_Average/Iterations', n_iter, fold)
        writer.add_scalar(f'Fold{fold}/MC_Average/Num_Samples', total_samples, fold)

    return {
        'all_softmax_iterations': all_softmax_iterations,  # Shape: (n_iter, num_samples, num_classes)
        'all_labels': all_labels_array                       # Shape: (num_samples,)
    }
