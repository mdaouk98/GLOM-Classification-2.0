# src/main_utils/train_utils/trainer_utils/train_one_epoch.py

import logging
import torch
from tqdm import tqdm
import math
import time
import numpy as np
from typing import Optional, Tuple, Any, Iterable, Callable
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from main_utils.helpers_utils import move_to_device, timeit  # Assumes `timeit` is a context manager.

def train_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    fold: int,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, Optional[float], float]:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device on which to perform training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        config (Any): Configuration object containing training parameters.
        fold (int): Fold number for cross-validation.
        epoch (int): Current epoch number.
        scaler (Optional[torch.cuda.amp.GradScaler], optional): GradScaler for mixed precision training. Defaults to None.
        writer (Optional[SummaryWriter], optional): TensorBoard SummaryWriter for logging. Defaults to None.

    Returns:
        Tuple[float, Optional[float], float]:
            - epoch_loss (float): Average loss over the epoch.
            - epoch_accuracy (Optional[float]): Training accuracy if applicable.
            - avg_gradient_norm (float): Average gradient norm.
    """
    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    gradient_norms: list[float] = []

    # Lists to store timing measurements.
    batch_load_times: list[float] = []      # Time to retrieve each batch from DataLoader.
    batch_process_times: list[float] = []     # Time for processing (forward/backward/optimizer step).

    # Initialize overall epoch timing.
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    progress_bar: Iterable[Tuple[Any, Any]] = tqdm(
        train_loader,
        desc=f'Fold {fold} | Epoch {epoch} - Training',
        leave=False
    )

    # Determine if targets are required for the loss function.
    requires_targets: bool = config.loss_function.lower() not in ['reversecrossentropyloss']

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        try:
            # Measure DataLoader retrieval time using the timeit context manager.
            with timeit() as get_load_time:  # type: Callable[[], float]
                inputs, labels = move_to_device((inputs, labels), device)
            load_time: float = get_load_time()
            batch_load_times.append(load_time)

            # Measure processing time (forward/backward pass, optimizer step, etc.).
            with timeit() as get_process_time:  # type: Callable[[], float]
                optimizer.zero_grad()

                # Use mixed precision if scaler is provided and device supports CUDA.
                if scaler is not None and device.type == 'cuda':
                    with autocast('cuda'):
                        outputs: torch.Tensor = model(inputs, mc_dropout=False)
                        loss: torch.Tensor = (criterion(outputs, labels) if requires_targets 
                                              else criterion(outputs))
                else:
                    outputs = model(inputs, mc_dropout=False)
                    loss = criterion(outputs, labels) if requires_targets else criterion(outputs)

                if scaler is not None and device.type == 'cuda':
                    scaler.scale(loss).backward()
                    if config.training.gradient_clipping:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clipping)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.training.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clipping)
                    optimizer.step()

            process_time: float = get_process_time()
            batch_process_times.append(process_time)

            # Calculate gradient norms for monitoring.
            total_norm: float = math.sqrt(
                sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
            )
            gradient_norms.append(total_norm)

            running_loss += loss.item() * inputs.size(0)
            if requires_targets:
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                current_loss: float = running_loss / total
                current_acc: float = 100.0 * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}', 
                    'Acc': f'{current_acc:.2f}%',
                    'LoadT (s)': f'{load_time:.3f}',
                    'ProcT (s)': f'{process_time:.3f}'
                })
            else:
                total += inputs.size(0)
                progress_bar.set_postfix({
                    'Loss': f'{running_loss / total:.4f}',
                    'LoadT (s)': f'{load_time:.3f}',
                    'ProcT (s)': f'{process_time:.3f}'
                })
        except Exception as e:
            logging.error(
                f"Error processing batch {batch_idx} in epoch {epoch} on fold {fold}: {e}",
                exc_info=True
            )
            continue

        # Log parameter distributions every 100 batches.
        if writer is not None and (batch_idx + 1) % 100 == 0:
            try:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Fold{fold}/Parameters/{name}', param.detach().cpu().numpy(), epoch)
            except Exception as e:
                logging.warning(
                    f"Failed to log histograms for batch {batch_idx} in epoch {epoch}: {e}"
                )

    # End overall epoch timing.
    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to complete.
        epoch_time_ms = start_event.elapsed_time(end_event)
        logging.info(f"Fold {fold} Epoch {epoch} - Training Epoch Time: {epoch_time_ms/1000:.2f} sec")
    else:
        epoch_time = time.time() - start_time
        logging.info(f"Fold {fold} Epoch {epoch} - Training Epoch Time: {epoch_time:.2f} sec")

    # Compute and log average retrieval and processing times.
    avg_load_time: float = sum(batch_load_times) / len(batch_load_times) if batch_load_times else 0.0
    avg_process_time: float = sum(batch_process_times) / len(batch_process_times) if batch_process_times else 0.0
    logging.info(f"Fold {fold} Epoch {epoch} - Average DataLoader retrieval time: {avg_load_time:.4f} sec per batch")
    logging.info(f"Fold {fold} Epoch {epoch} - Average processing time: {avg_process_time:.4f} sec per batch")

    epoch_loss: float = running_loss / len(train_loader.dataset)
    train_accuracy: Optional[float] = (100.0 * correct / total) if requires_targets and total > 0 else None
    avg_gradient_norm: float = np.mean(gradient_norms) if gradient_norms else 0.0

    return epoch_loss, train_accuracy, avg_gradient_norm
