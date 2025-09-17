# src/trainer/train_one_epoch.py

import logging
import torch
from tqdm import tqdm
import time
import math
import numpy as np
from typing import Optional, Tuple, Any
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

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
    Train model for one epoch.
    Returns:
        epoch_loss (float): Average loss over the epoch.
        epoch_accuracy (Optional[float]): Training accuracy if applicable.
        avg_gradient_norm (float): Average gradient norm.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []

    # Lists to store timing measurements.
    batch_load_times = []      # Time to retrieve each batch from DataLoader.
    batch_process_times = []   # Time for processing (forward/backward/optimizer step).

    # Start the timer for the very first batch retrieval.
    prev_time = time.time()

    progress_bar = tqdm(
        train_loader,
        desc=f'Fold {fold} | Epoch {epoch} - Training',
        leave=False
    )

    requires_targets = config.loss_function.lower() not in ['reversecrossentropyloss']

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        # Record retrieval time.
        current_time = time.time()
        load_time = current_time - prev_time
        batch_load_times.append(load_time)
        # Reset timer for processing.
        process_start_time = time.time()

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Use bfloat16 for mixed precision.
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs, mc_dropout=False)
            loss = criterion(outputs, labels) if requires_targets else criterion(outputs)

        if scaler is not None:
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

        # Record processing time for this batch.
        process_end_time = time.time()
        process_time = process_end_time - process_start_time
        batch_process_times.append(process_time)

        # Calculate gradient norms.
        total_norm = math.sqrt(
            sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
        )
        gradient_norms.append(total_norm)

        running_loss += loss.item() * inputs.size(0)
        if requires_targets:
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
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

        # Prepare for the next iteration: record the current time as previous.
        prev_time = time.time()

        # Log parameter distributions every 100 batches.
        if writer and (batch_idx + 1) % 100 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Fold{fold}/Parameters/{name}', param.detach().cpu().numpy(), epoch)

    # Compute and log average retrieval and processing times.
    avg_load_time = sum(batch_load_times) / len(batch_load_times) if batch_load_times else 0.0
    avg_process_time = sum(batch_process_times) / len(batch_process_times) if batch_process_times else 0.0
    logging.info(f"Fold {fold} Epoch {epoch} - Average DataLoader retrieval time: {avg_load_time:.4f} sec per batch")
    logging.info(f"Fold {fold} Epoch {epoch} - Average processing time: {avg_process_time:.4f} sec per batch")

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = (100.0 * correct / total) if requires_targets else None
    avg_gradient_norm = np.mean(gradient_norms)

    return epoch_loss, train_accuracy, avg_gradient_norm
