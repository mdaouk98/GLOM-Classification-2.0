# src/main_utils/train_utils/trainer_utils/train_one_epoch.py

import logging
import time
import math
import torch
import math
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from main_utils.helpers_utils import timeit

def train_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterions: Dict[str, torch.nn.Module],  # head_name -> loss module
    loss_weights: Dict[str, float],           # head_name -> weight scalar
    optimizer: torch.optim.Optimizer,
    config: Any,
    fold: int,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, Dict[str, float], float]:
    """
    Train the model for one epoch (single- or multi-head).

    Returns:
      - epoch_loss (float): average loss over all samples
      - epoch_accs (Dict[str, float]): accuracy (%) per head
      - avg_gradient_norm (float): average L2 norm of gradients
    """
    model.train()

    # --- 1) Initialize running counters ---
    running_loss = 0.0
    total_samples = 0
    correct = {h: 0 for h in criterions}
    total   = {h: 0 for h in criterions}
    grad_norms, load_times, proc_times = [], [], []

    # --- 2) Set up timing events if on CUDA ---
    use_cuda = (device.type == 'cuda')
    if use_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        epoch_start = time.time()

    # --- 3) Progress bar over batches ---
    pbar = tqdm(
        train_loader,
        desc=f"Fold {fold} | Epoch {epoch} - Training",
        leave=False
    )
    # remember single-head key if only one
    single_head = next(iter(criterions)) if len(criterions)==1 else None
    
    # --- setup for gradient accumulation ---
    accumulation_steps = config.training.accumulation_steps  # e.g. 2 for 16 -> 32
    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(pbar):
        # 3a) Move data to device, measure load time
        with timeit() as t0:
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            if isinstance(labels, dict):
                labels = {h: v.to(device, non_blocking=True) for h, v in labels.items()}
            else:
                labels = labels.to(device, non_blocking=True)
        load_times.append(t0())

        # 3b) Forward + backward pass, measure processing time
        
        with timeit() as t1:
            # optionally wrap in autocast for mixed precision
            if scaler is not None and use_cuda:
                with autocast(device_type='cuda'):
                    outputs = model(inputs, mc_dropout=False)
            else:
                outputs = model(inputs, mc_dropout=False)

            # unify to dict for multi-head
            if not isinstance(outputs, dict):
                outputs = {single_head: outputs}

            # compute total weighted loss
            loss = 0.0
            if scaler is not None and use_cuda:
                # — compute the loss in full precision —
                with autocast(enabled=False, device_type='cuda'):
                    for head, crit in criterions.items():
                        preds = outputs[head].float()
                        tgt   = labels[head] if isinstance(labels, dict) else labels
                        try:
                            this_loss = crit(preds, tgt)
                        except TypeError:
                            this_loss = crit(preds)
                        loss += loss_weights[head] * this_loss
            else:
                # no AMP, just do the usual
                for head, crit in criterions.items():
                    preds = outputs[head]
                    tgt   = labels[head] if isinstance(labels, dict) else labels
                    try:
                        this_loss = crit(preds, tgt)
                    except TypeError:
                        this_loss = crit(preds)
                    loss += loss_weights[head] * this_loss

            # backward + optimizer step (with optional gradient clipping)
            
            if scaler is not None and use_cuda:
                scaler.scale(loss).backward()
                       
                # only step/update every 'accumulation_steps' micro-batches:
                if (batch_idx + 1) % accumulation_steps == 0:
                    if config.training.gradient_clipping:
                        scaler.unscale_(optimizer)
                        # 3c-1) Record gradient norm
                        total_norm = math.sqrt(sum(
                            p.grad.data.norm(2).item()**2
                            for p in model.parameters() if p.grad is not None
                        ))
                        grad_norms.append(total_norm)
                        
                        if not math.isfinite(total_norm):
                            logging.error(f"Skipping update: non-finite grad_norm={total_norm:.1e} at batch {batch_idx}")
                            if config.training.remove_infinite_grad_batch:
                              optimizer.zero_grad()    # clear whatever garbage grads you have
                              scaler.update()       # <- reset the scaler's "unscaled" flag
                              continue
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.training.gradient_clipping)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
            else:
                loss.backward()
                
                # 3c-2) Record gradient norm
                total_norm = math.sqrt(sum(
                    p.grad.data.norm(2).item()**2
                    for p in model.parameters() if p.grad is not None
                ))
                grad_norms.append(total_norm)
                
                if not math.isfinite(total_norm):
                      logging.error(f"Skipping update: non-finite grad_norm={total_norm:.1e} at batch {batch_idx}")
                      if config.training.remove_infinite_grad_batch:
                          optimizer.zero_grad()    # clear whatever garbage grads you have
                          continue
                
                if config.training.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.training.gradient_clipping)
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
        proc_times.append(t1())       
        

        # 3d) Accumulate loss & update accuracy counts
        bsz = inputs.size(0)
        running_loss += loss.item() * bsz
        total_samples += bsz
        postfix = {'Loss': f"{running_loss/total_samples:.4f}"}
        for head in criterions:
            preds = outputs[head].argmax(dim=1)
            true  = labels[head] if isinstance(labels, dict) else labels
            correct[head] += (preds == true).sum().item()
            total[head]   += bsz
            acc = 100.0 * correct[head] / total[head]
            postfix[f"Acc_{head}"] = f"{acc:.2f}%"
        pbar.set_postfix(postfix)

        # 3e) Periodic TensorBoard histograms
        if writer and (batch_idx+1) % 100 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"Fold{fold}/Param/{name}",
                                     param.detach().cpu().numpy(),
                                     epoch)

    # --- 4) Finalize timing ---
    if use_cuda:
        end_evt.record()
        torch.cuda.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0
    else:
        elapsed = time.time() - epoch_start
    logging.info(
        f"Fold {fold}, Epoch {epoch} training time: {elapsed:.2f}s | "
        f"avg load {np.mean(load_times):.4f}s, avg proc {np.mean(proc_times):.4f}s"
    )

    # --- 5) Compute epoch-level metrics ---
    epoch_loss = running_loss / total_samples
    epoch_accs = {
        head: (100.0 * correct[head] / total[head]) if total[head] > 0 else 0.0
        for head in criterions
    }
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    
    logging.info(f"Fold {fold} | Epoch {epoch} completed | avg_grad_norm = {avg_grad_norm:.4f}")


    return epoch_loss, epoch_accs, avg_grad_norm


