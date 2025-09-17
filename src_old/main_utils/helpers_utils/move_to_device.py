# src/main_utils/helpers_utils/move_to_device.py

import torch

def move_to_device(batch, device):
    """
    Moves a batch of tensors to the specified device.
    If the batch is a tuple or list, all tensors are moved.
    """
    if isinstance(batch, (tuple, list)):
        return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
    elif isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
