# src/main_utils/helpers_utils/move_to_device.py

import torch

def move_to_device(batch, device):
    """
    Recursively move all torch.Tensor objects in a batch to the given device.

    Supports:
      - torch.Tensor
      - tuple or list of tensors / nested structures
      - dicts with tensor values
      - leaves non-tensor objects untouched

    Args:
        batch: A Tensor, or a collection (tuple/list/dict) containing Tensors.
        device: The target device (e.g., 'cuda', 'cpu', or torch.device).
    
    Returns:
        The same structure as 'batch', but with all Tensors moved to 'device'.
    """
    # Case 1: Tuple or list -> move each element, preserve type as tuple
    if isinstance(batch, (tuple, list)):
        return tuple(
            # If element is a Tensor, .to(device); otherwise leave as-is
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in batch
        )

    # Case 2: Dictionary -> move each value, keep same keys
    elif isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # Case 3: Single Tensor -> move directly
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)

    # Case 4: Other types (numbers, strings, etc.) -> return unchanged
    else:
        return batch
