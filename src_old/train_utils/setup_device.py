# src/train_utils/setup_device.py

import torch
from torch import cuda
import logging
from typing import Any

def setup_device(config: Any) -> torch.device:
    """
    Set up the computing device and CUDA configuration.

    Args:
        config (Any): Configuration object containing device settings.

    Returns:
        torch.device: The device (GPU or CPU) to be used.
    """
    device = torch.device(f'cuda:{config.misc.cuda}' if cuda.is_available() else 'cpu')
    if cuda.is_available():
        logging.info(f"[Device Setup] CUDA device {config.misc.cuda} is available")
        cuda.set_device(device)
        cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    logging.info(f"[Device Setup] Using device: {device}")
    return device
