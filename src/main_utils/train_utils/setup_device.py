# src/main_utils/train_utils/setup_device.py

import logging
import torch
from torch import cuda
from typing import Any

def setup_device(config: Any) -> torch.device:
    """
    Configure and return the appropriate torch.device (GPU or CPU).

    Steps:
      1) Check for CUDA availability.
      2) Select the GPU index from config.misc.cuda if available, else CPU.
      3) Set the active CUDA device and clear its cache.
      4) Enable CuDNN autotuner for improved performance.
      5) Log the chosen device.

    Args:
        config (Any): Configuration object with attribute `misc.cuda` (GPU index).

    Returns:
        torch.device: The selected device.
    """
    # 1) Determine device string: 'cuda:<index>' if CUDA available else 'cpu'
    if cuda.is_available():
        device_str = f"cuda:{config.misc.cuda}"
    else:
        device_str = "cpu"

    device = torch.device(device_str)

    # 2) If using CUDA, set the GPU and clear any cached memory
    if cuda.is_available():
        logging.info(f"[Device Setup] CUDA is available. Using GPU {config.misc.cuda}.")
        cuda.set_device(device)
        cuda.empty_cache()
    else:
        logging.info("[Device Setup] CUDA not available. Falling back to CPU.")

    # 3) Enable CuDNN benchmark for optimized convolution performance
    torch.backends.cudnn.deterministic = config.misc.deterministic
    torch.backends.cudnn.benchmark = not config.misc.deterministic

    logging.info(f"[Device Setup] Final device: {device}")
    return device


