# src/main_utils/helper_utils/set_seed.py


import torch
import numpy as np
import random

def set_seed(seed: int):
    """
    Set random seeds across various libraries for reproducible experiments.

    Args:
        seed (int): The seed value to use.
    """
    # 1) PyTorch CPU seed
    torch.manual_seed(seed)
    
    # 2) PyTorch CUDA seeds (for single- and multi-GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 3) NumPy seed
    np.random.seed(seed)
    
    # 4) Python built-in random module seed
    random.seed(seed)
    
    # 5) Configure CuDNN for reproducibility:
    #    - deterministic: forces cuDNN to use only deterministic convolution algorithms
    #    - benchmark=False: disables the auto-tuner that selects best algorithm at runtime
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

