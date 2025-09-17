# src/utils/sanitize_string.py

import re

from torch.utils.tensorboard import SummaryWriter

def sanitize_string(s: str) -> str:
    """Sanitize a string to be used in file or directory names."""
    s = re.sub(r'[^\w\-]', '_', s)
    return s
