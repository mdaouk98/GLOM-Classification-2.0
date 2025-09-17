# src/utils/convert_to_serializable.py

import numpy as np

def convert_to_serializable(obj):
    """Convert objects to serializable formats (e.g., numpy arrays to lists)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
