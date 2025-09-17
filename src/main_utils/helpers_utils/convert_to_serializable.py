# src/main_utils/helper_utils/convert_to_serializable.py

import numpy as np

def convert_to_serializable(obj):
    """
    Recursively convert Python objects into JSON-serializable formats.

    - NumPy arrays -> Python lists
    - Dicts            -> dicts with serializable values
    - Lists            -> lists with serializable items
    - Other types      -> returned unchanged (assuming already serializable)
    """
    # 1) NumPy array: convert to nested Python lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 2) Dictionary: apply conversion to each value
    elif isinstance(obj, dict):
        serializable_dict = {}
        for key, value in obj.items():
            # Recursively convert each value
            serializable_dict[key] = convert_to_serializable(value)
        return serializable_dict

    # 3) List (or tuple): convert each element
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]

    # 4) Everything else: assume it's already JSON-serializable
    else:
        return obj
