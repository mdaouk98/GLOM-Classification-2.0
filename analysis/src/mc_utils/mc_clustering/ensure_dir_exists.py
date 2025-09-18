# src/mc_utils/mc_clustering/ensure_dir_exists.py

import os

def ensure_dir_exists(path: str) -> None:
    """
    Ensure that the directory at the given path exists.
    If the directory does not exist, it is created.
    
    Args:
        path (str): The path of the directory to ensure.
    """
    os.makedirs(path, exist_ok=True)
