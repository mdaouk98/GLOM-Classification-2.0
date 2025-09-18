# src/main_utils/helper_utils/construct_directory_structure.py

import os
from os.path import join

def construct_directory_structure(config):
    """
    Create a standard set of directories for a training run based on the
    provided configuration.

    Args:
        config: Configuration object with a `paths.run_name` attribute.

    Returns:
        dict: Paths to the created directories and the run name:
            {
                'checkpoints_dir': ...,
                'metrics_dir': ...,
                'logs_dir': ...,
                'runs_dir': ...,
                'dir_name': ...
            }
    """

    # 1) Determine the unique run name from the config
    dir_name = config.paths.run_name

    # 2) Define base directory names
    base_checkpoints_dir = 'checkpoints'
    base_metrics_dir    = 'metrics'
    base_runs_dir       = 'runs'

    # 3) Build full paths by joining base + run name
    checkpoints_dir = join(base_checkpoints_dir, dir_name)
    metrics_dir     = join(base_metrics_dir, dir_name)
    logs_dir        = join(metrics_dir, 'logs')  # nested under metrics
    runs_dir        = join(base_runs_dir, dir_name)

    # 4) Create directories if they don’t already exist
    #    exist_ok=True avoids error if directory is already present
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir,     exist_ok=True)
    os.makedirs(logs_dir,        exist_ok=True)
    os.makedirs(runs_dir,        exist_ok=True)

    # 5) Return a dictionary of all relevant paths
    return {
        'checkpoints_dir': checkpoints_dir,
        'metrics_dir':     metrics_dir,
        'logs_dir':        logs_dir,
        'runs_dir':        runs_dir,
        'dir_name':        dir_name
    }

