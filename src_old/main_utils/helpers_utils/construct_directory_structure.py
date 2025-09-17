# src/utils/construct_directory_structure.py

from os import makedirs
from os.path import join

def construct_directory_structure(config):
    # Use the run_name from the configuration instead of generating one.
    dir_name = config.paths.run_name

    base_checkpoints_dir = 'checkpoints'
    base_metrics_dir = 'metrics'
    base_runs_dir = 'runs'

    checkpoints_dir = join(base_checkpoints_dir, dir_name)
    metrics_dir = join(base_metrics_dir, dir_name)
    logs_dir = join(metrics_dir, 'logs')
    runs_dir = join(base_runs_dir, dir_name)

    makedirs(checkpoints_dir, exist_ok=True)
    makedirs(metrics_dir, exist_ok=True)
    makedirs(logs_dir, exist_ok=True)
    makedirs(runs_dir, exist_ok=True)

    return {
        'checkpoints_dir': checkpoints_dir,
        'metrics_dir': metrics_dir,
        'logs_dir': logs_dir,
        'runs_dir': runs_dir,
        'dir_name': dir_name
    }

