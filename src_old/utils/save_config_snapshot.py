# src/util/save_config_snapshot.py

from os.path import join
from yaml import dump
from logging import info, error

def save_config_snapshot(config, metrics_dir):
    """Save a snapshot of the configuration used for training."""
    config_snapshot_path = join(metrics_dir, 'config_snapshot.yaml')
    try:
        with open(config_snapshot_path, 'w') as f:
            # Serialize the Pydantic model to a dictionary
            config_dict = config.model_dump()

            # Dump the dictionary as YAML
            dump(config_dict, f, sort_keys=False)
        info(f"Configuration snapshot saved at {config_snapshot_path}")
    except Exception as e:
        error(f"Failed to save configuration snapshot: {e}")
        raise

