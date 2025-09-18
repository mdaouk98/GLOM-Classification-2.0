# src/main_utils/helper_util/save_config_snapshot.py

from os.path import join
from yaml import dump
from logging import info, error

def save_config_snapshot(config, metrics_dir):
    """
    Save a YAML snapshot of the training configuration for later reference.

    Args:
        config:       A Pydantic model (or similar) holding configuration parameters.
        metrics_dir:  Directory where the snapshot file should be written.
    """
    # 1) Build the full file path under the metrics directory
    config_snapshot_path = join(metrics_dir, 'config_snapshot.yaml')

    try:
        # 2) Open the target file in write mode (overwrites existing file)
        with open(config_snapshot_path, 'w') as f:
            # 3) Convert the Pydantic config model into a plain Python dict
            config_dict = config.model_dump()

            # 4) Serialize the dict to YAML, preserving key order
            dump(config_dict, f, sort_keys=False)

        # 5) Log success message
        info(f"Configuration snapshot saved at {config_snapshot_path}")

    except Exception as e:
        # 6) Log error and re-raise so upstream code knows it failed
        error(f"Failed to save configuration snapshot: {e}")
        raise

