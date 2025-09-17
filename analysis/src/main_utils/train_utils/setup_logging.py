# src/main_utils/train_utils/setup_logging.py

import logging
from os import makedirs
from os.path import join

def setup_logging(logs_dir: str) -> None:
    """
    Configure Python logging to write both to a file and to the console.

    Steps:
      1) Ensure the logs directory exists.
      2) Clear any pre-existing handlers to avoid duplicate logs.
      3) Set up a FileHandler (writes to training.log) and StreamHandler (console).
      4) Use a consistent timestamped format and INFO level by default.

    Args:
        logs_dir (str): Directory where 'training.log' will be created.
    """
    # 1) Create logs directory if needed
    makedirs(logs_dir, exist_ok=True)
    log_file = join(logs_dir, 'training.log')

    # 2) Remove any existing handlers on the root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3) Configure basic logging:
    #    - INFO level (or DEBUG for more verbosity)
    #    - timestamp, level, and message in each record
    #    - two handlers: file + stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # 4) Log a startup message
    logging.info(f"Logging initialized. Writing logs to: {log_file}")
