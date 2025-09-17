# src/train_utils/setup_logging.py

from os import makedirs
from os.path import join
import logging

def setup_logging(logs_dir: str) -> None:
    """
    Set up logging configuration.

    Args:
        logs_dir (str): Directory where log files will be stored.
    """
    makedirs(logs_dir, exist_ok=True)
    log_file = join(logs_dir, 'training.log')

    # Clear any existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")