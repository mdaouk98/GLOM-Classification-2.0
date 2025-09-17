# src/train_utils/setup_transforms_and_processor.py

import logging
from augmentations import get_augmentation_pipeline
from typing import Any, Dict, List, Optional, Tuple

def setup_transforms_and_processor(config: Any) -> Tuple[Optional[Any], Optional[Any], Optional[Any], bool]:
    """
    Prepare data augmentation pipelines and vision processor if required.

    Args:
        config (Any): Configuration object with augmentation and model details.

    Returns:
        Tuple containing:
          - transform_for_train: Transformation pipeline for training.
          - transform_for_val: Transformation pipeline for validation.
          - processor: Vision processor if needed, else None.
          - for_vision (bool): Flag indicating if a vision model is used.
    """
    augmentation_type = config.augmentation.type.lower()
    if augmentation_type not in ['mixup', 'cutmix']:
        transform_for_train = get_augmentation_pipeline(augmentation_type, config.model.input_size)
    else:
        transform_for_train = get_augmentation_pipeline('basic', config.model.input_size)
    transform_for_val = get_augmentation_pipeline('none', config.model.input_size)

    processor = None
    for_vision = False
    if config.model.name.lower() == 'vision':
        try:
            from transformers import ViTImageProcessor
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            for_vision = True
            # For vision models, rely on the processor instead of augmentation transforms.
            transform_for_train = None
            transform_for_val = None
            logging.info("[Transforms] Vision model detected, using processor instead of custom transforms.")
        except Exception as e:
            logging.error(f"[Transforms] Failed to load ViTImageProcessor: {e}")
            raise

    return transform_for_train, transform_for_val, processor, for_vision