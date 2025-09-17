# src/train_utils/setup_transforms_and_processor.py

import logging
from typing import Any, Optional, Tuple
from main_utils.augmentations_utils import get_augmentation_pipeline

def setup_transforms_and_processor(config: Any) -> Tuple[Optional[Any], Optional[Any], Optional[Any], bool]:
    """
    Prepare augmentation pipelines and a vision processor based on the model configuration.

    Returns:
        transform_for_train (callable or None): Augmentation pipeline for training.
        transform_for_val   (callable or None): Pipeline for validation (usually just resize/normalize).
        processor           (object or None):    HuggingFace vision processor for ViT models.
        for_vision          (bool):              True if a vision transformer is used.
    """
    # --- 1) Determine basic augmentation pipelines ---
    aug_type = config.augmentation.type.lower()

    # Mixup/CutMix pipelines are applied at training time separately, so use 'basic' here
    train_type = 'basic' if aug_type in ['mixup', 'cutmix'] else aug_type
    transform_for_train = get_augmentation_pipeline(train_type, config.model.input_size)

    # For validation, apply no augmentation—just resize + normalize
    transform_for_val = get_augmentation_pipeline('none', config.model.input_size)

    # --- 2) Check if we're using a Vision Transformer ---
    processor = None
    for_vision = False

    if config.model.name.lower() == 'vision':
        try:
            # Use HuggingFace processor to handle resizing, normalization, etc.
            from transformers import ViTImageProcessor
            processor = ViTImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k"
            )
            for_vision = True

            # Disable custom transforms when using the processor
            transform_for_train = None
            transform_for_val = None

            logging.info(
                "[Transforms] Vision model detected ? "
                "using ViTImageProcessor for preprocessing."
            )
        except Exception as e:
            logging.error(f"[Transforms] Failed to load ViTImageProcessor: {e}", exc_info=True)
            raise

    return transform_for_train, transform_for_val, processor, for_vision

