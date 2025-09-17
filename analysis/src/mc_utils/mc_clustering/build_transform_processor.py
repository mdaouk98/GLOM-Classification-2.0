# src/mc_utils/mc_clustering/build_transform_processor.py

from typing import Any, Optional, Tuple
from main_utils.augmentations_utils import get_augmentation_pipeline

def build_transform_processor(config: Any) -> Tuple[Optional[Any], Optional[Any], bool]:
    """
    Build the appropriate transformation pipeline and image processor based on the model type.
    
    For vision models, it loads a pretrained ViTImageProcessor (from transformers) and
    sets transform to None. For other model types, it builds a custom augmentation pipeline.
    
    Args:
        config (Any): Configuration object with model and augmentation details.
        
    Returns:
        Tuple containing:
            - transform: Augmentation pipeline (or None for vision models)
            - processor: Image processor instance (or None for non-vision models)
            - for_vision: Boolean flag indicating if the model is a vision model.
    """
    if config.model.name.lower() == "vision":
        from transformers import ViTImageProcessor  # Import locally since it's only used for vision
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        transform = None  # No additional transforms needed for vision models
        for_vision = True
    else:
        processor = None
        for_vision = False
        transform = get_augmentation_pipeline(config.augmentation.type.lower(), config.model.input_size)
    return transform, processor, for_vision
