# src/main_utils/models_utils/get_bayesian_model.py

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
    densenet121, DenseNet121_Weights,
    densenet169, DenseNet169_Weights,
    densenet201, DenseNet201_Weights
)

from main_utils.models_utils import BayesianVisionTransformer, BayesianModel

def get_bayesian_model(
    model_name: str,
    num_classes: int = 2,
    input_size: int = 224,
    use_checkpointing: bool = False,
    dropout_p: float = 0.5,
    device: Optional[torch.device] = None
) -> nn.Module:

    """
    Factory function to create a Bayesian model with an optional gradient checkpointing setting.
    The returned model's forward function now accepts an additional parameter, return_features.
    When set to True, calling forward will return a tuple (logits, features) where features is the
    intermediate flattened feature vector extracted from the backbone.

    Args:
        model_name (str): Backbone name ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                          'densenet121', etc., or 'vision').
        num_classes (int): Number of output classes.
        input_size (int): Input image size (relevant for Vision models).
        use_checkpointing (bool): Whether to enable gradient checkpointing (for CNN backbones).
        dropout_p (float): Dropout probability.
        device (Optional[torch.device]): Device to move the model to.

    Returns:
        nn.Module: The constructed Bayesian model.
    """
    model_map = {
        'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
        'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
        'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
        'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V1, 2048),
        'resnet152': (resnet152, ResNet152_Weights.IMAGENET1K_V1, 2048),
        'densenet121': (densenet121, DenseNet121_Weights.IMAGENET1K_V1, 1024),
        'densenet169': (densenet169, DenseNet169_Weights.IMAGENET1K_V1, 1664),
        'densenet201': (densenet201, DenseNet201_Weights.IMAGENET1K_V1, 1920),
    }

    model_name_lower = model_name.lower()

    if model_name_lower == 'vision':
        if use_checkpointing:
            print("Warning: Gradient checkpointing is not explicitly implemented for Vision Transformer. "
                  "Consider using the model's built-in checkpointing via the transformers configuration.")
        model = BayesianVisionTransformer(
            num_classes=num_classes,
            input_size=input_size,
            dropout_p=dropout_p
        )
    elif model_name_lower in model_map:
        base_model_fn, weights, feature_dim = model_map[model_name_lower]
        base_model = base_model_fn(weights=weights)

        # Replace the final classification layer with Identity so that the backbone returns features.
        if 'resnet' in model_name_lower:
            base_model.fc = nn.Identity()
        elif 'densenet' in model_name_lower:
            base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone for BayesianModel: {model_name}")

        model = BayesianModel(
            base_model=base_model,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout_p=dropout_p
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if device:
        model.to(device)

    return model
