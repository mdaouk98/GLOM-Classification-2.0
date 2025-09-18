# src/main_utils/hyperparameters_utils/initialize_optimizer.py

from typing import Any
from torch import device
from torch.nn import Module
import torch.optim as optim

def initialize_optimizer(
    config: Any,
    model: Module,
    device: device
) -> optim.Optimizer:
    """
    Create and return an optimizer based on the training configuration.

    Supported optimizer types (config.optimizer.type):
      - 'adam'
      - 'adamw'

    For AdamW with a Vision Transformer model (config.model.name == 'vision'),
    applies layer-wise learning rates:
      * Embedding layers:   lr * 0.1
      * Transformer encoder: lr
      * Classifier head:     lr * 10

    Args:
        config: Configuration object with optimizer and model settings.
        model:  The neural network whose parameters will be optimized.
        device: Target device (unused here, but kept for interface consistency).

    Returns:
        An instantiated torch.optim.Optimizer.
    """
    # Extract optimizer settings
    opt_type = config.optimizer.type.lower()
    lr = config.optimizer.learning_rate

    # -------------------------------
    # 1) Standard Adam optimizer
    # -------------------------------
    if opt_type == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr
        )

    # -------------------------------
    # 2) AdamW optimizer
    # -------------------------------
    elif opt_type == 'adamw':
        # Special case: Vision Transformer with layer-wise LRs
        if config.model.name.lower() == 'vision':
            optimizer = optim.AdamW([
                {
                    'params': model.model.vit.embeddings.parameters(),
                    'lr': lr * 0.1
                },
                {
                    'params': model.model.vit.encoder.parameters(),
                    'lr': lr
                },
                {
                    'params': model.model.classifier.parameters(),
                    'lr': lr * 10
                }
            ])
        # Default: apply same LR to all parameters
        else:
            optimizer = optim.AdamW(
                params=model.parameters(),
                lr=lr
            )

    # -------------------------------
    # 3) Unsupported optimizer
    # -------------------------------
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer.type!r}")

    return optimizer
