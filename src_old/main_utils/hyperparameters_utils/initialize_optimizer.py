# src/utils/hyperparameters/initialize_optimizer.py

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
    Initialize the optimizer based on the configuration.
    """
    optimizer_type = config.optimizer.type.lower()
    learning_rate = config.optimizer.learning_rate

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamw':
        if config.model.name.lower() == 'vision':
            optimizer = optim.AdamW([
                {'params': model.model.vit.embeddings.parameters(), 'lr': learning_rate * 0.1},
                {'params': model.model.vit.encoder.parameters(), 'lr': learning_rate},
                {'params': model.model.classifier.parameters(), 'lr': learning_rate * 10}
            ])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer.type}")

    return optimizer