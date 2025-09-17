# src/utils/hyperparameters/initialize_scheduler.py


from typing import Any
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup

def initialize_scheduler(
    config: Any,
    optimizer: Optimizer,
    train_loader: DataLoader
) -> _LRScheduler:
    """
    Initialize the learning rate scheduler based on the configuration.
    """
    scheduler_type = config.scheduler.type.lower()

    if scheduler_type == 'cosine_warm_up':
        total_steps = len(train_loader) * config.training.epochs
        num_warmup_steps = int(config.scheduler.warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler.type}")

    return scheduler