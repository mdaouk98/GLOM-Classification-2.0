# src/main_utils/hyperparameters_utils/initialize_scheduler.py


from typing import Any
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup

def initialize_scheduler(
    config: Any,
    optimizer: Optimizer,
    train_loader: DataLoader
) -> _LRScheduler:
    """
    Create and return a learning rate scheduler based on configuration.

    Supported schedulers (config.scheduler.type):
      - 'cosine_warm_up': Cosine decay with linear warmup.
      - 'reduce_on_plateau': Reduce LR when a monitored metric plateaus.

    Args:
        config:       Configuration object with scheduler settings.
        optimizer:    The optimizer whose LR will be scheduled.
        train_loader: DataLoader for training, used to infer total steps.

    Returns:
        An instance of a torch.optim.lr_scheduler or transformers scheduler.
    """
    sched_type = config.scheduler.type.lower()

    # ----------------------------------------
    # 1) Cosine schedule with warmup
    # ----------------------------------------
    if sched_type == 'cosine_warm_up':
        # Total training steps = epochs x iterations per epoch
        total_steps = len(train_loader) * config.training.epochs
        # Warmup steps as a fraction of total
        num_warmup_steps = int(config.scheduler.warmup_ratio * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

    # ----------------------------------------
    # 2) Reduce LR on plateau
    # ----------------------------------------
    elif sched_type == 'reduce_on_plateau':
        _args = dict(mode='min',                      # lower metric is better
            factor=config.scheduler.factor, # LR multiplier on plateau
            patience=config.scheduler.patience)
        try:
            # if this version of PyTorch supports verbose, use it
            from inspect import signature
            if 'verbose' in signature(ReduceLROnPlateau).parameters:
                _args['verbose'] = True
        except ImportError:
            pass
    
        scheduler = ReduceLROnPlateau(optimizer, **_args)

    # ----------------------------------------
    # 3) Unsupported scheduler type
    # ----------------------------------------
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler.type!r}")

    return scheduler
