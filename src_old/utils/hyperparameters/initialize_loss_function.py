# src/utils/hyperparamters/initialize_loss_function.py

import torch
from typing import Any
from torch import device
from torch import nn
from utils.hyperparameters.losses import FocalLoss,ReverseCrossEntropyLoss,TotalCrossEntropyLoss

def initialize_loss_function(
    config: Any,
    device: device
) -> nn.Module:
    """
    Initialize the loss function based on the configuration.
    """
    loss_function_name = config.loss_function.lower()
    loss_parameters = getattr(config, 'loss_parameters', {})

    if loss_function_name == 'crossentropyloss':
        criterion_weight = config.misc.criterion_weight
        if criterion_weight == 'None':
            criterion = nn.CrossEntropyLoss()
        elif criterion_weight == 'equal_weight':
            class_weights = torch.tensor([1.0, 7767 / 1907], dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif criterion_weight == 'weight10':
            class_weights = torch.tensor([1.0, 10.0], dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported criterion_weight option: {criterion_weight}")

    elif loss_function_name == 'focalloss':
        focal_params = loss_parameters.get('focal_loss', {})
        alpha = focal_params.get('alpha', None)
        gamma = focal_params.get('gamma', 2.0)
        ce_weight = config.misc.criterion_weight

        if ce_weight == 'None':
            ce_weight_tensor = None
        elif ce_weight == 'equal_weight':
            ce_weight_tensor = torch.tensor([1.0, 7767 / 1907], dtype=torch.float).to(device)
        elif ce_weight == 'weight10':
            ce_weight_tensor = torch.tensor([1.0, 10.0], dtype=torch.float).to(device)
        else:
            raise ValueError(f"Unsupported criterion_weight option for FocalLoss: {ce_weight}")

        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean', weight=ce_weight_tensor)

    elif loss_function_name == 'reversecrossentropyloss':
        criterion_weight = config.misc.criterion_weight
        if criterion_weight == 'None':
            rce_weight_tensor = None
        elif criterion_weight == 'equal_weight':
            rce_weight_tensor = torch.tensor([1.0, 7767 / 1907], dtype=torch.float).to(device)
        elif criterion_weight == 'weight10':
            rce_weight_tensor = torch.tensor([1.0, 10.0], dtype=torch.float).to(device)
        else:
            raise ValueError(f"Unsupported criterion_weight option for ReverseCrossEntropyLoss: {criterion_weight}")

        criterion = ReverseCrossEntropyLoss(reduction='mean', weight=rce_weight_tensor)

    elif loss_function_name == 'totalcrossentropyloss':
        # Initialize Cross Entropy Loss
        ce_weight = config.misc.criterion_weight
        if ce_weight == 'None':
            ce_weight_tensor = None
        elif ce_weight == 'equal_weight':
            ce_weight_tensor = torch.tensor([1.0, 7767 / 1907], dtype=torch.float).to(device)
        elif ce_weight == 'weight10':
            ce_weight_tensor = torch.tensor([1.0, 10.0], dtype=torch.float).to(device)
        else:
            raise ValueError(f"Unsupported criterion_weight option for TotalCrossEntropyLoss: {ce_weight}")

        ce_loss = nn.CrossEntropyLoss(weight=ce_weight_tensor)

        # Initialize Reverse Cross Entropy Loss
        rce_weight_tensor = ce_weight_tensor  # Use the same weights for RCE
        rce_loss = ReverseCrossEntropyLoss(reduction='mean', weight=rce_weight_tensor)

        # Initialize Total Cross Entropy Loss with weights for CE and RCE
        total_ce_params = loss_parameters.get('total_ce_loss', {})
        ce_weight_param = total_ce_params.get('ce_weight', 1.0)
        rce_weight_param = total_ce_params.get('rce_weight', 1.0)

        criterion = TotalCrossEntropyLoss(
            ce_loss=ce_loss,
            rce_loss=rce_loss,
            ce_weight=ce_weight_param,
            rce_weight=rce_weight_param
        )

    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function}")

    return criterion