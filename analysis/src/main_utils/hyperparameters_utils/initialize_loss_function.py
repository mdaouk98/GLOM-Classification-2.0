# src/main_utils/hyperparamters_utils/initialize_loss_function.py

import torch
from typing import Any
from torch import nn, device
from main_utils.hyperparameters_utils.loss_functions_utils import (
    FocalLoss,
    ReverseCrossEntropyLoss,
    TotalCrossEntropyLoss,
)

def initialize_loss_function(
    config: Any,
    device: device
) -> nn.Module:
    """
    Initialize and return a loss function based on the provided configuration.

    Supported loss_function options in config.loss_function:
      - 'CrossEntropyLoss'
      - 'FocalLoss'
      - 'ReverseCrossEntropyLoss'
      - 'TotalCrossEntropyLoss'

    For weighted variants, config.misc.criterion_weight may be:
      - 'None'         ? no class weighting
      - 'equal_weight' ? weights = [1.0, majority/minority]
      - 'weight10'     ? weights = [1.0, 10.0]

    Additional parameters for FocalLoss and TotalCrossEntropyLoss can be
    provided via config.loss_parameters.
    """
    def _get_weight_tensor(option: str):
        """
        Helper to build a class-weight tensor on the given device.
        """
        if option == 'None':
            return None
        elif option == 'equal_weight':
            # balance: majority/minority
            weights = [1.0, 7767 / 1907]
        elif option == 'weight10':
            weights = [1.0, 10.0]
        else:
            raise ValueError(f"Unsupported criterion_weight option: {option!r}")
        return torch.tensor(weights, dtype=torch.float, device=device)

    loss_name = config.loss_function.lower()
    loss_params = getattr(config, 'loss_parameters', {})
    weight_option = config.misc.criterion_weight

    # -------------------------------
    # 1) Standard CrossEntropyLoss
    # -------------------------------
    if loss_name == 'crossentropyloss':
        weight_tensor = _get_weight_tensor(weight_option)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # -------------------------------
    # 2) FocalLoss
    # -------------------------------
    elif loss_name == 'focalloss':
        # Extract focal-specific params, with defaults
        focal_cfg = loss_params.get('focal_loss', {})
        alpha = focal_cfg.get('alpha', None)
        gamma = focal_cfg.get('gamma', 2.0)

        # Build optional CE weight tensor
        ce_weight = _get_weight_tensor(weight_option)

        criterion = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='mean',
            weight=ce_weight
        )

    # -------------------------------
    # 3) ReverseCrossEntropyLoss
    # -------------------------------
    elif loss_name == 'reversecrossentropyloss':
        rce_weight = _get_weight_tensor(weight_option)
        criterion = ReverseCrossEntropyLoss(
            reduction='mean',
            weight=rce_weight
        )

    # -------------------------------
    # 4) TotalCrossEntropyLoss
    # -------------------------------
    elif loss_name == 'totalcrossentropyloss':
        # (a) CrossEntropy component
        ce_weight = _get_weight_tensor(weight_option)
        ce_loss = nn.CrossEntropyLoss(weight=ce_weight)

        # (b) ReverseCrossEntropy component (reuse same weight)
        rce_loss = ReverseCrossEntropyLoss(
            reduction='mean',
            weight=ce_weight
        )

        # (c) TotalCE parameters (mixing coefficients)
        total_cfg = loss_params.get('total_ce_loss', {})
        ce_w = total_cfg.get('ce_weight', 1.0)
        rce_w = total_cfg.get('rce_weight', 1.0)

        criterion = TotalCrossEntropyLoss(
            ce_loss=ce_loss,
            rce_loss=rce_loss,
            ce_weight=ce_w,
            rce_weight=rce_w
        )

    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function!r}")

    return criterion
