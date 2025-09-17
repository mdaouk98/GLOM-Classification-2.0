# src/main_utils/hyperparameters_utils/loss_functions_utils/TotalCrossEntropyLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalCrossEntropyLoss(nn.Module):
    """
    Total Cross Entropy Loss combines Cross Entropy Loss with Reverse Cross Entropy Loss.
    It balances accuracy with prediction uncertainty.
    """
    def __init__(self, ce_loss, rce_loss, ce_weight=1.0, rce_weight=1.0):
        """
        Args:
            ce_loss (nn.Module): An instance of Cross Entropy Loss.
            rce_loss (nn.Module): An instance of Reverse Cross Entropy Loss.
            ce_weight (float): Weight for Cross Entropy Loss.
            rce_weight (float): Weight for Reverse Cross Entropy Loss.
        """
        super(TotalCrossEntropyLoss, self).__init__()
        self.ce_loss = ce_loss
        self.rce_loss = rce_loss
        self.ce_weight = ce_weight
        self.rce_weight = rce_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) with shape (batch_size, num_classes).
            targets: Ground truth labels with shape (batch_size).
        Returns:
            Computed Total Cross Entropy Loss.
        """
        ce = self.ce_loss(inputs, targets)
        rce = self.rce_loss(inputs)
        total_loss = self.ce_weight * ce + self.rce_weight * rce
        return total_loss
