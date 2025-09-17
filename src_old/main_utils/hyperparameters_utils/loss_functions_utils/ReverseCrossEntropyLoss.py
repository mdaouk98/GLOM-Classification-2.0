# src/main_utils/hyperparameters_utils/loss_functions_utils/ReverseCrossEntropyLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReverseCrossEntropyLoss(nn.Module):
    """
    Reverse Cross Entropy Loss encourages higher entropy in the predictions.
    It acts as a regularizer to prevent the model from being overconfident.
    """
    def __init__(self, reduction='mean', weight=None):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            weight (Tensor, optional): A manual rescaling weight given to each class.
        """
        super(ReverseCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs):
        """
        Args:
            inputs: Predictions (logits) with shape (batch_size, num_classes).
        Returns:
            Computed Reverse Cross Entropy Loss.
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        reverse_ce = -torch.sum(probs * log_probs, dim=1)  # Entropy

        if self.reduction == 'mean':
            return reverse_ce.mean()
        elif self.reduction == 'sum':
            return reverse_ce.sum()
        else:
            return reverse_ce

