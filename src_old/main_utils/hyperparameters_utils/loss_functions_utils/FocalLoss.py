# src/main_utils/hyperparameters_utils/loss_functions_utils/FocalLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', weight=None):
        """
        Args:
            alpha (float or list, optional): Weighting factor for classes. Can be a single float or a list of floats.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            weight (Tensor, optional): A manual rescaling weight given to each class.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) with shape (batch_size, num_classes).
            targets: Ground truth labels with shape (batch_size).
        Returns:
            Computed Focal Loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            alpha = self.alpha[targets]
            ce_loss = ce_loss * alpha
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

