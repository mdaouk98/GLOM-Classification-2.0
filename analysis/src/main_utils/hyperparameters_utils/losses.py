# src/losses.py
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
