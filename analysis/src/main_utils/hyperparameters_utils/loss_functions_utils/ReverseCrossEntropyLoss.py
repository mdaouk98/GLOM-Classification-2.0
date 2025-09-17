# src/main_utils/hyperparameters_utils/loss_functions_utils/ReverseCrossEntropyLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseCrossEntropyLoss(nn.Module):
    """
    Reverse Cross Entropy Loss encourages higher entropy in the predictions.
    Acts as a regularizer to prevent the model from becoming overconfident.
    Reference: Encouraging prediction distributions with larger entropy.
    """
    def __init__(
        self,
        reduction: str = 'mean',
        weight: torch.Tensor = None
    ):
        """
        Args:
            reduction (str): 'none' | 'mean' | 'sum' - how to reduce per-sample losses.
            weight (Tensor, optional): Class-wise weights (not used directly here but
                kept for interface consistency with cross-entropy variants).
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight  # Placeholder for compatibility; not applied in RCE

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Reverse Cross Entropy (negative entropy) loss.

        Args:
            inputs (Tensor): Logits with shape (batch_size, num_classes).

        Returns:
            Tensor: The per-sample or aggregated RCE loss.
        """
        # 1) Convert logits to log-probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # 2) Recover probabilities from log-probs
        probs = torch.exp(log_probs)

        # 3) Compute negative entropy per sample: -sum(p * log p)
        per_sample_rce = -torch.sum(probs * log_probs, dim=1)

        # 4) Apply desired reduction
        if self.reduction == 'mean':
            return per_sample_rce.mean()
        elif self.reduction == 'sum':
            return per_sample_rce.sum()
        else:  # 'none'
            return per_sample_rce


