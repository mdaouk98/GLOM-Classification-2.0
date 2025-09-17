# src/main_utils/hyperparameters_utils/loss_functions_utils/TotalCrossEntropyLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalCrossEntropyLoss(nn.Module):
    """
    Total Cross Entropy Loss: a weighted sum of standard Cross Entropy Loss
    and Reverse Cross Entropy Loss to balance accuracy with prediction uncertainty.
    """
    def __init__(
        self,
        ce_loss: nn.Module,
        rce_loss: nn.Module,
        ce_weight: float = 1.0,
        rce_weight: float = 1.0
    ):
        """
        Initialize the combined loss.

        Args:
            ce_loss (nn.Module): Cross Entropy Loss instance (e.g., nn.CrossEntropyLoss()).
            rce_loss (nn.Module): Reverse Cross Entropy Loss instance.
            ce_weight (float): Scaling factor for the CE component.
            rce_weight (float): Scaling factor for the RCE component.
        """
        super().__init__()
        # Store the two loss modules and their relative weights
        self.ce_loss = ce_loss
        self.rce_loss = rce_loss
        self.ce_weight = ce_weight
        self.rce_weight = rce_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the total loss for a batch.

        Steps:
          1) Compute standard cross entropy: CE(input, target)
          2) Compute reverse cross entropy: RCE(input)
          3) Combine: ce_weight * CE + rce_weight * RCE

        Args:
            inputs (Tensor): Logits of shape (batch_size, num_classes).
            targets (Tensor): Ground-truth labels of shape (batch_size,).

        Returns:
            Tensor: Scalar or per-sample loss, depending on the reduction settings
                    of the underlying ce_loss and rce_loss modules.
        """
        # 1) Cross entropy component
        ce_value = self.ce_loss(inputs, targets)
        
        # 2) Reverse cross entropy component
        rce_value = self.rce_loss(inputs)
        
        # 3) Weighted sum of both components
        total_loss = self.ce_weight * ce_value + self.rce_weight * rce_value
        
        return total_loss
