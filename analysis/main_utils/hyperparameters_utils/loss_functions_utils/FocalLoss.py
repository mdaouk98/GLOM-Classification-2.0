# src/main_utils/hyperparameters_utils/loss_functions_utils/FocalLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(
        self,
        alpha=None,
        gamma: float = 2,
        reduction: str = 'mean',
        weight: torch.Tensor = None
    ):
        """
        Args:
            alpha (float or list, optional): Class-wise weighting factor. If a list/tuple,
                should have length = num_classes; if a float, applies equally to all.
            gamma (float): Focusing parameter; higher gamma reduces loss contribution from
                well-classified examples.
            reduction (str): 'none' | 'mean' | 'sum'-how to aggregate the batch losses.
            weight (Tensor, optional): Manual rescaling weight per class (as in cross_entropy).
        """
        super().__init__()
        # Store/convert alpha to tensor if needed
        self.alpha = alpha
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs (Tensor): Logits of shape (batch_size, num_classes).
            targets (Tensor): Ground-truth labels of shape (batch_size,).

        Returns:
            Tensor: Scalar loss if reduction is 'mean' or 'sum', else per-sample losses.
        """
        # 1) Compute standard cross-entropy loss (per sample, no reduction)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction='none'
        )

        # 2) Convert CE loss to probability of true class: pt = exp(–CE)
        pt = torch.exp(-ce_loss)

        # 3) Apply alpha balancing if provided
        if self.alpha is not None:
            # Ensure alpha tensor is same type/device as inputs
            if isinstance(self.alpha, torch.Tensor) and self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Gather the weight for each sample's true class
            alpha_t = (
                self.alpha[targets] if isinstance(self.alpha, torch.Tensor)
                else torch.tensor(self.alpha, device=inputs.device)
            )
            ce_loss = ce_loss * alpha_t

        # 4) Apply the focal modulation factor: (1 – pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 5) Reduce loss as specified
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


