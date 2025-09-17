# src/main_utils/models_utils/bayesian_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

class BayesianModel(nn.Module):
    """
    A wrapper model that adds MC Dropout, gradient checkpointing, and a final
    classification head on top of a backbone network.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout_p: float
    ):
        """
        Args:
            base_model (nn.Module): Pretrained backbone (e.g., ResNet, DenseNet).
            feature_dim (int): Size of the feature vector output by the backbone.
            num_classes (int): Number of target classes.
            dropout_p (float): Dropout probability for MC Dropout.
        """
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(feature_dim, num_classes)

        # Detect backbone type for selective checkpointing
        self.is_resnet = hasattr(self.base, 'layer1')
        self.is_densenet = hasattr(self.base, 'features')

    def forward(
        self,
        x: torch.Tensor,
        mc_dropout: bool = False,
        use_checkpointing: bool = False,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input images, shape (B, C, H, W).
            mc_dropout (bool): If True, apply Dropout at inference for uncertainty.
            use_checkpointing (bool): If True and in training mode, use
                                      torch.utils.checkpoint to save memory.
            return_features (bool): If True, return (logits, features_before_dropout).

        Returns:
            logits (Tensor) or (logits, features) tuple.
        """
        # --------------------------------------------------
        # 1) Extract features, with optional checkpointing
        # --------------------------------------------------
        if use_checkpointing and self.training:
            if self.is_resnet:
                # Manually run early ResNet layers, checkpoint each block
                x = self.base.conv1(x)
                x = self.base.bn1(x)
                x = self.base.relu(x)
                x = self.base.maxpool(x)
                x = checkpoint(self.base.layer1, x)
                x = checkpoint(self.base.layer2, x)
                x = checkpoint(self.base.layer3, x)
                x = checkpoint(self.base.layer4, x)
                x = self.base.avgpool(x)
                features = torch.flatten(x, 1)

            elif self.is_densenet:
                # Checkpoint the entire DenseNet feature extractor
                feat_maps = checkpoint(self.base.features, x)
                feat_maps = F.relu(feat_maps, inplace=True)
                pooled = F.adaptive_avg_pool2d(feat_maps, (1, 1))
                features = torch.flatten(pooled, 1)

            else:
                # Fallback: checkpoint the entire backbone
                features = checkpoint(self.base, x)
        else:
            # Simple forward through backbone
            features = self.base(x)

        # --------------------------------------------------
        # 2) Preserve pre-dropout features if requested
        # --------------------------------------------------
        features_pre_dropout = features

        # --------------------------------------------------
        # 3) Apply MC Dropout at inference
        # --------------------------------------------------
        if mc_dropout:
            features = self.dropout(features)

        # --------------------------------------------------
        # 4) Classification head
        # --------------------------------------------------
        logits = self.fc(features)

        # --------------------------------------------------
        # 5) Return either logits alone or with features
        # --------------------------------------------------
        if return_features:
            return logits, features_pre_dropout
        return logits

