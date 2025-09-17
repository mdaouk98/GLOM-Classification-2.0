# src/main_utils/models_utils/bayesian_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

class BayesianModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout_p: float
    ):
        """
        Initializes the Bayesian model with a given backbone.

        Args:
            base_model (nn.Module): Backbone model (e.g. ResNet or DenseNet).
            feature_dim (int): Dimensionality of features output by the backbone.
            num_classes (int): Number of output classes.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(feature_dim, num_classes)
        # Determine backbone type for selective checkpointing
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
        Forward pass through the Bayesian model with optional MC Dropout, selective
        gradient checkpointing, and optional feature extraction.

        Args:
            x (torch.Tensor): Input tensor.
            mc_dropout (bool): Whether to apply dropout for MC sampling.
            use_checkpointing (bool): Whether to use gradient checkpointing.
            return_features (bool): If True, return a tuple (logits, features) where features
                                    is the intermediate flattened feature vector.
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Logits or a tuple (logits, features).
        """
        if use_checkpointing and self.training:
            # Selective checkpointing: for known architectures, we recompute segments only
            if self.is_resnet:
                x = self.base.conv1(x)
                x = self.base.bn1(x)
                x = self.base.relu(x)
                x = self.base.maxpool(x)
                # Checkpoint each of the four residual blocks:
                x = checkpoint(self.base.layer1, x)
                x = checkpoint(self.base.layer2, x)
                x = checkpoint(self.base.layer3, x)
                x = checkpoint(self.base.layer4, x)
                x = self.base.avgpool(x)
                features = torch.flatten(x, 1)
            elif self.is_densenet:
                features = checkpoint(self.base.features, x)
                features = F.relu(features, inplace=True)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
            else:
                features = checkpoint(self.base, x)
        else:
            features = self.base(x)
        
        # Save the features before applying dropout
        features_pre_dropout = features

        if mc_dropout:
            features = self.dropout(features)
        
        logits = self.fc(features)
        
        if return_features:
            return logits, features_pre_dropout
        else:
            return logits
