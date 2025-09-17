# src/main_utils/models_utils/bayesian_multihead_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict

class BayesianMultiHeadModel(nn.Module):
    """
    A multi-head Bayesian model wrapper that adds MC Dropout, gradient checkpointing,
    and separate classification heads on top of a backbone network.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        heads: Dict[str, int],
        dropout_p: float
    ):
        """
        Args:
            base_model (nn.Module): Pretrained backbone (e.g., ResNet, DenseNet).
            feature_dim (int): Size of the feature vector output by the backbone.
            heads (Dict[str,int]): Mapping from head name to its output dimension.
            dropout_p (float): Dropout probability for MC sampling.
        """
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(p=dropout_p)

        # Create one linear layer per head/task
        self.heads = nn.ModuleDict({
            name: nn.Linear(feature_dim, dim)
            for name, dim in heads.items()
        })

        # Flags to detect backbone architecture for checkpointing
        self.is_resnet = hasattr(self.base, 'layer1')
        self.is_densenet = hasattr(self.base, 'features')

    def forward(
        self,
        x: torch.Tensor,
        mc_dropout: bool = False,
        use_checkpointing: bool = False,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor] or Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass with optional MC Dropout, gradient checkpointing, and feature return.

        Args:
            x (Tensor): Input batch, shape (B, C, H, W).
            mc_dropout (bool): If True, apply dropout at inference for uncertainty.
            use_checkpointing (bool): If True and in train mode, use torch.checkpoint
                                      to save memory by recomputing activations.
            return_features (bool): If True, also return the pre-dropout features.

        Returns:
            If return_features:
                (outputs_dict, features_pre_dropout)
            else:
                outputs_dict
            where outputs_dict maps head name -> logits tensor.
        """
        # --------------------------------------------------
        # 1) Feature extraction with optional checkpointing
        # --------------------------------------------------
        if use_checkpointing and self.training:
            if self.is_resnet:
                # Manually run early ResNet layers, checkpoint each residual block
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
                # Fallback: checkpoint the whole backbone
                features = checkpoint(self.base, x)
        else:
            # Standard forward through backbone
            features = self.base(x)

        # --------------------------------------------------
        # 2) Preserve pre-dropout features if requested
        # --------------------------------------------------
        features_pre_dropout = features

        # --------------------------------------------------
        # 3) Apply MC Dropout at inference time
        # --------------------------------------------------
        if mc_dropout:
            features = self.dropout(features)

        # --------------------------------------------------
        # 4) Compute outputs for each head
        # --------------------------------------------------
        outputs: Dict[str, torch.Tensor] = {}
        for name, head_layer in self.heads.items():
            outputs[name] = head_layer(features)

        # --------------------------------------------------
        # 5) Return outputs (and features if asked)
        # --------------------------------------------------
        if return_features:
            return outputs, features_pre_dropout
        return outputs
