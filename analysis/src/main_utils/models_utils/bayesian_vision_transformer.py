# src/main_utils/models_utils/bayesian_vision_transformer.py

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class BayesianVisionTransformer(nn.Module):
    """
    A Vision Transformer wrapper that adds Monte Carlo Dropout and optional
    feature extraction on top of a pretrained ViT backbone.
    """
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        dropout_p: float
    ):
        """
        Args:
            num_classes (int): Number of output classes.
            input_size (int): Height/width of input images (ViT expects square inputs).
            dropout_p (float): Dropout probability for MC sampling.
        """
        super().__init__()

        # 1) Load ViT configuration with the desired image size & number of labels
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            image_size=input_size
        )

        # 2) Load pretrained ViT model, allowing size mismatches for classifier head
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            config=config,
            ignore_mismatched_sizes=True
        )

        # 3) Dropout layer applied to the [CLS] token for MC Dropout
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        mc_dropout: bool = False,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optional Monte Carlo Dropout and feature return.

        Args:
            x (Tensor): Input batch of images, shape (B, C, H, W).
            mc_dropout (bool): If True, apply dropout to the CLS token at inference.
            return_features (bool): If True, also return the pre-dropout CLS features.

        Returns:
            - logits: Tensor of shape (B, num_classes)
            - optionally, features_pre_dropout: Tensor of shape (B, hidden_dim)
        """
        # 1) Run the transformer layers (without the classification head)
        outputs = self.model.vit(x)

        # 2) Extract the [CLS] token representation (first token)
        cls_token = outputs.last_hidden_state[:, 0]  # shape: (B, hidden_dim)

        # 3) Keep a copy of raw features before dropout
        features_pre_dropout = cls_token.clone()

        # 4) Apply MC Dropout if requested
        if mc_dropout:
            cls_token = self.dropout(cls_token)

        # 5) Run the classification head on (possibly dropped-out) CLS token
        logits = self.model.classifier(cls_token)

        # 6) Return logits (and features if requested)
        if return_features:
            return logits, features_pre_dropout
        return logits

