# src/main_utils/models_utils/bayesian_vision_transformer.py

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class BayesianVisionTransformer(nn.Module):
    def __init__(self, num_classes: int, input_size: int, dropout_p: float):
        """
        Initializes the Bayesian Vision Transformer model.

        Note:
            For the ViT backbone, consider using the built-in gradient checkpointing
            available via the transformers library (e.g., by setting
            `self.model.config.gradient_checkpointing = True`).
        """
        super().__init__()
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            image_size=input_size
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            config=config,
            ignore_mismatched_sizes=True
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer with optional Monte Carlo Dropout and
        optional feature extraction.

        Args:
            x (torch.Tensor): Input tensor.
            mc_dropout (bool): Whether to apply dropout for MC sampling.
            return_features (bool): If True, return a tuple (logits, features) where features
                                    is the intermediate representation (CLS token) before dropout.
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Logits or a tuple (logits, features).
        """
        outputs = self.model.vit(x)  # Run the ViT backbone
        # Extract the CLS token as the feature representation
        cls_token = outputs.last_hidden_state[:, 0]
        features_pre_dropout = cls_token.clone()  # Store the feature vector before dropout
        
        if mc_dropout:
            cls_token = self.dropout(cls_token)
        
        logits = self.model.classifier(cls_token)
        
        if return_features:
            return logits, features_pre_dropout
        else:
            return logits
