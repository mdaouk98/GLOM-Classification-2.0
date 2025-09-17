# src/main_utils/models_utils/bayesian_multihead_vision_transformer.py

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

class BayesianMultiHeadVisionTransformer(nn.Module):
    """
    A Vision Transformer wrapper with MC Dropout, optional feature return,
    and multiple classification heads on the CLS token.

    Args:
        pretrained_name (str):
            HF model name, e.g. "google/vit-base-patch16-224-in21k"
        input_size (int):
            Height/width of the square input images (e.g. 224).
        heads (Dict[str,int]):
            Mapping from head name -> number of classes.
        dropout_p (float):
            Dropout probability for MC Dropout on the CLS embedding.
    """
    def __init__(
        self,
        input_size: int,
        heads: dict,  # e.g. {"class": 10, "aux": 4}
        dropout_p: float
    ):
        """
        Args:
            input_size (int): Size of the input images (assumes square).
            heads (Dict[str, int]): Mapping from head name ? number of classes.
            dropout_p (float): Dropout probability for MC Dropout.
        """
        super().__init__()

        # 1) Base transformer config (no classification head)
        self.config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            image_size=input_size,
            num_labels=1  # dummy, unused
        )

        # 2) Load just the feature extractor (no classifier head)
        self.backbone = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            config=self.config
        )

        # 3) Dropout for MC Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # 4) Multi-head classifier heads
        hidden_dim = self.config.hidden_size
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, out_dim)
            for name, out_dim in heads.items()
        })

    def forward(
        self,
        x: torch.Tensor,
        mc_dropout: bool = False,
        return_features: bool = False
    ) -> dict or tuple:
        """
        Forward pass through ViT with MC Dropout and multiple output heads.

        Returns:
            - outputs: Dict[str, Tensor] of logits per head
            - optionally features_pre_dropout (CLS token before dropout)
        """
        # 1) Run ViT backbone
        vit_outputs = self.backbone(pixel_values=x)
        cls_token = vit_outputs.last_hidden_state[:, 0]  # [CLS] token

        # 2) Save pre-dropout CLS token
        features_pre_dropout = cls_token.clone()

        # 3) Apply MC Dropout
        if mc_dropout:
            cls_token = self.dropout(cls_token)

        # 4) Compute logits for each head
        outputs = {
            name: head(cls_token) for name, head in self.heads.items()
        }

        if return_features:
            return outputs, features_pre_dropout
        return outputs