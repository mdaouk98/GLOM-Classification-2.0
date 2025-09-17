# src/main_utils/models_utils/get_bayesian_model.py

import torch
import timm
import torch.nn as nn
from typing import Optional, Dict
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
    densenet121, DenseNet121_Weights,
    densenet169, DenseNet169_Weights,
    densenet201, DenseNet201_Weights
)

from main_utils.models_utils import (
    BayesianVisionTransformer,
    BayesianModel,
    BayesianMultiHeadModel,
    BayesianMultiHeadVisionTransformer
)

def get_bayesian_model(
    model_name: str,
    heads: Dict[str, int],
    input_size: int = 224,
    use_checkpointing: bool = False,
    dropout_p: float = 0.5,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory for Bayesian models with various backbones and multi-head outputs.

    Args:
        model_name (str): One of:
            - 'vision' (Vision Transformer)
            - torchvision ResNets / DenseNets: 'resnet18', ..., 'densenet201'
            - timm models: 'densenet264d', 'efficientnet_b0', 'resnext50_32x4d', 'regnety_8gf', etc.
        heads (Dict[str,int]): Mapping head-name ? number of output classes for that head.
        input_size (int): Input image size (only used for ViT).
        use_checkpointing (bool): Enable gradient checkpointing for CNN backbones.
        dropout_p (float): Dropout probability for MC Dropout.
        device (torch.device, optional): Device to move the model to.

    Returns:
        nn.Module: A BayesianModel or BayesianMultiHeadModel instance.
    """

    # --- Predefined map for torchvision ResNets & DenseNets (weights + feature dims)
    model_map = {
        'resnet18':    (resnet18,    ResNet18_Weights.IMAGENET1K_V1,  512),
        'resnet34':    (resnet34,    ResNet34_Weights.IMAGENET1K_V1,  512),
        'resnet50':    (resnet50,    ResNet50_Weights.IMAGENET1K_V1, 2048),
        'resnet101':   (resnet101,   ResNet101_Weights.IMAGENET1K_V1, 2048),
        'resnet152':   (resnet152,   ResNet152_Weights.IMAGENET1K_V1, 2048),
        'densenet121': (densenet121, DenseNet121_Weights.IMAGENET1K_V1,1024),
        'densenet169': (densenet169, DenseNet169_Weights.IMAGENET1K_V1,1664),
        'densenet201': (densenet201, DenseNet201_Weights.IMAGENET1K_V1,1920),
    }

    name = model_name.lower()

    # ------------------------
    # 1) Vision Transformer
    # ------------------------
    if name == 'vision':
        if use_checkpointing:
            print("Warning: Use built-in ViT checkpointing via config.gradient_checkpointing=True.")
        model = BayesianVisionTransformer(
            num_classes=sum(heads.values()) if len(heads)==1 else next(iter(heads.values())),
            input_size=input_size,
            dropout_p=dropout_p
        )
        # Choose single-head vs. multi-head wrapper
        if len(heads) == 1:
            num_cls = next(iter(heads.values()))
            model = BayesianVisionTransformer(
                num_classes=sum(heads.values()) if len(heads)==1 else next(iter(heads.values())),
                input_size=input_size,
                dropout_p=dropout_p
            )
        else:
            model = BayesianMultiHeadVisionTransformer(
                input_size=input_size,
                heads=heads,
                dropout_p=dropout_p
                )

    # -----------------------------------------
    # 2) torchvision ResNet / DenseNet backbones
    # -----------------------------------------
    elif name in model_map:
        base_fn, weights, feat_dim = model_map[name]
        base = base_fn(weights=weights)

        # Remove the original classifier so backbone returns raw features
        if 'resnet' in name:
            base.fc = nn.Identity()
        else:  # densenet
            base.classifier = nn.Identity()

        # Choose single-head vs. multi-head wrapper
        if len(heads) == 1:
            num_cls = next(iter(heads.values()))
            model = BayesianModel(
                base_model=base,
                feature_dim=feat_dim,
                num_classes=num_cls,
                dropout_p=dropout_p
            )
        else:
            model = BayesianMultiHeadModel(
                base_model=base,
                feature_dim=feat_dim,
                heads=heads,
                dropout_p=dropout_p
            )

    # -----------------------------------------
    # 3) timm DenseNet-264 (“densenet264d”)
    # -----------------------------------------
    elif name == 'densenet264d':
        try:
            base = timm.create_model('densenet264d', pretrained=True)
        except RuntimeError as e:
            if 'No pretrained weights exist' in str(e):
                print("Warning: initializing densenet264d randomly (no pretrained weights).")
                base = timm.create_model('densenet264d', pretrained=False)
            else:
                raise

        # Remove classifier to get features
        in_feats = base.classifier.in_features
        base.classifier = nn.Identity()

        Wrapper = BayesianModel if len(heads)==1 else BayesianMultiHeadModel
        args = {
            'base_model': base,
            'feature_dim': in_feats,
            'dropout_p': dropout_p
        }
        if len(heads) == 1:
            args['num_classes'] = next(iter(heads.values()))
        else:
            args['heads'] = heads
        model = Wrapper(**args)

    # -----------------------------------------
    # 4) timm EfficientNets
    # -----------------------------------------
    elif name.startswith('efficientnet'):
        try:
            base = timm.create_model(name, pretrained=True)
        except RuntimeError as e:
            if 'No pretrained weights exist' in str(e):
                print(f"Warning: initializing {name} randomly (no pretrained weights).")
                base = timm.create_model(name, pretrained=False)
            else:
                raise

        in_feats = base.classifier.in_features
        base.classifier = nn.Identity()

        Wrapper = BayesianModel if len(heads)==1 else BayesianMultiHeadModel
        args = {
            'base_model': base,
            'feature_dim': in_feats,
            'dropout_p': dropout_p
        }
        if len(heads) == 1:
            args['num_classes'] = next(iter(heads.values()))
        else:
            args['heads'] = heads
        model = Wrapper(**args)

    # -----------------------------------------
    # 5) timm ResNeXt / RegNetY families
    # -----------------------------------------
    elif name.startswith('resnext') or name.startswith('regnety'):
        base = timm.create_model(name, pretrained=True)

        # Find and replace classifier layer to get feature dim
        if hasattr(base, 'fc'):
            in_feats = base.fc.in_features
            base.fc = nn.Identity()
        elif hasattr(base, 'classifier'):
            in_feats = base.classifier.in_features
            base.classifier = nn.Identity()
        elif hasattr(base, 'head') and hasattr(base.head, 'fc'):
            in_feats = base.head.fc.in_features
            base.head.fc = nn.Identity()
        else:
            raise ValueError(f"Cannot locate classifier on {name}")

        Wrapper = BayesianModel if len(heads)==1 else BayesianMultiHeadModel
        args = {
            'base_model': base,
            'feature_dim': in_feats,
            'dropout_p': dropout_p
        }
        if len(heads) == 1:
            args['num_classes'] = next(iter(heads.values()))
        else:
            args['heads'] = heads
        model = Wrapper(**args)

    # ------------------------
    # 6) Unsupported model
    # ------------------------
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Move to device if specified
    if device:
        model.to(device)

    return model
