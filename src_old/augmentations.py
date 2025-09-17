# src/augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation_pipeline(augmentation_type, input_size, mixup_alpha=1.0, cutmix_alpha=1.0):
    """
    Returns an Albumentations augmentation pipeline based on the specified type.
    """
    if augmentation_type == 'basic':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomResizedCrop(height=input_size, width=input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    elif augmentation_type == 'advanced':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomResizedCrop(height=input_size, width=input_size, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    elif augmentation_type in ['mixup', 'cutmix']:
        # Basic transforms only, actual mixup/cutmix handled in training
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomResizedCrop(height=input_size, width=input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    elif augmentation_type == 'none':
        transform = A.Compose([
            A.Resize(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type}")

    return transform
