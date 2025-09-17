# src/main_utils/augmentation_utils/get_augmentation_pipeline.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation_pipeline(augmentation_type, input_size, mixup_alpha=1.0, cutmix_alpha=1.0):
    """
    Build and return an Albumentations pipeline based on the requested augmentation_type.

    Args:
        augmentation_type (str): One of 'basic', 'advanced', 'mixup', 'cutmix', or 'none'.
        input_size (int): Target height and width for random crops and resizing.
        mixup_alpha (float): (Unused here) mixup strength, applied at training time.
        cutmix_alpha (float): (Unused here) cutmix strength, applied at training time.

    Returns:
        A.Compose: Composed transform to apply to PIL/NumPy images.
    """
    # -------------------------------------------------------------------------
    # 1) STANDARD “BASIC” AUGMENTATIONS
    #    – flips, small rotations, random crop ? normalize ? tensor
    # -------------------------------------------------------------------------
    if augmentation_type == 'basic':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # 50% chance to flip left-right
            A.VerticalFlip(p=0.5),    # 50% chance to flip up-down
            A.Rotate(limit=15, p=0.5),  # rotate ±15° half the time
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.8, 1.0),        # random area between 80–100% of original
                ratio=(0.9, 1.1),        # allow slight aspect-ratio change
                p=0.5
            ),
            A.Normalize(                # scale pixel values to ImageNet stats
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),               # convert HxWxC NumPy to CxHxW tensor
        ])

    # -------------------------------------------------------------------------
    # 2) “ADVANCED” AUGMENTATIONS
    #    – combine geometric, elastic, photometric distortions, noise, dropout...
    # -------------------------------------------------------------------------
    elif augmentation_type == 'advanced':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),              # more aggressive rotation
            A.ElasticTransform(                     # elastic deformations
                alpha=1.0,
                sigma=50.0,
                approximate=False,
                p=0.5
            ),
            A.GridDistortion(p=0.3),                # grid warping
            A.OpticalDistortion(                    # lens-like distortions
                distort_limit=(0.05, 0.05),
                p=0.5
            ),
            A.RandomBrightnessContrast(             # brightness & contrast jitter
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                p=0.5
            ),
            A.GaussNoise(                           # add Gaussian noise
                mean_range=(0.0, 0.0),
                std_range=(0.01, 0.03),
                per_channel=False,
                p=0.3
            ),
            A.CoarseDropout(                        # random blocks of zeroed pixels
                num_holes_range=(1, 8),
                hole_height_range=(16 / input_size, 16 / input_size),
                hole_width_range=(16 / input_size, 16 / input_size),
                fill=0,
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # -------------------------------------------------------------------------
    # 3) MIXUP / CUTMIX PREPROCESSING
    #    – only geometric & photometric; actual mixup/cutmix logic applied in training loop
    # -------------------------------------------------------------------------
    elif augmentation_type in ['mixup', 'cutmix']:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # -------------------------------------------------------------------------
    # 4) NO AUGMENTATION
    #    – just resize, normalize, to-tensor
    # -------------------------------------------------------------------------
    elif augmentation_type == 'none':
        transform = A.Compose([
            A.Resize(height=input_size, width=input_size),  # simple resize
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # -------------------------------------------------------------------------
    # ERROR FOR UNKNOWN TYPE
    # -------------------------------------------------------------------------
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type!r}")

    return transform
