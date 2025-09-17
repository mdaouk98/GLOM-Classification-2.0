# src/main_utils/datasets_utils/TransformWrapper.py

import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class TransformWrapper(Dataset):
    """
    Dataset wrapper that applies preprocessing to images and converts labels to tensors.

    Wraps a base_dataset which yields (image, label) pairs, where:
      - image can be a PIL.Image or NumPy array
      - label can be a scalar, array, or dict of scalars/arrays

    Parameters:
        base_dataset: Dataset returning (img, label)
        transform:   Albumentations-style transform (callable that accepts image=np.array)
                     or torchvision.transforms.Compose
        processor:   HuggingFace vision processor (e.g., ViTFeatureExtractor)
        for_vision:  If True, use 'processor' instead of 'transform'
    """
    def __init__(self, base_dataset, transform=None, processor=None, for_vision=False):
        self.base_dataset = base_dataset
        self.transform     = transform
        self.processor     = processor
        self.for_vision    = for_vision

    def __len__(self):
        """Return total samples from the wrapped dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Fetch one sample, apply image preprocessing, and convert label(s) to torch.Tensor.

        Returns:
            img:   preprocessed image tensor (e.g., CxHxW)
            label: torch.Tensor or dict of torch.Tensor
        """
        # --- 0) Retrieve raw sample from base dataset
        img, label = self.base_dataset[idx]

        # --- 1) Ensure image is a PIL.Image
        #     If it's a NumPy array, convert to uint8 PIL
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.astype('uint8'))

        # --- 2) Apply the chosen image preprocessing pipeline
        if self.for_vision and self.processor:
            # Use HF processor (returns a batch, so take [0])
            processed = self.processor(images=img, return_tensors="pt")
            img = processed['pixel_values'][0]
        elif self.transform:
            # Distinguish between torchvision.Compose vs. Albumentations
            if isinstance(self.transform, torchvision.transforms.Compose):
                img = self.transform(img)
            else:
                # Albumentations-style: expects dict input, returns dict with 'image'
                img = self.transform(image=np.array(img))['image']

        # --- 3) Convert label(s) to torch.Tensor
        if isinstance(label, dict):
            # Multi-head labels come as dicts
            if len(label) == 1:
                # If only one head, unwrap the single value for simplicity
                only_val = next(iter(label.values()))
                label = torch.tensor(only_val, dtype=torch.long)
            else:
                # True multi-head: convert each head separately
                label = {
                    head: torch.tensor(val, dtype=torch.long)
                    for head, val in label.items()
                }
        else:
            # Scalar or array label ? single tensor
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)

        return img, label
