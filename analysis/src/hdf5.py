import os
# allow large allocations to be carved into smaller segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from time import time
from logging import error, info
from traceback import format_exc
import gc
import torch

from main_utils.config import load_config
from main_utils.helpers_utils import set_seed
from main_utils.train_utils import load_hdf5_data, setup_transforms_and_processor, setup_device, DataLoadingError
from main_utils.train_utils.dataloader_utils import create_dataloader_from_dataset


# Load configuration and set up directories.
config_path = f"configs/avg_norm/norm_1/model_Resnet18.yaml"
config = load_config(config_path)

# Setup device and seed.
device = setup_device(config)
set_seed(config.training.seed)

# Setup augmentation transforms and processor.
transform_for_train, transform_for_val, processor, for_vision = setup_transforms_and_processor(config)

# Load HDF5 data and perform train/validation/test split.
try:
    all_idxs, all_lbls, all_wsis, _, _, total = load_hdf5_data(config)
    info(f"[Startup] Loaded {total} samples from HDF5.")
except DataLoadingError as e:
    error(f"[Error] Data loading failed: {e}")
    sys.exit(1)
    
    
    
    
    
    
    
    
    
    
    
    
# src/main_utils/datasets_utils/HDF5Dataset.py

import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class HDF5Dataset(Dataset):
    """
    PyTorch Dataset for loading images and multi-head labels from an HDF5 file.

    Directory structure within the HDF5 file:
        /images                - dataset of uint8 images, shape (N, H, W, C)
        /labels/{head_name}    - one dataset per label head, each of shape (N, ...)

    Args:
        hdf5_file (str): Path to the .h5 file.
        indices (list[int]): List of sample indices to include.
        transform (callable, optional): torchvision- or PIL-based transform for images.
        processor (callable, optional): e.g., a Vision Transformer processor returning tensors.
        for_vision (bool): If True, use 'processor' instead of 'transform'.
        cache_in_memory (bool): If True, load all selected samples into RAM at init.
        label_names (list[str]): Names of the label heads to load (must be non-empty).
    """
    def __init__(
        self,
        hdf5_file: str,
        indices: list[int],
        transform=None,
        processor=None,
        for_vision: bool = False,
        cache_in_memory: bool = False,
        label_names: list[str] = []
    ):
        assert label_names, "You must pass at least one head in label_names"
        
        # Store arguments
        self.hdf5_file = hdf5_file
        self.indices = indices
        self.transform = transform
        self.processor = processor
        self.for_vision = for_vision
        self.cache_in_memory = cache_in_memory
        self.label_names = label_names

        # Placeholder for in-memory cache
        self.cached_data = None

        # If requested, preload all data into memory once
        if self.cache_in_memory:
            self._open_file()
            self._cache_data()
            self.close()  # close file handle after caching
        else:
            self.file = None  # will be opened lazily in each worker

    def _open_file(self):
        """Open the HDF5 file and set up image & label handles."""
        self.file = h5py.File(self.hdf5_file, 'r', libver='latest', swmr=True)
        self.images = self.file['images']
        labels_group = self.file['labels']
        print('group',labels_group.keys())
        # Build a dict of handles for each requested label head
        self.labels = {name: labels_group[name] for name in self.label_names}
        a = self.labels
        print('a',a)

    def _cache_data(self):
        """Load all selected (image, label) pairs into a Python list."""
        print("Caching HDF5 into memory...")
        self.cached_data = []
        for idx in self.indices:
            img_np = self.images[idx]  # NumPy array, e.g., uint8
            lbl = {name: self.labels[name][idx] for name in self.label_names}
            self.cached_data.append((img_np, lbl))
        print("Cache complete.")

    def __len__(self):
        """Total number of samples in this dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Fetch one sample (image, labels) by index.

        Returns:
            img: PIL image or tensor, after transform/processing.
            lbl: Dict mapping each head name to its label value.
        """
        # --- 1) Load raw NumPy arrays:
        if self.cache_in_memory:
            img_np, lbl = self.cached_data[idx]
        else:
            # Open file on first access in this worker
            if self.file is None:
                self._open_file()
            actual_idx = self.indices[idx]
            img_np = self.images[actual_idx]
            print(self.labels)
            lbl = {name: self.labels[name][actual_idx] for name in self.label_names}

        # --- 2) Convert NumPy image to PIL for consistency
        img = Image.fromarray(img_np.astype('uint8'), 'RGB')

        # --- 3) Apply vision processor or torchvision/PIL transform
        if self.for_vision and self.processor is not None:
            # e.g., HuggingFace vision processor returning a batch tensor
            img = self.processor(images=img, return_tensors="pt")['pixel_values'][0]
        elif self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def close(self):
        """Close the HDF5 file handle if open."""
        if getattr(self, 'file', None) is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        """Ensure file is closed upon deletion of the dataset object."""
        self.close()

    def __getstate__(self):
        """
        Prepare for pickling (e.g., spawning DataLoader workers).
        Drop HDF5 handles so they aren't shared across processes.
        """
        state = self.__dict__.copy()
        state['file'] = None
        state['images'] = None
        state['labels'] = None
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling. Re-open the file if not caching in memory.
        """
        self.__dict__.update(state)
        if not self.cache_in_memory:
            self._open_file()







# Load the dataset once.
label_names = config.multihead.labels if config.model.multihead else [config.model.label]
full_dataset = HDF5Dataset(
    hdf5_file=config.paths.hdf5_path,
    indices=all_idxs,            # initially load all samples
    transform=None,              # wrap later with TransformWrapper
    processor=processor,
    for_vision=for_vision,
    cache_in_memory=False,
    label_names=label_names
)

with h5py.File(config.paths.hdf5_path, 'r') as f:
    # 2) Validate required groups
    if 'images' not in f or 'labels' not in f:
        raise DataLoadingError("HDF5 must contain both 'images' and 'labels' groups.")
  
    # 3) Determine total number of samples
    total_samples = f['images'].shape[0]
    all_indices = np.arange(total_samples)
  
    lbl_grp = f['labels']
    
print(lbl_grp)
