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
        # Build a dict of handles for each requested label head
        self.labels = {name: labels_group[name] for name in self.label_names}

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
