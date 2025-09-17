# src/datasets.py
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms

class HDF5Dataset(Dataset):
    """
    Custom Dataset for loading images and labels from an HDF5 file using memory mapping.
    It uses SWMR (Single Writer Multiple Reader) mode for efficient concurrent reads.
    
    Optionally, if cache_in_memory is set to True and the dataset fits in RAM,
    all images and labels are loaded into memory at initialization to speed up access.
    """
    def __init__(self, hdf5_file, indices, transform=None, processor=None, for_vision=False, cache_in_memory=False):
        self.hdf5_file = hdf5_file
        self.indices = indices
        self.transform = transform
        self.processor = processor
        self.for_vision = for_vision
        self.cache_in_memory = cache_in_memory
        self.cached_data = None

        if self.cache_in_memory:
            # Open the file to cache data and then close it (no need to keep open).
            self._open_file()
            self._cache_data()
            self.close()
        else:
            # For lazy initialization in DataLoader workers.
            self.file = None

    def _open_file(self):
        # Open the HDF5 file in read-only mode with SWMR enabled.
        self.file = h5py.File(self.hdf5_file, 'r', libver='latest', swmr=True)
        self.images = self.file['images']
        self.labels = self.file['labels']

    def _cache_data(self):
        # Cache images and labels into memory for faster access.
        print("Caching dataset into memory...")
        self.cached_data = []
        for idx in self.indices:
            img = self.images[idx]
            label = self.labels[idx]
            self.cached_data.append((img, label))
        print("Caching complete.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.cache_in_memory and self.cached_data is not None:
            img, label = self.cached_data[idx]
        else:
            # Lazy initialization: open the file if it hasn't been opened yet in this worker.
            if self.file is None:
                self._open_file()
            actual_idx = self.indices[idx]
            img = self.images[actual_idx]
            label = self.labels[actual_idx]

        # Convert the numpy image to a PIL Image for processing.
        img = Image.fromarray(img.astype('uint8'))

        if not self.for_vision:
            if self.transform:
                # If using torchvision transforms
                if isinstance(self.transform, torchvision.transforms.Compose):
                    img = self.transform(img)
                else:
                    # Assume an Albumentations transform which expects a numpy array.
                    img = self.transform(image=np.array(img))['image']
        else:
            # For vision models, use the provided processor (e.g., from Hugging Face)
            img = self.processor(images=img, return_tensors="pt")['pixel_values'][0]

        return img, label

    def close(self):
        """Explicitly close the HDF5 file handle."""
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        """
        Exclude the HDF5 file handle (and its datasets) from the pickled state.
        This ensures that each DataLoader worker reopens its own file handle.
        """
        state = self.__dict__.copy()
        state['file'] = None
        state['images'] = None
        state['labels'] = None
        return state

    def __setstate__(self, state):
        """
        Reinitialize the HDF5 file handle and datasets after unpickling (e.g., in DataLoader workers).
        """
        self.__dict__.update(state)
        # Only open the file if not caching; if caching is True, the data is already loaded.
        if not self.cache_in_memory:
            self._open_file()
