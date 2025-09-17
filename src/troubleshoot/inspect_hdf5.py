# src/inspect_hdf5.py
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def inspect_hdf5(hdf5_path, index=0):
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        img = images[index]
        label = labels[index]
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        try:
            pil_img = Image.fromarray(img.astype('uint8'))
            pil_img.show()
            plt.imshow(pil_img)
            plt.title(f"Label: {label}")
            plt.show()
        except Exception as e:
            print(f"Failed to convert image: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect HDF5 Dataset')
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--index', type=int, default=0, help='Index of the image to inspect')
    args = parser.parse_args()
    inspect_hdf5(args.hdf5_path, args.index)
