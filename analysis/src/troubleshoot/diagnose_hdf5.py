# diagnose_hdf5.py

import h5py
import sys

def diagnose_hdf5(hdf5_path):
    try:
        with h5py.File(hdf5_path, 'r') as f:
            print(f"Keys in the HDF5 file: {list(f.keys())}")
            images = f['images']
            labels = f['labels']
            print(f"Images shape: {images.shape}, dtype: {images.dtype}")
            print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    except Exception as e:
        print(f"Failed to open HDF5 file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != '--hdf5_path':
        print("Usage: python diagnose_hdf5.py --hdf5_path <path_to_hdf5>")
    else:
        diagnose_hdf5(sys.argv[2])

