import h5py
import numpy as np
from tqdm import tqdm

def verify_hdf5(hdf5_path, batch_size=100):
    """
    Verifies the integrity of an HDF5 file by checking the shape and values of images and labels.

    Parameters:
    - hdf5_path: Path to the HDF5 file.
    - batch_size: Number of samples to process in each batch.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            images = f['images']
            labels = f['labels']
            
            total_samples = len(images)
            print(f"Total samples: {total_samples}")

            invalid_images = []
            invalid_labels = []
            
            # Use tqdm for progress tracking
            for i in tqdm(range(0, total_samples, batch_size), desc="Verifying HDF5 data"):
                batch_indices = slice(i, i + batch_size)
                
                # Load a batch of images and labels
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]

                # Check each image and label in the batch
                for idx, (img, lbl) in enumerate(zip(batch_images, batch_labels), start=i):
                    if img.shape != (224, 224, 3):
                        invalid_images.append((idx, img.shape))
                    if lbl not in [0, 1]:
                        invalid_labels.append((idx, lbl))
            
            # Summary of results
            print("\nVerification completed.")
            if invalid_images:
                print(f"Invalid images found: {len(invalid_images)}")
                for idx, shape in invalid_images:
                    print(f" - Image {idx} has shape {shape}")
            else:
                print("All images have valid shapes.")
            
            if invalid_labels:
                print(f"Invalid labels found: {len(invalid_labels)}")
                for idx, lbl in invalid_labels:
                    print(f" - Label {idx} is invalid: {lbl}")
            else:
                print("All labels are valid.")

    except KeyboardInterrupt:
        print("\nVerification interrupted.")
    except Exception as e:
        print(f"Error while verifying HDF5 file: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Verify HDF5 Dataset Integrity')
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 file')
    args = parser.parse_args()
    verify_hdf5(args.hdf5_path)
