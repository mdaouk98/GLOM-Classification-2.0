import numpy as np
import argparse

def read_npz_file(npz_file_path: str) -> dict:
    """
    Reads an NPZ file and prints out the keys, array shapes, and types.
    
    Args:
        npz_file_path (str): Path to the NPZ file.
    
    Returns:
        dict: A dictionary containing the arrays stored in the NPZ file.
    """
    try:
        # Load the NPZ file (allow_pickle=True in case there are pickled objects)
        data = np.load(npz_file_path, allow_pickle=True)
        print("Keys found in NPZ file:")
        for key in data.keys():
            arr = data[key]
            print(f"Key: {key} | Shape: {arr.shape if hasattr(arr, 'shape') else 'N/A'} | Type: {type(arr)}")
        # Return as a normal dictionary for further processing
        return dict(data)
    except Exception as e:
        print(f"Error reading NPZ file: {e}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and display contents of an NPZ file.")
    parser.add_argument("npz_file", type=str, help="Path to the NPZ file.")
    args = parser.parse_args()
    
    loaded_data = read_npz_file(args.npz_file)

print(loaded_data)