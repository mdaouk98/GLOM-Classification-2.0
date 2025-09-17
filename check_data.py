# src/train_utils/load_hdf5_data.py

import logging
import numpy as np
from h5py import File
from typing import Any, Dict, List, Optional, Tuple


try:
    with File('data/images_v2_224.hdf5', 'r') as f:
        print(f.keys())
        for key in f.keys():
            print(key,len(key))
        print(f['wsis'])
        print(f['stains'])
        print(f['scanners'])
        if 'images' not in f or 'labels' not in f:
            raise DataLoadingError("HDF5 file must contain 'images' and 'labels' datasets.")
        total_samples = len(f['labels'])
        all_indices = np.arange(total_samples)
        all_labels = f['labels'][:]
        all_wsis = f['wsis'][:]
        all_stains = f['stains'][:]
        all_scanners = f['scanners'][:]
        print(set(all_stains))
        print(set(all_scanners))
        logging.info(f"[Data Loading] Loaded {total_samples} samples from HDF5.")
except FileNotFoundError as fnfe:
    logging.error(f"[Data Loading] HDF5 file not found at path: {config.paths.hdf5_path}")
    raise fnfe
except Exception as e:
    logging.error(f"[Data Loading] Error reading HDF5 file: {e}")
    raise DataLoadingError(f"Error reading HDF5 file: {e}") from e
    
scanner_0_labels = []
scanner_1_labels = []
scanner_2_labels = []
scanner_0_wsis = []
scanner_1_wsis = []
scanner_2_wsis = []
scanner_0_stains = []
scanner_1_stains = []
scanner_2_stains = []
sc0st0 = 0
sc0st0l0 = 0
sc0st0l1 = 0
sc0st1 = 0
sc0st1l0 = 0
sc0st1l1 = 0
sc0st2 = 0
sc0st2l0 = 0
sc0st2l1 = 0
sc0st3 = 0
sc0st3l0 = 0
sc0st3l1 = 0
sc1st0 = 0
sc1st0l0 = 0
sc1st0l1 = 0
sc1st1 = 0
sc1st1l0 = 0
sc1st1l1 = 0
sc1st2 = 0
sc1st2l0 = 0
sc1st2l1 = 0
sc1st3 = 0
sc1st3l0 = 0
sc1st3l1 = 0
sc2st0 = 0
sc2st0l0 = 0
sc2st0l1 = 0
sc2st1 = 0
sc2st1l0 = 0
sc2st1l1 = 0
sc2st2 = 0
sc2st2l0 = 0
sc2st2l1 = 0
sc2st3 = 0
sc2st3l0 = 0
sc2st3l1 = 0
for i in range(len(all_labels)):
    if all_scanners[i] == 0:
        scanner_0_labels.append(all_labels[i])
        scanner_0_wsis.append(all_wsis[i])
        scanner_0_stains.append(all_stains[i])
        if all_stains[i] == 0:
          sc0st0 += 1
          if all_labels[i] == 0:
            sc0st0l0 += 1
          elif all_labels[i] == 1:
            sc0st0l1 += 1
        elif all_stains[i] == 1:
          sc0st1 += 1
          if all_labels[i] == 0:
            sc0st1l0 += 1
          elif all_labels[i] == 1:
            sc0st1l1 += 1
        elif all_stains[i] == 2:
          sc0st2 += 1
          if all_labels[i] == 0:
            sc0st2l0 += 1
          elif all_labels[i] == 1:
            sc0st2l1 += 1
        elif all_stains[i] == 3:
          sc0st3 += 1
          if all_labels[i] == 0:
            sc0st3l0 += 1
          elif all_labels[i] == 1:
            sc0st3l1 += 1
    elif all_scanners[i] == 1:
        scanner_1_labels.append(all_labels[i])
        scanner_1_wsis.append(all_wsis[i])
        scanner_1_stains.append(all_stains[i])
        if all_stains[i] == 0:
          sc1st0 += 1
          if all_labels[i] == 0:
            sc1st0l0 += 1
          elif all_labels[i] == 1:
            sc1st0l1 += 1
        elif all_stains[i] == 1:
          sc1st1 += 1
          if all_labels[i] == 0:
            sc1st1l0 += 1
          elif all_labels[i] == 1:
            sc1st1l1 += 1
        elif all_stains[i] == 2:
          sc1st2 += 1
          if all_labels[i] == 0:
            sc1st2l0 += 1
          elif all_labels[i] == 1:
            sc1st2l1 += 1
        elif all_stains[i] == 3:
          sc1st3 += 1
          if all_labels[i] == 0:
            sc1st3l0 += 1
          elif all_labels[i] == 1:
            sc1st3l1 += 1
    elif all_scanners[i] == 2:
        scanner_2_labels.append(all_labels[i])
        scanner_2_wsis.append(all_wsis[i])
        scanner_2_stains.append(all_stains[i])   
        if all_stains[i] == 0:
          sc2st0 += 1
          if all_labels[i] == 0:
            sc2st0l0 += 1
          elif all_labels[i] == 1:
            sc2st0l1 += 1
        elif all_stains[i] == 1:
          sc2st1 += 1
          if all_labels[i] == 0:
            sc2st1l0 += 1
          elif all_labels[i] == 1:
            sc2st1l1 += 1
        elif all_stains[i] == 2:
          sc2st2 += 1
          if all_labels[i] == 0:
            sc2st2l0 += 1
          elif all_labels[i] == 1:
            sc2st2l1 += 1
        elif all_stains[i] == 3:
          sc2st3 += 1
          if all_labels[i] == 0:
            sc2st3l0 += 1
          elif all_labels[i] == 1:
            sc2st3l1 += 1 
    
from collections import Counter

print(f'scanner 0: labels: {Counter(scanner_0_labels)}, wsis: {len(set(scanner_0_wsis))}, stains: {Counter(scanner_0_stains)}')
print(f'scanner 1: labels: {Counter(scanner_1_labels)}, wsis: {len(set(scanner_1_wsis))}, stains: {Counter(scanner_1_stains)}')
print(f'scanner 2: labels: {Counter(scanner_2_labels)}, wsis: {len(set(scanner_2_wsis))}, stains: {Counter(scanner_2_stains)}')

print("\n \n \n")
print(f"""scanner 0: stain0: {sc0st0}, non-prolif: {sc0st0l0}, prolif: {sc0st0l1}
                     stain1: {sc0st1}, non-prolif: {sc0st1l0}, prolif: {sc0st1l1}
                     stain2: {sc0st2}, non-prolif: {sc0st2l0}, prolif: {sc0st2l1}
                     stain3: {sc0st3}, non-prolif: {sc0st3l0}, prolif: {sc0st3l1}""")
                     
print(f"""scanner 1: stain0: {sc1st0}, non-prolif: {sc1st0l0}, prolif: {sc1st0l1}
                     stain1: {sc1st1}, non-prolif: {sc1st1l0}, prolif: {sc1st1l1}
                     stain2: {sc1st2}, non-prolif: {sc1st2l0}, prolif: {sc1st2l1}
                     stain3: {sc1st3}, non-prolif: {sc1st3l0}, prolif: {sc1st3l1}""")

print(f"""scanner 2: stain0: {sc2st0}, non-prolif: {sc2st0l0}, prolif: {sc2st0l1}
                     stain1: {sc2st1}, non-prolif: {sc2st1l0}, prolif: {sc2st1l1}
                     stain2: {sc2st2}, non-prolif: {sc2st2l0}, prolif: {sc2st2l1}
                     stain3: {sc2st3}, non-prolif: {sc2st3l0}, prolif: {sc2st3l1}""")