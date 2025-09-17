# src/mc_utils/mc_clustering/mc_clustering_main.py

"""
Extract intermediate feature vectors from a trained model for clustering/visualization.
This script uses the same configuration and data splitting logic as main_train.py.
It computes statistics on a main dataset (combining original train and val)
and also computes MC-dropout statistics on a subset (10%) of the test dataset,
called new_images.

Note: The statistics (and MC statistics) are computed over the entire aggregated dataset
across all folds.
"""



import argparse
import logging
import os
from os.path import join
import numpy as np
import random
import torch
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

from main_utils.config import load_config
from mc_utils.mc_clustering import build_transform_processor, ensure_dir_exists, create_metrics_dataframe, process_main_dataset, process_new_images_dataset, process_test_dataset, compute_p_value, retrain_main


def mc_clustering_main() -> None:
    """
    Main function to extract intermediate feature vectors from a trained model,
    compute feature statistics, and evaluate distance metrics for both the main (train+val)
    and new_images (subset of test) datasets. The results are saved to disk.
    """
    parser = argparse.ArgumentParser(
        description="Feature extraction and statistics computation for main and new_images splits."
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration YAML file.')
    args = parser.parse_args()
    
    # Setup logging configuration.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration and select computation device.
    config = load_config(args.config)
    device = torch.device(f"cuda:{config.misc.cuda}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Build transforms and processor based on model type.
    transform, processor, for_vision = build_transform_processor(config)
    
    # Read indices and labels from an HDF5 file.
    try:
        with h5py.File(config.paths.hdf5_path, 'r') as f:
            all_labels = f['labels'][:]
            total_samples = len(all_labels)
            all_indices = np.arange(total_samples)
    except Exception as e:
        logging.error(f"Error reading HDF5 file: {e}")
        return
    
    # Split the data into main (train+val) and test sets.
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices,
        all_labels,
        test_size=config.training.test_size,
        random_state=config.training.seed,
        stratify=all_labels
    )
    logging.info(f"Main dataset (train+val) samples: {len(train_val_indices)}")
    logging.info(f"Test dataset samples: {len(test_indices)}")
    
    # Ensure the output directory exists.
    output_dir = join("metrics", config.paths.run_name)
    ensure_dir_exists(output_dir)
    
    # Process the main dataset.
    main_results = process_main_dataset(train_val_indices, train_val_labels, transform, processor, for_vision, device, config)
    if main_results is None:
        logging.error("Main dataset processing failed. Exiting.")
        return

    # Save main dataset statistics.
    output_path_main = join(output_dir, "main_feature_stats_and_per_class_softmax_proababilities.npz")
    np.savez(output_path_main, **main_results)
    logging.info(f"Main dataset feature statistics per class softmax probabilities saved to {output_path_main}.")

    # Create an Excel report for main dataset distance metrics.
    main_length = main_results['euclidean_dists'].shape[0]
    df_main = create_metrics_dataframe(
        indices=np.arange(main_length),
        euclidean_dists=main_results['euclidean_dists'],
        cosine_sims=main_results['cosine_sims'],
        mahalanobis_dists=main_results['mahalanobis_dists'],
        main_softmax_class_1_probabilities= main_results['main_softmax_class_1_probabilities'],
        euclidean_thresh_95= main_results['euclidean_thresh_95'],
        euclidean_thresh_90= main_results['euclidean_thresh_90'],
        euclidean_thresh_85= main_results['euclidean_thresh_85'],
        euclidean_thresh_80= main_results['euclidean_thresh_80'],
        cosine_thresh_5= main_results['cosine_thresh_5'],
        cosine_thresh_10= main_results['cosine_thresh_10'],
        cosine_thresh_15= main_results['cosine_thresh_15'],
        cosine_thresh_20= main_results['cosine_thresh_20'],
        mahalanobis_thresh_95_percentile= main_results['mahalanobis_thresh_95_percentile'],
        mahalanobis_thresh_90_percentile= main_results['mahalanobis_thresh_90_percentile'],
        mahalanobis_thresh_85_percentile= main_results['mahalanobis_thresh_85_percentile'],
        mahalanobis_thresh_80_percentile= main_results['mahalanobis_thresh_80_percentile'],
        mahalanobis_thresh_chi2= main_results['mahalanobis_thresh_chi2'],
        class0_thresh_95= main_results['class0_thresh_95'],
        class0_thresh_90= main_results['class0_thresh_90'],
        class0_thresh_85= main_results['class0_thresh_85'],
        class0_thresh_80= main_results['class0_thresh_80'],
        class1_thresh_5= main_results['class1_thresh_5'],
        class1_thresh_10= main_results['class1_thresh_10'],
        class1_thresh_15= main_results['class1_thresh_15'],
        class1_thresh_20= main_results['class1_thresh_20']
    )
    excel_output_path_main = join(output_dir, "main_dataset_images_distance_metrics.xlsx")
    df_main.to_excel(excel_output_path_main, index=False)
    logging.info(f"Distance metrics for main dataset saved to {excel_output_path_main}.")

    # Process new_images dataset (subset of test) for MC-dropout statistics.
    new_images_results = process_new_images_dataset(test_indices, test_labels, transform, processor, for_vision, device, config, main_results)
    if new_images_results is None:
        logging.error("New images processing failed.")
        return

    # Save new_images statistics.
    output_path_new_images = join(output_dir, "new_images_feature_stats.npz")
    np.savez(output_path_new_images, **new_images_results)
    logging.info(f"New_images feature statistics saved to {output_path_new_images}.")
    
    # Compute p-value for distributions
    p_value_softmax_results = compute_p_value(new_images_results['mc_softmax_class_1_probabilities'],
                                      main_results['class0_probabilities'],
                                       main_results['class1_probabilities'],
                                       comparing = 'Softmax',
                                        alpha = 0.05
                                        )
    p_value_euclidean_dist_results = compute_p_value(new_images_results['mc_euclidean_dists'],
                                      main_results['class0_euclidean_dists'],
                                       main_results['class1_euclidean_dists'],
                                       comparing = 'Eucleadian Distance',
                                        alpha = 0.05
                                        )
    p_value_cosine_sims_results = compute_p_value(new_images_results['mc_cosine_sims'],
                                      main_results['class0_cosine_sims'],
                                       main_results['class1_cosine_sims'],
                                       comparing = 'Cosine Similarities',
                                        alpha = 0.05
                                        )
    p_value_mahalanobis_results = compute_p_value(new_images_results['mc_mahalanobis_dists'],
                                      main_results['class0_mahalanobis_dists'],
                                       main_results['class1_mahalanobis_dists'],
                                       comparing = 'Mahalanobis Distances',
                                        alpha = 0.05
                                        )
    p_value_dict = { 'p_value_softmax_results': p_value_softmax_results, 
                     'p_value_euclidean_dist_results': p_value_euclidean_dist_results,
                     'p_value_cosine_sims_results': p_value_cosine_sims_results,
                     'p_value_mahalanobis_results': p_value_mahalanobis_results
                     }          
    output_path_p_values_npz = join(output_dir, "p_values.npz")
    np.savez(output_path_p_values_npz, **p_value_dict)
    logging.info(f"p_values .npz file saved to {output_path_p_values_npz}.")
                                        
    p_value_softmax_df = pd.DataFrame(p_value_softmax_results)
    p_value_euclidean_dist_df = pd.DataFrame(p_value_euclidean_dist_results)
    p_value_cosine_sims_df = pd.DataFrame(p_value_cosine_sims_results)
    p_value_mahalanobis_df = pd.DataFrame(p_value_mahalanobis_results)
    
    p_value_df = pd.merge(p_value_softmax_df, p_value_euclidean_dist_df, on='new_image_index')
    p_value_df = pd.merge(p_value_df, p_value_cosine_sims_df, on='new_image_index')
    p_value_df = pd.merge(p_value_df, p_value_mahalanobis_df, on='new_image_index')
    
    excel_output_path_p_value = join(output_dir, "p_values.xlsx")
    p_value_df.to_excel(excel_output_path_p_value, index=False)
    logging.info(f"p-values for new images different distributions saved to {excel_output_path_p_value}.")
    

    # Create an Excel report for new_images distance metrics.
    num_new = new_images_results['means'].shape[0]
    df_new = create_metrics_dataframe(
        indices=np.arange(num_new),
        euclidean_dists=new_images_results['euclidean_dists'],
        cosine_sims=new_images_results['cosine_sims'],
        mahalanobis_dists=new_images_results['mahalanobis_dists'],
        main_softmax_class_1_probabilities= new_images_results['mc_softmax_class_1_probabilities_mean'],
        euclidean_thresh_95= main_results['euclidean_thresh_95'],
        euclidean_thresh_90= main_results['euclidean_thresh_90'],
        euclidean_thresh_85= main_results['euclidean_thresh_85'],
        euclidean_thresh_80= main_results['euclidean_thresh_80'],
        cosine_thresh_5= main_results['cosine_thresh_5'],
        cosine_thresh_10= main_results['cosine_thresh_10'],
        cosine_thresh_15= main_results['cosine_thresh_15'],
        cosine_thresh_20= main_results['cosine_thresh_20'],
        mahalanobis_thresh_95_percentile= main_results['mahalanobis_thresh_95_percentile'],
        mahalanobis_thresh_90_percentile= main_results['mahalanobis_thresh_90_percentile'],
        mahalanobis_thresh_85_percentile= main_results['mahalanobis_thresh_85_percentile'],
        mahalanobis_thresh_80_percentile= main_results['mahalanobis_thresh_80_percentile'],
        mahalanobis_thresh_chi2= main_results['mahalanobis_thresh_chi2'],
        class0_thresh_95= main_results['class0_thresh_95'],
        class0_thresh_90= main_results['class0_thresh_90'],
        class0_thresh_85= main_results['class0_thresh_85'],
        class0_thresh_80= main_results['class0_thresh_80'],
        class1_thresh_5= main_results['class1_thresh_5'],
        class1_thresh_10= main_results['class1_thresh_10'],
        class1_thresh_15= main_results['class1_thresh_15'],
        class1_thresh_20= main_results['class1_thresh_20']
    )
    excel_output_path_new = join(output_dir, "new_images_distance_metrics.xlsx")
    df_new.to_excel(excel_output_path_new, index=False)
    logging.info(f"Distance metrics for new images saved to {excel_output_path_new}.")
    
    # Process original test dataset (remaining subset of test) for MC-dropout statistics.
    original_test_results = process_test_dataset(test_indices, test_labels, transform, processor, for_vision, device, config)
    if original_test_results is None:
        logging.error("Original Test dataset processing failed.")
        return
        
    # Save original test statistics.
    output_path_original_test = join(output_dir, "original_test_uncertainty_stats.npz")
    np.savez(output_path_original_test, **original_test_results)
    logging.info(f"Original test uncertainty saved to {output_path_original_test}.")
    
    # Retrain model (Step 3)
    random.seed(config.training.seed)  # Set a reproducible seed
    random_numbers = random.sample(range(145), 5)
    logging.info(f"Our random numbers are {random_numbers}")
    for i in random_numbers:
      # Split out new_images from test indices (10% for new images).
      new_images_indices, remaining_test_indices, new_images_labels, remaining_test_labels = train_test_split(
          test_indices,
          test_labels,
          test_size=0.9,
          random_state=config.training.seed,
          stratify=test_labels
      )
      new_train_val_indices = train_val_indices.copy()
      new_train_val_labels = train_val_labels.copy()
      new_train_val_indices = np.append(new_train_val_indices,new_images_indices[i])
      new_train_val_labels = np.append(new_train_val_labels,new_images_labels[i])
      retrain_main(config, new_train_val_indices, new_train_val_labels)
      
      logging.info(f"new_train_val_indices is of shape{new_train_val_indices.shape} compared to train_val_indices of shape {train_val_indices}")
      
      # Process retrained test dataset (remaining subset of test) for MC-dropout statistics.
      retrained_test_results = process_test_dataset(test_indices, test_labels, transform, processor, for_vision, device, config,retrained = True)
      if retrained_test_results is None:
          logging.error("Retrained Test dataset processing failed for image {i}.")
          return
          
      # Save retrained test statistics.
      output_path_retrained_test = join(output_dir, f"retrained_test_uncertainty_stats_image{i}.npz")
      np.savez(output_path_retrained_test, **retrained_test_results)
      logging.info(f"Retrained test uncertainty for image {i} saved to {output_path_retrained_test}.")
      
      
    
