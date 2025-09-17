
#Glomerular Image Classification (Refactored)

This repository contains a training pipeline for classifying glomerular images using various deep learning architectures (ResNet, DenseNet, Vision Transformer) with added features:

- Stratified K-Fold Cross-Validation
- Gradient Checkpointing
- Hyperparameter Tuning (via Optuna)
- Flexible Data Augmentation (basic, advanced, Mixup, CutMix)
- Monte Carlo Dropout for Uncertainty Estimation
- TensorBoard for visualization
- Checkpointing and Resume Support
- Configuration Validation with Pydantic

## Structure

- `configs/`: YAML config files.
- `checkpoints/`: Saved model checkpoints and metrics.
- `metrics/`: Training logs, JSON result files, config snapshots.
- `src/`:
  - `train.py`: Entry point for training.
  - `config.py`: Configuration parsing and validation.
  - `trainer.py`: Training logic (train/validate loops).
  - `evaluation.py`: Monte Carlo evaluation logic.
  - `tuner.py`: Hyperparameter tuning logic (placeholder).
  - `utils.py`: Utilities (mixup, cutmix, early stopping, etc.).
  - `augmentations.py`: Data augmentation pipelines.
  - `datasets.py`: HDF5 dataset class.
  - `models.py`: Model definition and Bayesian wrappers.

## Setup

```bash
conda create -n glom-class python=3.8
conda activate glom-class
pip install -r requirements.txt

