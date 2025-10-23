# src/main_utils/config.py

from pydantic import BaseModel, Field, validator
from typing import Literal, Dict, Any
import yaml

# ------------------------------------------------------------------------------
# Configuration submodels
# ------------------------------------------------------------------------------

class PathsConfig(BaseModel):
    """File and directory paths used by the pipeline."""
    run_name: str                 # Unique identifier for this run (used in output dirs)
    output_path_model: str        # Filename (without dir) for saving model checkpoints
    output_path_dict: str         # Filename prefix for saving metrics JSON
    hdf5_path: str                # Path to the input HDF5 dataset

class ModelConfig(BaseModel):
    """
    Defines the backbone architecture and top-level model behavior.
    - name: one of torchvision/timm backbones or 'Vision' for ViT.
    - multihead: whether to train multiple label heads.
    - label / num_classes: used in single-head mode.
    - input_size: image size (square) expected by model.
    - use_checkpointing: whether to apply gradient checkpointing to CNNs.
    - dropout_p: final dropout probability for Bayesian sampling.
    """
    name: Literal[
        'Resnet18','Resnet34','Resnet50','Resnet101','Resnet152',
        'Densenet121','Densenet169','Densenet201','Densenet264d',
        'Vision','resnext101_32x8d','regnety_008','regnety_016',
        'efficientnet_b0','efficientnet_b1','efficientnet_b2',
        'efficientnet_b3','efficientnet_b4','efficientnet_b5',
        'efficientnet_b6','efficientnet_b7'
    ] = 'Resnet18'
    multihead: bool = False
    label: str = 'class'
    num_classes: int = 2
    input_size: int = 224
    use_checkpointing: bool = False
    dropout_p: float = 0.5

class MultiheadConfig(BaseModel):
    """
    Settings for multi-head training:
    - labels: list of head names
    - num_classes: list of number of classes per head
    - loss_functions: list of loss function names per head
    - loss_weights: list of weight scalars per head in total loss
    """
    labels: list[str]
    num_classes: list[int]
    loss_functions: list[str]
    loss_weights: list[float]

class OptimizerConfig(BaseModel):
    """Optimizer type and base learning rate."""
    type: Literal['Adam', 'AdamW'] = 'AdamW'
    learning_rate: float = 0.001

class SchedulerConfig(BaseModel):
    """
    Learning rate scheduler settings.
    - reduce_on_plateau: lower LR when metric plateaus.
    - cosine_warm_up: linear warmup then cosine decay.
    """
    type: Literal['reduce_on_plateau', 'cosine_warm_up'] = 'reduce_on_plateau'
    factor: float = 0.1
    patience: int = 3
    warmup_ratio: float = 0.1

class HDF5DatasetConfig(BaseModel):
    """Whether to preload the HDF5 dataset into RAM."""
    cache_in_memory: bool = True

class AugmentationConfig(BaseModel):
    """
    Data augmentation type:
      - none: only resize+normalize
      - basic / advanced: geometric & photometric
      - mixup / cutmix: geometric only here; mixing applied in training loop
    """
    type: Literal['none', 'basic', 'advanced', 'mixup', 'cutmix'] = 'basic'
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0

class TrainingConfig(BaseModel):
    """
    Training regime parameters.
    - split_level: the level at which we are splitting the data.
    - batch_size, epochs: core training loop sizes.
    - folds: number of CV folds.
    - test_size: fraction of data held out for final test.
    - tune_hyperparameters / n_trials: for Optuna or similar.
    - mc_iterations: Monte Carlo dropout passes for uncertainty.
    - seed: for reproducibility.
    - use_mixed_precision: whether to use torch.cuda.amp.
    - early_stopping_patience / delta: stop when no val-loss improvement.
    - gradient_clipping: max grad norm.
    """
    split_level: Literal['patch','wsi'] = 'patch'
    batch_size: int = Field(..., description="samples per batch")
    accumulation_steps: int = Field(..., description="batches accumulated for gradient")
    epochs: int = Field(..., description="training epochs")
    folds: int = Field(..., description="cross-validation folds")
    test_size: float = Field(..., description="fraction for hold-out test set")
    tune_hyperparameters: bool = Field(..., description="run hyperparameter search")
    n_trials: int = Field(..., description="number of tuning trials")
    mc_iterations: int = Field(10, description="MC-dropout iterations")
    seed: int = 42
    use_mixed_precision: bool = True
    early_stopping_patience: int = 7
    early_stopping_delta: float = 0.001
    gradient_clipping: float = 1.0
    remove_infinite_grad_batch: bool = True

class ProfilerConfig(BaseModel):
    """Settings for PyTorch profiler (if used)."""
    activation: bool = False
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 2

class TensorboardConfig(BaseModel):
    """Whether to launch a TensorBoard writer."""
    activation: bool = False

class MiscConfig(BaseModel):
    """
    Miscellaneous settings:
    - cuda: GPU index to use.
    - criterion_weight: default class weighting for CrossEntropy variants.
    - deterministic: True to ensure reproducibility at the cost of speed training.
    """
    cuda: int = 1
    criterion_weight: Literal['None', 'equal_weight', 'weight10'] = 'None'
    deterministic = True

class DataConfig(BaseModel):
    """DataLoader settings for performance."""
    num_workers: int = 4
    prefetch_factor: int = 4

# ------------------------------------------------------------------------------
# Top-level Config and loader
# ------------------------------------------------------------------------------

class Config(BaseModel):
    """Full configuration combining all submodels."""
    paths: PathsConfig
    model: ModelConfig
    multihead: MultiheadConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    HDF5Dataset: HDF5DatasetConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    profiler: ProfilerConfig
    tensorboard: TensorboardConfig
    misc: MiscConfig
    data: DataConfig

    # legacy / top-level default loss function
    loss_function: Literal['CrossEntropyLoss','ReverseCrossEntropyLoss','TotalCrossEntropyLoss','FocalLoss'] = 'CrossEntropyLoss'

    @validator('training')
    def check_test_size(cls, v):
        """Ensure test_size < 1.0."""
        if v.test_size >= 1.0:
            raise ValueError("test_size must be less than 1.0")
        return v

def load_config(config_path: str) -> Config:
    """
    Read a YAML file and parse it into a validated Config object.
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

