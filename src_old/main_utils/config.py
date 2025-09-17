# src/main_utils/config.py
from pydantic import BaseModel, Field, validator
from typing import Literal
import yaml

class PathsConfig(BaseModel):
    run_name: str
    output_path_model: str
    output_path_dict: str
    hdf5_path: str

class ModelConfig(BaseModel):
    name: Literal['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'Densenet121', 'Densenet169', 'Densenet201', 'Vision'] = 'Resnet18'
    num_classes: int = 2
    input_size: int = 224
    use_checkpointing: bool = False
    dropout_p: float = 0.5

class OptimizerConfig(BaseModel):
    type: Literal['Adam', 'AdamW'] = 'AdamW'
    learning_rate: float = 0.001

class SchedulerConfig(BaseModel):
    type: Literal['reduce_on_plateau', 'cosine_warm_up'] = 'reduce_on_plateau'
    factor: float = 0.1
    patience: int = 3
    warmup_ratio: float = 0.1
    
class HDF5DatasetConfig(BaseModel):
    cache_in_memory: bool = True

class AugmentationConfig(BaseModel):
    type: Literal['none', 'basic', 'advanced', 'mixup', 'cutmix'] = 'basic'
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0

class TrainingConfig(BaseModel):
    batch_size: int = Field(..., description="Number of samples per training batch")
    epochs: int = Field(..., description="Number of training epochs")
    folds: int = Field(..., description="Number of folds for cross-validation")
    test_size: float = Field(..., description="Proportion of the dataset to include in the test split")
    #val_size: float = Field(..., description="Proportion of the dataset to include in the validation split")
    tune_hyperparameters: bool = Field(..., description="Whether to perform hyperparameter tuning")
    n_trials: int = Field(..., description="Number of hyperparameter tuning trials")
    mc_iterations: int = Field(10, description="Number of Monte Carlo iterations for uncertainty estimation")
    seed: int = 42
    use_mixed_precision: bool = True
    early_stopping_patience: int = 7
    early_stopping_delta: float = 0.001
    gradient_clipping: float = 1.0
    
class ProfilerConfig(BaseModel):
    activation: bool = False
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 2
    
class TensorboardConfig(BaseModel):
    activation: bool = False


class MiscConfig(BaseModel):
    cuda: int = 1
    criterion_weight: Literal['None', 'equal_weight', 'weight10'] = 'None'
    
class DataConfig(BaseModel):
    num_workers: int = 4
    prefetch_factor: int = 4

class Config(BaseModel):
    paths: PathsConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    HDF5Dataset: HDF5DatasetConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    profiler: ProfilerConfig
    tensorboard: TensorboardConfig
    misc: MiscConfig
    data: DataConfig
    loss_function: Literal['CrossEntropyLoss','ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss'] = 'CrossEntropyLoss'  # Added field

    @validator('training')
    def check_splits(cls, v, values):
        # Ensure that test_size + val_size < 1.0
        if v.test_size >= 1.0:
            raise ValueError("Test size must sum to less than 1.0")
        return v

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return Config(**raw_config)
