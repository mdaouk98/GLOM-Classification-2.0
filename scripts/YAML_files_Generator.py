import yaml
import os
from copy import deepcopy

# Base configuration as a dictionary
base_config = {
    "paths": {
        "run_name": "loss_function/CrossEntropy/R18",
        "output_path_model": "model",
        "output_path_dict": "training_dict",
        "hdf5_path": "data/images_224.hdf5"
    },
    "model": {
        "name": "Resnet18",  # Options: Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, Densenet121, Densenet169, Densenet201, Vision 
        "num_classes": 2,
        "input_size": 224,
        "use_checkpointing": False,
        "dropout_p": 0.5
    },
    "optimizer": {
        "type": "Adam",  # Options: Adam, AdamW
        "learning_rate": 0.001
    },
    "scheduler": {
        "type": "cosine_warm_up",  # Options: reduce_on_plateau, cosine_warm_up
        "factor": 0.1,
        "patience": 3,
        "warmup_ratio": 0.1
    },
    "HDF5Dataset": {
        "cache_in_memory": True
    },
    "augmentation": {
        "type": "none",  # Options: none, basic, advanced, mixup, cutmix
        "mixup_alpha": 1.0,
        "cutmix_alpha": 1.0
    },
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "folds": 5,
        "test_size": 0.15,
        "tune_hyperparameters": False,
        "n_trials": 20,
        "mc_iterations": 50,
        "seed": 42,
        "use_mixed_precision": True,
        "early_stopping_patience": 7,
        "early_stopping_delta": 0.001,
        "gradient_clipping": 1.0
    },
    "profiler": {
        "activation": False,
        "wait_steps": 1,
        "warmup_steps": 1,
        "active_steps": 3,
        "repeat": 2
    },
    "tensorboard": {
        "activation": False
    },
    "misc": {
        "cuda": 1,
        "criterion_weight": None # Options: 'None', 'equal_weight', 'weight10'
    },
    "data": {
        "num_workers": 1,
        "prefetch_factor": 1
    },
    "loss_function": "CrossEntropyLoss"  # Options: 'CrossEntropyLoss','ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss'
}





for image_input in [224,512,1024]:
  for augmentation_type in ['none', 'basic', 'advanced']:
    for scheduler_type in ['reduce_on_plateau, cosine_warm_up']:
      for criterion_weight in ['None', 'equal_weight', 'weight10']:
        for optimizer_type in ['Adam, AdamW']:
          for optimizer_lr in [0.01,0.001,0.0001]:
            for loss_function in ['CrossEntropyLoss','ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss']:
              for model in ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'Densenet121', 'Densenet169', 'Densenet201', 'Vision']:
                # Create a deep copy of the base configuration for each variant
                config_variant = deepcopy(base_config)
                config_variant["model"]["name"] = model
                config_variant["model"]["input_size"] = image_input
                config_variant["misc"]["criterion_weight"] = criterion_weight
                config_variant["optimizer"]["type"] = optimizer_type
                config_variant["optimizer"]["learning_rate"] = optimizer_lr
                config_variant["scheduler"]["type"] = scheduler_type
                config_variant["augmentation"]["type"] = augmentation_type
                config_variant["loss_function"] = loss_function
                
                if model in ['Resnet18','Densenet201']:
                  cuda_number = 1
                if model in ['Resnet34','Vision']:
                  cuda_number = 2
                if model in ['Resnet50']:
                  cuda_number = 3
                if model in ['Resnet101']:
                  cuda_number = 4
                if model in ['Resnet152']:
                  cuda_number = 5
                if model in ['Densenet121']:
                  cuda_number = 6
                if model in ['Densenet169']:
                  cuda_number = 7
                config_variant["misc"]["cuda"] = cuda_number
          
                # Update the run_name accordingly
                config_variant["paths"]["run_name"] = f"image_size_{image_input}/augmentation_{augmentation_type}/scheduler_type_{scheduler_type}/criterion_weight_{criterion_weight}/optimizer_type_{optimizer_type}/optimizer_lr_{optimizer_lr}/loss_function_{loss_function}/model_{model}"
                config_variant["paths"]["hdf5_path"] = f"data/images_{image_input}.hdf5"
                
                # Define the filename (adjust the folder structure as needed)
                filename = os.path.join(f"{config_variant["paths"]["run_name"]}.yaml")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
          
                # Write the YAML file
                with open(filename, "w") as file:
                    yaml.dump(config_variant, file, sort_keys=False)
                
                print(f"Created: {filename}")
