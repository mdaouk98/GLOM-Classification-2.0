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
        "name": '',  # Options: Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, Densenet121, Densenet169, Densenet201, Vision 
        "multihead": True,
        "label": 'class',
        "num_classes": 2,
        "input_size": 224,
        "use_checkpointing": False,
        "dropout_p": 0.5
    },
    "multihead":{
        "labels":         ['class',       'stain'],
        "num_classes":    [2,             4],
        "loss_functions": ['CrossEntropyLoss','ReverseCrossEntropyLoss'],
        "loss_weights":   [1.0,           0.5],
    },    
    "optimizer": {
        "type": "Adam",  # Options: Adam, AdamW
        "learning_rate": 0.0001
    },
    "scheduler": {
        "type": "reduce_on_plateau",  # Options: reduce_on_plateau, cosine_warm_up
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
        "split_level": 'wsi',
        "batch_size": 32,
        "accumulation_steps": 1,
        "epochs": 50,
        "folds": 5,
        "test_size": 0.15,
        "tune_hyperparameters": False,
        "n_trials": 20,
        "mc_iterations": 50,
        "seed": 42,
        "use_mixed_precision": True,
        "early_stopping_patience": 7,
        "early_stopping_delta": 0.0001,
        "gradient_clipping": 1.0,
        "remove_infinite_grad_batch": False
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
        "cuda": '',
        "criterion_weight": 'None' # Options: 'None', 'equal_weight', 'weight10'
    },
    "data": {
        "num_workers": 4,
        "prefetch_factor": 4
    },
    "loss_function": "CrossEntropyLoss"  # Options: 'CrossEntropyLoss','ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss'
}




#for image_input in [224,384,512,1024]:
#  for augmentation_type in ['none', 'basic', 'advanced','mixup','cutmix']:
#    for scheduler_type in ['reduce_on_plateau','cosine_warm_up']:
#      for criterion_weight in ['None', 'equal_weight', 'weight10']:
#        for optimizer_type in ['Adam', 'AdamW']:
#          for optimizer_lr in [0.0001,0.00001]:
#            for loss_function in ['CrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss']:
#              for model in ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152', 'Densenet121', 'Densenet169', 'Densenet201', 'Vision']:
#                # Create a deep copy of the base configuration for each variant
#                config_variant = deepcopy(base_config)
#                config_variant["model"]["name"] = model
#                config_variant["model"]["input_size"] = image_input
#                config_variant["misc"]["criterion_weight"] = criterion_weight
#                config_variant["optimizer"]["type"] = optimizer_type
#                config_variant["optimizer"]["learning_rate"] = optimizer_lr
#                config_variant["scheduler"]["type"] = scheduler_type
#                config_variant["augmentation"]["type"] = augmentation_type
#                config_variant["loss_function"] = loss_function
#                
#                if model in ['Resnet18','Densenet201']:
#                  cuda_number = 1
#                if model in ['Resnet34']:
#                  cuda_number = 2
#                if model in ['Resnet50']:
#                  cuda_number = 3
#                if model in ['Resnet101']:
#                  cuda_number = 4
#                if model in ['Resnet152']:
#                  cuda_number = 5
#                if model in ['Densenet121']:
#                  cuda_number = 6
#                if model in ['Densenet169']:
#                  cuda_number = 7
#                if model in ['Vision']:
#                    cuda_number = 0
#                config_variant["misc"]["cuda"] = cuda_number
#          
#                # Update the run_name accordingly
#                config_variant["paths"]["run_name"] = f"image_size_{image_input}/augmentation_{augmentation_type}/scheduler_type_{scheduler_type}/criterion_weight_{criterion_weight}/optimizer_type_{optimizer_type}/optimizer_lr_{optimizer_lr}/loss_function_{loss_function}/model_{model}"
#                config_variant["paths"]["hdf5_path"] = f"data/images_{image_input}.hdf5"
                
#                # Define the filename (adjust the folder structure as needed)
#                filename = os.path.join(f"{config_variant["paths"]["run_name"]}.yaml")
#                os.makedirs(os.path.dirname(filename), exist_ok=True)
          
                # Write the YAML file
#                with open(filename, "w") as file:
#                    yaml.dump(config_variant, file, sort_keys=False)
                
#                print(f"Created: {filename}")

for image_input in [224]:
  for augmentation_type in ['basic']:
    for scheduler_type in ['reduce_on_plateau']:
      for criterion_weight in ['None']:
        for optimizer_type in ['Adam']:
          for optimizer_lr in [0.0001]:
            for loss_function in ['TotalCrossEntropyLoss']:
              for scale_multiplier in [0.5,0.4,0.3,0.2,0.1,0.01,0.001,0.0001,0.0,-0.0001,-0.001,-0.01,-0.1,-0.2,-0.3,-0.4,-0.5]:
                for label in ['stain','scanner']:
                  for model in ['Resnet18','Resnet34','Resnet50','Resnet101','Resnet152',
            'Densenet121','Densenet169','Densenet201',
            'Vision','resnext101_32x8d','regnety_008','regnety_016',
            'efficientnet_b0','efficientnet_b1','efficientnet_b2',
            'efficientnet_b3','efficientnet_b4','efficientnet_b5',
            ]:
            
            
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
                    config_variant["multihead"]["loss_weights"] = [1.0,scale_multiplier]
                    
                    
                    if model in ['Resnet18','Resnet34','efficientnet_b5']:
                      cuda_number = 1
                    if model in ['Vision','Densenet201']:
                      cuda_number = 2
                    if model in ['Resnet50','resnext101_32x8d']:
                      cuda_number = 3
                    if model in ['Resnet101','efficientnet_b3']:
                      cuda_number = 4
                    if model in ['Resnet152','regnety_016']:
                      cuda_number = 5
                    if model in ['Densenet121','efficientnet_b4']:
                      cuda_number = 6
                    if model in ['Densenet169','regnety_008']:
                      cuda_number = 7
                    if model in ['efficientnet_b0','efficientnet_b1','efficientnet_b2']:
                        cuda_number = 0
                    config_variant["misc"]["cuda"] = cuda_number
              
                    if label == 'stain':
                      config_variant["multihead"]["num_classes"] = [2,4]
                      config_variant["multihead"]["labels"] = ['class','stain']
                    if label == 'scanner':
                      config_variant["multihead"]["num_classes"] = [2,3]
                      config_variant["multihead"]["labels"] = ['class','scanner']
              
                    # Update the run_name accordingly
                    config_variant["paths"]["run_name"] = f"bias2_corr/{label}/scale_{scale_multiplier}/model_{model}"
                    config_variant["paths"]["hdf5_path"] = f"data/images_{image_input}.hdf5"
                    
                    # Define the filename (adjust the folder structure as needed)
                    filename = os.path.join(f"{config_variant["paths"]["run_name"]}.yaml")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
             
                    # Write the YAML file
                    with open(filename, "w") as file:
                        yaml.dump(config_variant, file, sort_keys=False)
                    
                    print(f"Created: {filename}")
  
