import yaml
import os
from copy import deepcopy

multihead_type = 'scanner'
multihead_study = 't'
model = 'Resnet18'
loss_function_main = 'TotalCrossEntropyLoss'
loss_function_secondary = 'CrossEntropyLoss'
cuda = 3

if multihead_type == 'all':
    multihead = {
        "labels":         ['class','stain', 'scanner'],
        "num_classes":    [2, 4,   3],
        "loss_functions": [loss_function_main, loss_function_secondary, loss_function_secondary],
        "loss_weights":   [1.0,           0.5, 0.5]
    } 
if multihead_type == 'stain':
    multihead = {
        "labels":         ['class',       'stain'],
        "num_classes":    [2,             4],
        "loss_functions": [loss_function_main, loss_function_secondary],
        "loss_weights":   [1.0,           0.5]
    }    
if multihead_type == 'scanner':
    multihead = {
        "labels":         ['class',       'scanner'],
        "num_classes":    [2,             3],
        "loss_functions": [loss_function_main, loss_function_secondary],
        "loss_weights":   [1.0,           0.5]
    }        
   


# Base configuration as a dictionary
base_config = {
    "paths": {
        "run_name": "loss_function/CrossEntropy/R18",
        "output_path_model": "model",
        "output_path_dict": "training_dict",
        "hdf5_path": "data/images_224.hdf5"
    },
    "model": {
        "name": model,  # Options: Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, Densenet121, Densenet169, Densenet201, Vision 
        "multihead": True,
        "label": 'class',
        "num_classes": 2,
        "input_size": 224,
        "use_checkpointing": False,
        "dropout_p": 0.5
    },
    "multihead": multihead,
        
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
        "split_level": 'patch'
        "batch_size": 32,
        "epochs": 50,
        "folds": 5,
        "test_size": 0.15,
        "tune_hyperparameters": False,
        "n_trials": 20,
        "mc_iterations": 50,
        "seed": 42,
        "use_mixed_precision": True,
        "early_stopping_patience": 15,
        "early_stopping_delta": 0.0005,
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
        "cuda": cuda,
        "criterion_weight": 'None' # Options: 'None', 'equal_weight', 'weight10'
    },
    "data": {
        "num_workers": 4,
        "prefetch_factor": 4
    },
    "loss_function": "CrossEntropyLoss"  # Options: 'CrossEntropyLoss','ReverseCrossEntropyLoss', 'TotalCrossEntropyLoss', 'FocalLoss'
}




#for image_input in [224,512,1024]:
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

if multihead_study == 't':
    scale_multiplier = [0.0,-0.1,-0.5,-1.0,-1.5,-2.0,-2.5,-5.0,-10.0]
    scale_order = [0,'n_1','n_5','n1','n1_5','n2','n2_5','n5','n10']
else:
    scale_multiplier = [-0.0001,-0.001,-0.01,-0.1,0,0.1,0.01,0.001,0.0001]
    scale_order = ['n4','n3','n2','n1',0,'p1','p2','p3','p4']
    
for index, scale in enumerate(scale_order):
    # Create a deep copy of the base configuration for each variant
    config_variant = deepcopy(base_config)
    if multihead_type == 'all':
      config_variant["multihead"]["loss_weights"] = [1.0,scale_multiplier[index],scale_multiplier[index]]
    else:
      config_variant["multihead"]["loss_weights"] = [1.0,scale_multiplier[index]]

    # Update the run_name accordingly
    config_variant["paths"]["run_name"] = f"multihead_{multihead_type}_{multihead_study}/{loss_function_secondary}/{model}/scale_{scale}"
    config_variant["paths"]["hdf5_path"] = f"data/images_224.hdf5"
    
    # Define the filename (adjust the folder structure as needed)
    filename = os.path.join(f"{config_variant["paths"]["run_name"]}.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write the YAML file
    with open(filename, "w") as file:
        yaml.dump(config_variant, file, sort_keys=False)
    
    print(f"Created: {filename}")