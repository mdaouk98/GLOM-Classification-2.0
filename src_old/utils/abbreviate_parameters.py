# src/utilabbreviate_parameters/.py

from utils import sanitize_string



def abbreviate_parameters(config):
    """Abbreviate training parameters for directory naming."""
    model_abbr = {
        'Resnet18': 'Res18',
        'Resnet34': 'Res34',
        'Densenet121': 'Den121',
        'Densenet169': 'Den169',
        'Densenet201': 'Den201',
        'Vision': 'ViT'
    }.get(config.model.name, config.model.name)

    loss_map = {
        'CrossEntropyLoss': 'CE',
        'FocalLoss': 'FL'
    }
    loss_abbr = loss_map.get(config.loss_function, config.loss_function)

    optimizer_abbr = config.optimizer.type
    scheduler_map = {
        'reduce_on_plateau': 'schedROP',
        'cosine_warm_up': 'schedCWU'
    }
    scheduler_abbr = scheduler_map.get(config.scheduler.type, config.scheduler.type)

    augmentation_map = {
        'none': 'augN',
        'basic': 'augB',
        'advanced': 'augA',
        'mixup': 'augM',
        'cutmix': 'augC'
    }
    augmentation_abbr = augmentation_map.get(config.augmentation.type, config.augmentation.type)

    dir_name = (f"{model_abbr}"
                f"_in{config.model.input_size}"
                f"_{loss_abbr}"
                f"_{optimizer_abbr}"
                f"_lr{config.optimizer.learning_rate}"
                f"_bs{config.training.batch_size}"
                f"_{scheduler_abbr}"
                f"_{augmentation_abbr}")

    dir_name = sanitize_string(dir_name)
    return dir_name

