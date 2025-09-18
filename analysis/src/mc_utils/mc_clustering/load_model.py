# src/mc_utils/mc_clustering/load_model.py

import logging
import os
from os.path import join
from typing import Any, Dict, List, Optional

import torch

from main_utils.models_utils.get_bayesian_model import get_bayesian_model


CHECKPOINT_DIR = "checkpoints"

def load_model(fold_idx: int, device: torch.device, config: Any, retrained: bool= False) -> Optional[torch.nn.Module]:
    """
    Loads the saved model checkpoint for a given fold.
    """
    if retrained == True:
      checkpoint_path = join(
          CHECKPOINT_DIR, config.paths.run_name,
          f"{config.paths.output_path_model}_fold{fold_idx}_final_retrained.pth"
      )
    else:
      checkpoint_path = join(
          CHECKPOINT_DIR, config.paths.run_name,
          f"{config.paths.output_path_model}_fold{fold_idx}_final.pth"
      )
    if not os.path.isfile(checkpoint_path):
        logging.error(f"Checkpoint not found for fold {fold_idx} at {checkpoint_path}")
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_bayesian_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        input_size=config.model.input_size,
        use_checkpointing=config.model.use_checkpointing,
        dropout_p=config.model.dropout_p,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

