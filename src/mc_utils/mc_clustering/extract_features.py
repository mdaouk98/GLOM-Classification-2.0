# src/mc_utils/mc_clustering/extract_features.py

import numpy as np
import torch
import tqdm

def extract_features(model, dataloader, device, mc_dropout=False):
    """
    Run the model in evaluation mode on the provided dataloader (with optional MC dropout)
    and collect the intermediate feature vectors (using return_features=True).
    """
    model.eval()
    features_list = []
    with torch.no_grad():
        for inputs, _ in tqdm.tqdm(dataloader, desc="Extracting features", unit="batch"):
            inputs = inputs.to(device)
            # Forward pass with dropout as specified
            output = model(inputs, mc_dropout=mc_dropout, return_features=True)
            # Expecting (logits, features)
            if isinstance(output, tuple):
                _, features = output
            else:
                features = output
            if features is not None:
                features_list.append(features.cpu().numpy())
    if features_list:
        features_array = np.concatenate(features_list, axis=0)
    else:
        features_array = np.array([])
    return features_array


