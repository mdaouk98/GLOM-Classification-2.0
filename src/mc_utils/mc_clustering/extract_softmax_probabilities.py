# src/mc_utils/mc_clustering/extract_softmax_probabilities.py

import numpy as np
import torch
import tqdm

def extract_softmax_probabilities(model, dataloader, device, mc_dropout=False):
    """
    Run the model in evaluation mode on the provided dataloader (with optional MC dropout)
    and collect the softmax probabilities (using return_features=False).
    """

    model.eval()  # Set the model to evaluation mode
    softmax_values_list = []
    with torch.no_grad():
        for inputs, _ in tqdm.tqdm(dataloader, desc="Extracting softmax distributions", unit="batch"):
            inputs = inputs.to(device)
            # Forward pass: set return_features=False to get logits only
            logits = model(inputs, mc_dropout=False, return_features=False)
            # Compute softmax to get probability distributions
            probs = torch.softmax(logits, dim=1)
            softmax_values_list.append(probs.cpu().numpy())
            
    if softmax_values_list:
        softmax_array = np.concatenate(softmax_values_list, axis=0)
    else:
        softmax_array = np.array([])
    return softmax_array