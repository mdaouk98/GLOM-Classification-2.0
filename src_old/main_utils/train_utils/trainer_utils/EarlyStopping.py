# src/utils/EarlyStopping.py

import numpy as np
import logging

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0.0):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            delta (float): Minimum change in loss to be considered an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.best_score = -np.Inf  # Avoids None-check
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        Checks whether training should be stopped based on validation loss.

        Args:
            val_loss (float): Current validation loss.
        """
        score = -val_loss  # Since lower loss is better, we negate it for easier comparison.

        if score > self.best_score:  # Any improvement
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0  # Reset counter since we improved
        elif score >= self.best_score - self.delta:  
            self.counter = 0  # Allow small fluctuations within delta range
        else:
            self.counter += 1  # Only count worsening beyond delta
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True


