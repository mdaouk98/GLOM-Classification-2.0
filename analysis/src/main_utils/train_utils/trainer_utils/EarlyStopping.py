# src/main_utils/train_utils/trainer_utils/EarlyStopping.py

import numpy as np
import logging

class EarlyStopping:
    """
    Monitor validation loss and stop training early if it does not improve
    after a specified number of epochs (patience).

    Attributes:
        patience (int): Epochs to wait after last improvement before stopping.
        delta (float): Minimum decrease in loss required to reset patience counter.
        counter (int): Counts epochs since last significant improvement.
        early_stop (bool): Flag indicating whether early stopping should occur.
        best_score (float): Best (highest) score seen so far, where score = -loss.
        best_loss (float): Validation loss corresponding to best_score.
    """

    def __init__(self, patience: int = 7, delta: float = 0.0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs with no sufficient improvement before stopping.
            delta (float): Minimum change in loss to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta

        # Internal state
        self.counter = 0
        self.early_stop = False

        # We negate loss to convert "minimize loss" into "maximize score"
        self.best_score = -np.inf  
        self.best_loss = np.inf

    def __call__(self, val_loss: float):
        """
        Check if validation loss has improved and update internal counter/state.

        Should be called at the end of each validation epoch.

        Args:
            val_loss (float): Current epoch's validation loss.
        """
        # Convert loss to a score: higher is better
        score = -val_loss

        # 1) If this is the best score so far, update bests and reset counter
        if score > self.best_score:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            logging.debug(f"Validation loss improved to {val_loss:.4f}. Counter reset.")

        # 2) If within delta of best score, treat as no significant change—reset counter
        elif score >= self.best_score - self.delta:
            self.counter = 0
            logging.debug(
                f"Validation loss {val_loss:.4f} within delta={self.delta} of best {self.best_loss:.4f}. "
                "Counter reset."
            )

        # 3) Otherwise, increment counter and possibly trigger early stopping
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} / {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                logging.warning(
                    f"No improvement in validation loss for {self.patience} epochs. "
                    "Triggering early stop."
                )



