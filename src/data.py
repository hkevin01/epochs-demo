# =============================================================================
# FILE: src/data.py
# PURPOSE: Dataset generation and loading utilities for the epochs-demo project.
# =============================================================================
#
# ID: DATA-001
# Requirement: Generate a reproducible 2-D synthetic classification dataset
#              with configurable class complexity and noise.
# Purpose: Provide a fast dataset so training completes in seconds while still
#          producing interesting decision boundaries.
# Rationale: Synthetic moon dataset exposes non-linear boundaries that make
#            underfitting and overfitting visually obvious.
# Inputs: n_samples (int, >0), noise (float, [0,1]), random_state (int).
# Outputs: DataLoader objects for train and validation splits.
# Preconditions: scikit-learn and torch must be installed.
# Postconditions: Returned DataLoaders yield (FloatTensor, LongTensor) pairs.
# Assumptions: Dataset fits entirely in RAM.
# Side Effects: None.
# Failure Modes: Invalid n_samples raises ValueError from sklearn.
# Error Handling: Delegates to sklearn.
# Constraints: None.
# Verification: Run main.py and confirm loader shapes.
# References: sklearn.datasets.make_moons.
# =============================================================================

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_moons


def generate_dataset(
    n_samples: int = 1000,
    noise: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    ID: DATA-002
    Purpose: Produce (X, y) arrays using the sklearn two-moons generator.
    Inputs:
        n_samples    - total number of points (int, >0)
        noise        - standard deviation of Gaussian noise (float, [0,1])
        random_state - RNG seed for reproducibility (int)
    Outputs:
        X - shape (n_samples, 2) float32 feature array
        y - shape (n_samples,)  int64   label array {0, 1}
    Preconditions: sklearn installed.
    Postconditions: X and y have consistent lengths.
    Failure Modes: n_samples <= 0 raises ValueError.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float32), y.astype(np.int64)


def get_dataloaders(
    n_samples: int = 1000,
    noise: float = 0.20,
    batch_size: int = 64,
    val_split: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    ID: DATA-003
    Purpose: Build train/val DataLoaders from the synthetic dataset.
    Inputs:
        n_samples    - dataset size (int, >0)
        noise        - label noise level (float, [0,1])
        batch_size   - mini-batch size (int, >0)
        val_split    - fraction reserved for validation (float, (0,1))
        random_state - RNG seed (int)
    Outputs:
        train_loader - DataLoader for training split
        val_loader   - DataLoader for validation split
        X_all        - full feature array (for visualization)
        y_all        - full label array  (for visualization)
    Preconditions: 0 < val_split < 1.
    Postconditions: train + val sizes sum to n_samples.
    Failure Modes: Propagates ValueError from generate_dataset.
    """
    X, y = generate_dataset(n_samples=n_samples, noise=noise, random_state=random_state)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(random_state)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X, y
