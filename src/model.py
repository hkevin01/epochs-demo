# =============================================================================
# FILE: src/model.py
# PURPOSE: MLP architecture definition for the epochs-demo project.
# =============================================================================
#
# ID: MODEL-001
# Requirement: Define a configurable Multi-Layer Perceptron for binary
#              classification on 2-D input features.
# Purpose: Demonstrate how model capacity interacts with training epochs.
# Rationale: A small MLP is sufficient to learn the moon boundary without
#            dominating GPU/CPU time, keeping the epoch effect visible.
# Inputs: in_features (int), hidden_sizes (list[int]), dropout (float).
# Outputs: nn.Module instance.
# Preconditions: PyTorch installed.
# Postconditions: Forward pass returns raw logits of shape (B, 2).
# Assumptions: Binary classification only (num_classes=2).
# Side Effects: None.
# Failure Modes: Empty hidden_sizes produces a linear model.
# Error Handling: Validates hidden_sizes type in __init__.
# Constraints: None.
# Verification: Confirmed via forward-pass smoke test in main.py.
# References: PyTorch nn.Module docs.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    ID: MODEL-002
    Purpose: Configurable MLP with BatchNorm, ReLU activations, and Dropout.
    Inputs:
        in_features  - number of input features (int, default 2)
        hidden_sizes - list of hidden layer widths (list[int])
        dropout      - dropout probability per hidden layer (float, [0,1))
        num_classes  - number of output classes (int, default 2)
    Outputs: Logit tensor of shape (batch_size, num_classes).
    Preconditions: in_features > 0, all hidden_sizes > 0.
    Postconditions: Output shape is always (B, num_classes).
    Failure Modes: Empty hidden_sizes degrades to a linear model (expected).
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_sizes: list[int] = None,
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers: list[nn.Module] = []
        prev = in_features

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ID: MODEL-003
        Purpose: Run a single forward pass.
        Inputs:  x - (B, in_features) float tensor
        Outputs: (B, num_classes) logit tensor
        Preconditions: x.shape[1] == in_features.
        Postconditions: Output is not softmax-normalised (raw logits).
        Failure Modes: Shape mismatch raises RuntimeError from nn.Linear.
        """
        return self.net(x)


def build_model(capacity: str = "medium") -> MLP:
    """
    ID: MODEL-004
    Purpose: Factory that returns a pre-configured MLP by capacity name.
    Inputs:
        capacity - one of {"tiny", "small", "medium", "large", "overfit"}
    Outputs: MLP instance.
    Preconditions: capacity must be a recognised key.
    Postconditions: Returns a ready-to-use nn.Module.
    Failure Modes: Unknown capacity raises ValueError.
    """
    configs: dict[str, dict] = {
        "tiny":    {"hidden_sizes": [8],            "dropout": 0.0},
        "small":   {"hidden_sizes": [32, 16],       "dropout": 0.0},
        "medium":  {"hidden_sizes": [64, 64],       "dropout": 0.1},
        "large":   {"hidden_sizes": [128, 128, 64], "dropout": 0.2},
        "overfit": {"hidden_sizes": [256, 256, 256, 128], "dropout": 0.0},
    }

    if capacity not in configs:
        raise ValueError(f"Unknown capacity '{capacity}'. Choose from {list(configs)}")

    return MLP(in_features=2, num_classes=2, **configs[capacity])
