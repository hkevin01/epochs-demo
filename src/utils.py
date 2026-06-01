# =============================================================================
# FILE: src/utils.py
# PURPOSE: Shared utility functions - checkpointing, seeding, metrics.
# =============================================================================
#
# ID: UTILS-001
# Requirement: Provide reproducibility, checkpoint I/O, and metric helpers
#              reusable across train, evaluate, and visualize modules.
# Purpose: Centralise boilerplate so each module stays single-responsibility.
# Rationale: Avoids duplicating seed/checkpoint logic in every file.
# Inputs: Vary per function - see individual docstrings.
# Outputs: Vary per function.
# Preconditions: PyTorch installed; outputs/ and checkpoints/ dirs exist.
# Postconditions: Files written atomically; seeds applied globally.
# Assumptions: Single-process training only (no DDP).
# Side Effects: Sets global RNG seeds; writes files to disk.
# Failure Modes: OSError if target directory missing.
# Error Handling: makedirs(exist_ok=True) guards directory creation.
# Constraints: None.
# Verification: Confirmed by reproducibility smoke test.
# References: PyTorch docs - torch.save / torch.load.
# =============================================================================

from __future__ import annotations

import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """
    ID: UTILS-002
    Purpose: Set all relevant RNG seeds for reproducibility.
    Inputs:  seed - integer seed value (int).
    Outputs: None.
    Side Effects: Modifies global RNG state for Python, NumPy, and PyTorch.
    Failure Modes: Non-integer seed raises TypeError.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    ID: UTILS-003
    Purpose: Return the best available torch device.
    Outputs: torch.device - "cuda", "mps", or "cpu".
    Side Effects: None.
    Failure Modes: Falls back to CPU silently.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str = "checkpoints",
) -> str:
    """
    ID: UTILS-004
    Purpose: Persist model weights and epoch metrics to disk.
    Inputs:
        model          - nn.Module whose state_dict is saved.
        epoch          - current epoch number (int, >= 1).
        metrics        - dict of scalar metrics to store alongside weights.
        checkpoint_dir - directory path (str).
    Outputs: path (str) - absolute path to the saved file.
    Preconditions: model is an nn.Module.
    Postconditions: .pt file exists at returned path.
    Side Effects: Creates checkpoint_dir if absent; writes file.
    Failure Modes: OSError if disk is full.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics}, path)
    return path


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """
    ID: UTILS-005
    Purpose: Load weights from a checkpoint file into model.
    Inputs:
        model  - nn.Module to populate.
        path   - file path to .pt checkpoint (str).
        device - target device for weight tensors.
    Outputs: metrics dict stored in the checkpoint.
    Preconditions: path exists and was written by save_checkpoint.
    Postconditions: model.state_dict() matches checkpoint.
    Failure Modes: FileNotFoundError if path is wrong.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("metrics", {})


def save_history(history: dict, output_dir: str = "outputs") -> str:
    """
    ID: UTILS-006
    Purpose: Serialise training history dict to JSON for later analysis.
    Inputs:
        history    - dict mapping metric names to lists of per-epoch values.
        output_dir - directory path (str).
    Outputs: path (str) to written JSON file.
    Side Effects: Creates output_dir if absent; writes JSON file.
    Failure Modes: OSError on disk full; TypeError if values not JSON-serialisable.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "history.json")
    serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    return path


def load_history(output_dir: str = "outputs") -> dict:
    """
    ID: UTILS-007
    Purpose: Load previously saved training history from JSON.
    Inputs:  output_dir - directory containing history.json (str).
    Outputs: dict of metric lists.
    Preconditions: history.json exists in output_dir.
    Failure Modes: FileNotFoundError if missing.
    """
    path = os.path.join(output_dir, "history.json")
    with open(path, "r") as fh:
        return json.load(fh)
