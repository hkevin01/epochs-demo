# =============================================================================
# FILE: src/evaluate.py
# PURPOSE: Model evaluation utilities - accuracy, loss, classification report.
# =============================================================================
#
# ID: EVAL-001
# Requirement: Compute loss, accuracy, and per-class metrics on any DataLoader.
# Purpose: Separate evaluation from training for clean separation of concerns.
# Rationale: Allows post-hoc evaluation of checkpointed models without
#            re-running the full training loop.
# Inputs: model, DataLoader, device.
# Outputs: dict of scalar metrics; prints classification report.
# Preconditions: model loaded and on correct device.
# Postconditions: Returned dict contains "loss" and "accuracy" keys.
# Assumptions: Binary classification (num_classes=2).
# Side Effects: Prints to stdout.
# Failure Modes: RuntimeError on device mismatch.
# Error Handling: Propagates torch exceptions.
# Constraints: None.
# Verification: Verified by inspecting printed report in main.py.
# References: sklearn.metrics.classification_report.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str] | None = None,
) -> dict:
    """
    ID: EVAL-002
    Purpose: Evaluate model on loader, return metrics dict, print full report.
    Inputs:
        model       - trained nn.Module.
        loader      - DataLoader for evaluation split.
        device      - torch.device.
        label_names - list of class name strings (optional).
    Outputs:
        metrics - dict with keys: "loss" (float), "accuracy" (float),
                  "confusion_matrix" (np.ndarray), "report" (str).
    Preconditions: model in eval mode is handled internally.
    Postconditions: Metrics reflect full dataset in loader.
    Failure Modes: Empty loader returns zeros.
    """
    if label_names is None:
        label_names = ["Class 0", "Class 1"]

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    all_preds: list[int]  = []
    all_labels: list[int] = []
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1).cpu().numpy()
            labels      = y_batch.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            n_samples += len(y_batch)

    if n_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "confusion_matrix": np.zeros((2,2)), "report": ""}

    mean_loss = total_loss / n_samples
    report    = classification_report(all_labels, all_preds, target_names=label_names)
    cm        = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Loss:     {mean_loss:.4f}")
    print(f"Accuracy: {sum(p==l for p,l in zip(all_preds,all_labels))/n_samples:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("="*50)

    return {
        "loss":             mean_loss,
        "accuracy":         sum(p == l for p, l in zip(all_preds, all_labels)) / n_samples,
        "confusion_matrix": cm,
        "report":           report,
    }
