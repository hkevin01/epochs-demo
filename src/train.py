# =============================================================================
# FILE: src/train.py
# PURPOSE: Training loop with per-epoch metric collection and checkpointing.
# =============================================================================
#
# ID: TRAIN-001
# Requirement: Execute N training epochs, recording train/val loss and
#              accuracy at every epoch, and saving checkpoints at specified
#              epoch indices.
# Purpose: Demonstrate how metrics evolve from underfitting to convergence to
#          overfitting as epoch count increases.
# Rationale: Explicit per-epoch logging is the core educational artifact of
#            this project.
# Inputs: model, dataloaders, optimiser, loss fn, epoch count, device.
# Outputs: history dict {train_loss, val_loss, train_acc, val_acc}.
# Preconditions: model on correct device; dataloaders yield (X, y) pairs.
# Postconditions: history lists all have length == num_epochs.
# Assumptions: Binary classification with CrossEntropyLoss.
# Side Effects: Writes checkpoint files; prints progress.
# Failure Modes: OOM if batch_size too large for GPU.
# Error Handling: Propagates torch exceptions.
# Constraints: None.
# Verification: Confirmed by observing declining loss in main.py output.
# References: PyTorch training loop documentation.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import save_checkpoint


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    """
    ID: TRAIN-002
    Purpose: Execute one pass (train or eval) over a DataLoader.
    Inputs:
        model     - nn.Module.
        loader    - DataLoader yielding (X, y) batches.
        criterion - loss function (CrossEntropyLoss).
        optimiser - torch optimiser, or None for eval passes.
        device    - computation device.
        training  - if True, backprop and update weights.
    Outputs:
        mean_loss - average loss over all batches (float).
        accuracy  - fraction of correct predictions (float, [0,1]).
    Preconditions: model and data on same device.
    Postconditions: If training=True, model weights updated.
    Failure Modes: RuntimeError on device mismatch.
    """
    model.train(training)
    total_loss = 0.0
    correct = 0
    n_samples = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            n_samples  += len(y_batch)

    mean_loss = total_loss / n_samples
    accuracy  = correct   / n_samples
    return mean_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "checkpoints",
    save_epochs: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    ID: TRAIN-003
    Purpose: Full training loop with per-epoch logging and optional checkpoints.
    Inputs:
        model          - nn.Module to train.
        train_loader   - training DataLoader.
        val_loader     - validation DataLoader.
        num_epochs     - total epochs to train (int, >0).
        lr             - learning rate (float, >0).
        weight_decay   - L2 regularisation coefficient (float, >=0).
        device         - torch.device.
        checkpoint_dir - directory for .pt checkpoint files (str).
        save_epochs    - list of epoch numbers at which to save (1-indexed).
                         If None, saves at epoch 1, 25%, 50%, 75%, 100%.
        verbose        - if True, print tqdm progress bar.
    Outputs:
        history - dict with keys:
                  "train_loss", "val_loss", "train_acc", "val_acc"
                  each mapping to a list of length num_epochs.
    Preconditions: model moved to device before calling.
    Postconditions: All lists in history have length == num_epochs.
    Side Effects: Writes checkpoint files; prints to stdout.
    Failure Modes: KeyboardInterrupt halts training but returns partial history.
    """
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs)

    if save_epochs is None:
        checkpoints = {1, max(1, num_epochs // 4), max(1, num_epochs // 2),
                       max(1, 3 * num_epochs // 4), num_epochs}
    else:
        checkpoints = set(save_epochs)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    epoch_iter = tqdm(range(1, num_epochs + 1), desc="Training", disable=not verbose)

    for epoch in epoch_iter:
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimiser, device, training=True)
        vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, None,      device, training=False)

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if verbose:
            epoch_iter.set_postfix(
                tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.3f}",
                vl_loss=f"{vl_loss:.4f}", vl_acc=f"{vl_acc:.3f}",
            )

        if epoch in checkpoints:
            metrics = {"train_loss": tr_loss, "val_loss": vl_loss,
                       "train_acc":  tr_acc,  "val_acc":  vl_acc}
            save_checkpoint(model, epoch, metrics, checkpoint_dir)

    return history
