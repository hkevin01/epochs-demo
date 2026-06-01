# =============================================================================
# FILE: src/visualize.py
# PURPOSE: All plotting functions - curves, decision boundaries, animations.
# =============================================================================
#
# ID: VIZ-001
# Requirement: Produce loss curves, accuracy curves, and decision-boundary
#              plots from training history and checkpointed models.
# Purpose: Make the effect of epochs visually obvious.
# Rationale: A learner can see underfitting, learning, convergence, and
#            overfitting in a single figure rather than reading numbers.
# Inputs: history dict; model + checkpoint paths; dataset arrays.
# Outputs: PNG files saved to outputs/; optionally GIF animation.
# Preconditions: matplotlib installed; outputs/ dir exists.
# Postconditions: Files written to disk; file paths returned.
# Assumptions: 2-D feature space only for decision boundary plots.
# Side Effects: Writes PNG/GIF files; shows plots if show=True.
# Failure Modes: Missing checkpoint files raise FileNotFoundError.
# Error Handling: Skips missing checkpoints with a warning.
# Constraints: None.
# Verification: Visual inspection of generated plots.
# References: matplotlib.pyplot, matplotlib.animation.
# =============================================================================

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_grid(X: np.ndarray, resolution: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ID: VIZ-002
    Purpose: Build a meshgrid covering the feature space of X.
    Inputs:
        X          - (N, 2) feature array.
        resolution - number of grid points per axis (int, >0).
    Outputs:
        xx, yy     - meshgrid arrays.
        grid       - (resolution^2, 2) array of grid points.
    """
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    return xx, yy, grid


def _predict_grid(model: nn.Module, grid: np.ndarray, device: torch.device) -> np.ndarray:
    """
    ID: VIZ-003
    Purpose: Run model on meshgrid points and return class probabilities.
    Inputs:
        model  - trained nn.Module.
        grid   - (M, 2) float32 numpy array.
        device - torch.device.
    Outputs: (M,) numpy array of P(class=1) values in [0, 1].
    Preconditions: model in eval mode after call.
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(grid, dtype=torch.float32).to(device)
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_loss_curve(
    history: dict,
    output_dir: str = "outputs",
    show: bool = False,
) -> str:
    """
    ID: VIZ-004
    Purpose: Plot train and validation loss vs. epoch.
    Inputs:
        history    - dict with "train_loss" and "val_loss" lists.
        output_dir - directory to save the PNG (str).
        show       - if True, call plt.show() (bool).
    Outputs: path (str) to saved PNG.
    Side Effects: Writes PNG; optionally opens display window.
    Failure Modes: KeyError if history lacks expected keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss",      color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Validation loss", color="#F44336", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss Curve - How Loss Evolves Across Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate phases
    n = len(epochs)
    ax.axvspan(1,     max(2, n//5),       alpha=0.08, color="red",    label="Underfitting")
    ax.axvspan(max(2, n//5), max(3, n//2),alpha=0.08, color="yellow", label="Learning")
    ax.axvspan(max(3, n//2), n,           alpha=0.08, color="green",  label="Convergence/Overfit")

    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_accuracy_curve(
    history: dict,
    output_dir: str = "outputs",
    show: bool = False,
) -> str:
    """
    ID: VIZ-005
    Purpose: Plot train and validation accuracy vs. epoch.
    Inputs:
        history    - dict with "train_acc" and "val_acc" lists.
        output_dir - directory to save the PNG (str).
        show       - if True, call plt.show() (bool).
    Outputs: path (str) to saved PNG.
    Side Effects: Writes PNG; optionally opens display window.
    Failure Modes: KeyError if history lacks expected keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train accuracy",      color="#4CAF50", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Validation accuracy", color="#FF9800", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curve - How Accuracy Evolves Across Epochs")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "accuracy_curve.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_decision_boundary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epoch: int,
    device: torch.device,
    output_dir: str = "outputs",
    show: bool = False,
) -> str:
    """
    ID: VIZ-006
    Purpose: Plot the model decision boundary overlaid on the dataset scatter.
    Inputs:
        model      - trained nn.Module (weights at desired epoch).
        X          - (N, 2) feature array.
        y          - (N,) label array {0, 1}.
        epoch      - epoch number for title annotation (int).
        device     - torch.device.
        output_dir - save directory (str).
        show       - display window flag (bool).
    Outputs: path (str) to saved PNG.
    Side Effects: Writes PNG.
    Failure Modes: Requires 2-D features; raises if X.shape[1] != 2.
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plots require exactly 2 input features.")

    os.makedirs(output_dir, exist_ok=True)
    xx, yy, grid = _make_grid(X)
    Z = _predict_grid(model, grid, device).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
    plt.colorbar(contour, ax=ax, label="P(class=1)")

    colors = ["#2196F3", "#F44336"]
    for cls in (0, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=20,
                   edgecolors="white", linewidths=0.4, label=f"Class {cls}", alpha=0.9)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"Decision Boundary at Epoch {epoch}")
    ax.legend(markerscale=1.5)
    fig.tight_layout()

    path = os.path.join(output_dir, f"boundary_epoch_{epoch:04d}.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_epoch_snapshots(
    model: nn.Module,
    checkpoint_paths: list[str],
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    output_dir: str = "outputs",
    show: bool = False,
) -> str:
    """
    ID: VIZ-007
    Purpose: Create a multi-panel figure showing decision boundaries at several
             saved epochs side by side to show training progression.
    Inputs:
        model            - nn.Module (weights will be overwritten per checkpoint).
        checkpoint_paths - ordered list of .pt file paths.
        X                - (N, 2) feature array.
        y                - (N,) label array.
        device           - torch.device.
        output_dir       - save directory (str).
        show             - display window flag (bool).
    Outputs: path (str) to saved PNG.
    Side Effects: Mutates model.state_dict() temporarily; writes PNG.
    Failure Modes: Missing checkpoint files raise FileNotFoundError.
    """
    from utils import load_checkpoint

    valid_paths = []
    for p in checkpoint_paths:
        if os.path.exists(p):
            valid_paths.append(p)
        else:
            warnings.warn(f"Checkpoint not found, skipping: {p}")

    if not valid_paths:
        raise FileNotFoundError("No valid checkpoints found.")

    n_panels = len(valid_paths)
    n_cols   = min(n_panels, 3)
    n_rows   = (n_panels + n_cols - 1) // n_cols

    xx, yy, grid = _make_grid(X)
    colors = ["#2196F3", "#F44336"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for idx, path in enumerate(valid_paths):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        epoch = ckpt["epoch"]

        Z = _predict_grid(model, grid, device).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.7)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.2)

        for cls in (0, 1):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=12,
                       edgecolors="white", linewidths=0.3, alpha=0.85)

        metrics = ckpt.get("metrics", {})
        vl = metrics.get("val_loss", float("nan"))
        va = metrics.get("val_acc",  float("nan"))
        ax.set_title(f"Epoch {epoch}\nVal Loss={vl:.3f}  Acc={va:.1%}", fontsize=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    # Hide any unused panels
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Decision Boundary Evolution Across Epochs", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path_out = os.path.join(output_dir, "epoch_snapshots.png")
    fig.savefig(path_out)
    if show:
        plt.show()
    plt.close(fig)
    return path_out


def animate_training(
    model: nn.Module,
    checkpoint_paths: list[str],
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    output_dir: str = "outputs",
    fps: int = 4,
) -> str:
    """
    ID: VIZ-008
    Purpose: Create an animated GIF showing how the decision boundary evolves
             epoch by epoch across all saved checkpoints.
    Inputs:
        model            - nn.Module (weights mutated per frame).
        checkpoint_paths - ordered list of .pt checkpoint paths.
        X                - (N, 2) feature array.
        y                - (N,) label array.
        device           - torch.device.
        output_dir       - save directory (str).
        fps              - frames per second for the GIF (int, >0).
    Outputs: path (str) to saved GIF.
    Side Effects: Mutates model weights; writes GIF file.
    Failure Modes: PillowWriter requires Pillow; falls back to warning.
    """
    from utils import load_checkpoint

    valid_paths = sorted(
        [p for p in checkpoint_paths if os.path.exists(p)],
        key=lambda p: int(Path(p).stem.split("_")[-1]),
    )

    if not valid_paths:
        raise FileNotFoundError("No valid checkpoints for animation.")

    xx, yy, grid = _make_grid(X)
    colors = ["#2196F3", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 5))

    def _frame(path: str):
        ax.clear()
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        epoch = ckpt["epoch"]

        Z = _predict_grid(model, grid, device).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.7)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.2)

        for cls in (0, 1):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=12,
                       edgecolors="white", linewidths=0.3, alpha=0.85)

        metrics = ckpt.get("metrics", {})
        vl = metrics.get("val_loss", float("nan"))
        va = metrics.get("val_acc",  float("nan"))
        ax.set_title(f"Epoch {epoch}  |  Val Loss={vl:.3f}  Acc={va:.1%}", fontsize=11)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    anim = animation.FuncAnimation(fig, _frame, frames=valid_paths, repeat=True)

    os.makedirs(output_dir, exist_ok=True)
    path_out = os.path.join(output_dir, "training_animation.gif")
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(path_out, writer=writer)
    except Exception as exc:
        warnings.warn(f"GIF export failed ({exc}). Pillow may not be installed.")
        path_out = ""

    plt.close(fig)
    return path_out
