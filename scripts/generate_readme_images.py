# =============================================================================
# FILE: scripts/generate_readme_images.py
# PURPOSE: Generate PNG images for embedding in README.md.
# =============================================================================
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import generate_dataset
from model import build_model

OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
CMAP_DATA = ["#e05050", "#5080e0"]
DPI = 130

X, y = generate_dataset(n_samples=1000, noise=0.20, random_state=42)


def plot_raw_data() -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cls, col, lbl in [(0, CMAP_DATA[0], "Class 0 - lower moon"),
                          (1, CMAP_DATA[1], "Class 1 - upper moon")]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=col, s=16, alpha=0.75, linewidths=0, label=lbl)
    ax.set_title("make_moons - raw input data  (n=1000, noise=0.20)", fontsize=12)
    ax.set_xlabel("Feature x0  (horizontal position)")
    ax.set_ylabel("Feature x1  (vertical position)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = OUT / "moons_raw_data.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")


def _meshgrid(margin: float = 0.5, res: int = 300):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    return xx, yy, grid


def plot_boundary(ax, model, title: str, val_acc: float) -> None:
    xx, yy, grid = _meshgrid()
    model.eval()
    with torch.no_grad():
        t = torch.tensor(grid).to(DEVICE)
        probs = torch.softmax(model(t), dim=1)[:, 1].cpu().numpy()
    Z = probs.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.55, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=1.5, linestyles="--")
    for cls, col in [(0, CMAP_DATA[0]), (1, CMAP_DATA[1])]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=col, s=10, alpha=0.55, linewidths=0)
    ax.set_title(f"{title}\nval_acc = {val_acc:.1%}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


CHECKPOINTS = [
    ("epoch_0001.pt",  1, 0.9200),
    ("epoch_0012.pt", 12, 0.9650),
    ("epoch_0025.pt", 25, 0.9650),
    ("epoch_0037.pt", 37, 0.9650),
    ("epoch_0050.pt", 50, 0.9650),
]


def plot_progression() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    fig.suptitle("Decision boundary at each checkpoint - make_moons", fontsize=12, y=1.02)
    for ax, (fname, epoch, val_acc) in zip(axes, CHECKPOINTS):
        ck = ROOT / "checkpoints" / fname
        if not ck.exists():
            ax.set_title(f"Epoch {epoch}\n(checkpoint missing)")
            ax.axis("off")
            continue
        state = torch.load(ck, map_location=DEVICE, weights_only=True)
        model = build_model("medium").to(DEVICE)
        sd = state.get("state_dict", state.get("model_state_dict", state))
        model.load_state_dict(sd)
        plot_boundary(ax, model, f"Epoch {epoch}", val_acc)
    patch0 = mpatches.Patch(color=CMAP_DATA[0], label="Class 0")
    patch1 = mpatches.Patch(color=CMAP_DATA[1], label="Class 1")
    fig.legend(handles=[patch0, patch1], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = OUT / "decision_boundary_progression.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_individual_checkpoints() -> None:
    for fname, epoch, val_acc in CHECKPOINTS:
        ck = ROOT / "checkpoints" / fname
        if not ck.exists():
            print(f"  WARNING: {ck} not found, skipping")
            continue
        state = torch.load(ck, map_location=DEVICE, weights_only=True)
        model = build_model("medium").to(DEVICE)
        sd = state.get("state_dict", state.get("model_state_dict", state))
        model.load_state_dict(sd)
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_boundary(ax, model, f"Epoch {epoch}", val_acc)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        fig.tight_layout()
        path = OUT / f"boundary_epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  saved {path}")


def plot_annotated_sample() -> None:
    idx0 = np.where(y == 0)[0][42]
    idx1 = np.where(y == 1)[0][17]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cls, col in [(0, CMAP_DATA[0]), (1, CMAP_DATA[1])]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=col, s=14, alpha=0.35, linewidths=0)
    for idx, col, lbl in [
        (idx0, CMAP_DATA[0], f"x=[{X[idx0, 0]:.2f}, {X[idx0, 1]:.2f}]  y=0"),
        (idx1, CMAP_DATA[1], f"x=[{X[idx1, 0]:.2f}, {X[idx1, 1]:.2f}]  y=1"),
    ]:
        ax.scatter(X[idx, 0], X[idx, 1], c=col, s=120, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(lbl, xy=(X[idx, 0], X[idx, 1]),
                    xytext=(X[idx, 0] + 0.25, X[idx, 1] + 0.15),
                    fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.set_title("Single sample highlighted - (x0, x1) coordinates fed into the model\n"
                 "and true label y used to compute the loss", fontsize=10)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = OUT / "moons_annotated_sample.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating README images...")
    plot_raw_data()
    plot_annotated_sample()
    plot_progression()
    plot_individual_checkpoints()
    print(f"\nDone - {len(list(OUT.glob('*.png')))} PNG files in {OUT}")
