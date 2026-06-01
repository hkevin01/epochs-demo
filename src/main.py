# =============================================================================
# FILE: src/main.py
# PURPOSE: Orchestration entry-point for the epochs-demo project.
# =============================================================================
#
# ID: MAIN-001
# Requirement: Execute the full pipeline - data generation, model training,
#              evaluation, and visualisation - from a single command.
# Purpose: Provide a one-shot runnable script that demonstrates epoch effects.
# Rationale: Separating orchestration from logic keeps each module testable.
# Inputs: Command-line arguments (see argparse setup below).
# Outputs: Trained model checkpoints, history JSON, and PNG/GIF plots.
# Preconditions: All dependencies installed (pip install -r requirements.txt).
# Postconditions: outputs/ and checkpoints/ populated.
# Assumptions: Python >= 3.10.
# Side Effects: Writes files to disk; prints progress to stdout.
# Failure Modes: Missing packages raise ImportError with clear pip message.
# Error Handling: Argparse validates CLI inputs; propagates other errors.
# Constraints: None.
# Verification: Run `python src/main.py` and inspect outputs/.
# References: All src/* modules.
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
import glob

# Allow imports from src/ when running as `python src/main.py`
sys.path.insert(0, os.path.dirname(__file__))

from data      import get_dataloaders
from model     import build_model
from train     import train
from evaluate  import evaluate
from visualize import (
    plot_loss_curve,
    plot_accuracy_curve,
    plot_decision_boundary,
    plot_epoch_snapshots,
    animate_training,
)
from utils import seed_everything, get_device, save_history


def parse_args() -> argparse.Namespace:
    """
    ID: MAIN-002
    Purpose: Parse and validate command-line arguments.
    Outputs: argparse.Namespace with all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Epochs Demo - visualise the effect of training epochs."
    )
    parser.add_argument("--epochs",      type=int,   default=200,    help="Total training epochs.")
    parser.add_argument("--batch-size",  type=int,   default=64,     help="Mini-batch size.")
    parser.add_argument("--lr",          type=float, default=1e-2,   help="Learning rate.")
    parser.add_argument("--noise",       type=float, default=0.20,   help="Dataset noise level.")
    parser.add_argument("--n-samples",   type=int,   default=1000,   help="Dataset size.")
    parser.add_argument("--capacity",    type=str,   default="medium",
                        choices=["tiny","small","medium","large","overfit"],
                        help="Model capacity preset.")
    parser.add_argument("--seed",        type=int,   default=42,     help="RNG seed.")
    parser.add_argument("--output-dir",  type=str,   default="outputs",     help="Plot output directory.")
    parser.add_argument("--ckpt-dir",    type=str,   default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--animate",     action="store_true", help="Generate GIF animation.")
    parser.add_argument("--show-plots",  action="store_true", help="Display plots interactively.")
    return parser.parse_args()


def main() -> None:
    """
    ID: MAIN-003
    Purpose: Run full training + evaluation + visualisation pipeline.
    Inputs:  CLI arguments parsed by parse_args().
    Outputs: Files written to --output-dir and --ckpt-dir.
    Side Effects: Prints progress; writes disk files.
    Failure Modes: ImportError if dependencies missing.
    """
    args = parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}  |  Capacity: {args.capacity}  |  LR: {args.lr}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, X_all, y_all = get_dataloaders(
        n_samples=args.n_samples,
        noise=args.noise,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    print(f"Dataset: {args.n_samples} samples  |  Noise: {args.noise}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(capacity=args.capacity).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.capacity} MLP  |  Parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.ckpt_dir,
        verbose=True,
    )

    save_history(history, output_dir=args.output_dir)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    evaluate(model, val_loader, device)

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    loss_path = plot_loss_curve(history, output_dir=args.output_dir, show=args.show_plots)
    acc_path  = plot_accuracy_curve(history, output_dir=args.output_dir, show=args.show_plots)
    db_path   = plot_decision_boundary(model, X_all, y_all, epoch=args.epochs,
                                       device=device, output_dir=args.output_dir,
                                       show=args.show_plots)

    checkpoint_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "epoch_*.pt")))
    if checkpoint_files:
        snap_path = plot_epoch_snapshots(model, checkpoint_files, X_all, y_all,
                                         device=device, output_dir=args.output_dir,
                                         show=args.show_plots)
        print(f"Epoch snapshots: {snap_path}")

        if args.animate:
            gif_path = animate_training(model, checkpoint_files, X_all, y_all,
                                        device=device, output_dir=args.output_dir)
            if gif_path:
                print(f"Animation:       {gif_path}")

    print("\n" + "="*55)
    print("OUTPUTS")
    print("="*55)
    print(f"Loss curve:       {loss_path}")
    print(f"Accuracy curve:   {acc_path}")
    print(f"Decision boundary:{db_path}")
    print(f"History JSON:     {os.path.join(args.output_dir, 'history.json')}")
    print("="*55)


if __name__ == "__main__":
    main()
