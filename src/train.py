"""
Train a single branch.

PATHS USED BY THIS SCRIPT (all set at the top of src/config.py):
    reads  <- PROCESSED_FILE    written by `python -m src.preprocess`
    reads  <- DATASET_DIR       only for ICBHI_challenge_train_test.txt
    writes -> OUTPUT_DIR/checkpoints/<run>.pth
    writes -> OUTPUT_DIR/reports/<run>_report.txt, <run>_history.json
    writes -> OUTPUT_DIR/logits/<run>_val.npz     (consumed by src.ensemble)

Override the data path without editing config.py:
    python -m src.train --branch signal --data "E:/processed_icbhi.npz"

Usage
-----
    python -m src.train --branch signal
    python -m src.train --branch image
    python -m src.train --branch image --backbone resnet34

Both branches share the same split, the same engine, and the same metrics, so
their results are directly comparable and their logits can be fused later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import get_config
from .dataset import build_loaders
from .engine import set_seed, train_model, evaluate, save_run
from .losses import make_class_weights
from .metrics import full_report
from .models import SignalBranch, build_image_branch
from .splits import get_split, describe_split


def parse_args():
    ap = argparse.ArgumentParser(description="Train one branch")
    ap.add_argument("--branch", choices=["signal", "image"], required=True)
    ap.add_argument("--backbone", default="scratch",
                    help="image branch only: scratch | resnet18 | resnet34")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", default=None, help="path to the .npz archive")
    ap.add_argument("--force-patient-split", action="store_true",
                    help="ignore the official ICBHI list and generate our own "
                         "patient-wise split")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    # CLI overrides
    if args.epochs:      cfg.train.epochs = args.epochs
    if args.batch_size:  cfg.train.batch_size = args.batch_size
    if args.lr:          cfg.train.lr = args.lr
    if args.seed:        cfg.train.seed = args.seed
    if args.data:        cfg.paths.processed_file = Path(args.data)

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = Path(cfg.paths.processed_file)
    if not npz_path.exists():
        raise SystemExit(
            f"Processed data not found at {npz_path}\n"
            f"Run:  python -m src.preprocess"
        )

    print(f"Loading {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)
    labels = data["labels"]
    patient_ids = data["patient_ids"]
    filenames = data["filenames"]
    valid_frames = data["valid_frames"]

    key = "signal_features" if args.branch == "signal" else "image_features"
    features = data[key]
    print(f"  {key}: {features.shape}")

    # ---- Split (patient-wise; see src/splits.py for why this matters) ----
    print("\nBuilding split...")
    train_idx, val_idx, split_name = get_split(
        filenames, patient_ids, labels,
        cfg.paths.official_split_file,
        cfg.train.val_fraction, cfg.train.seed,
        force_patient_split=args.force_patient_split,
    )
    describe_split(train_idx, val_idx, labels, patient_ids,
                   cfg.train.class_names)

    # ---- Loaders (normalisation stats come from the train split only) ----
    train_loader, val_loader, norm_stats = build_loaders(
        features, labels, valid_frames, train_idx, val_idx, cfg, args.branch
    )

    # ---- Model ----
    if args.branch == "signal":
        model = SignalBranch(num_classes=cfg.train.num_classes,
                             in_channels=features.shape[1])
        run_name = args.run_name or "signal_branch"
    else:
        model = build_image_branch(num_classes=cfg.train.num_classes,
                                   backbone=args.backbone)
        run_name = args.run_name or f"image_branch_{args.backbone}"

    class_weights = make_class_weights(labels, train_idx,
                                       cfg.train.num_classes,
                                       power=0.25, device=device)

    best_metrics, best_state, history = train_model(
        model, train_loader, val_loader, cfg, device,
        class_weights, run_name
    )

    # ---- Final report from the best checkpoint ----
    model.load_state_dict(best_state)
    metrics, logits, targets = evaluate(model, val_loader, device,
                                        cfg.train.class_names,
                                        return_logits=True)
    preds = logits.argmax(axis=1)
    report = full_report(targets, preds, cfg.train.class_names)
    print("\n" + report)

    save_run(cfg.paths.output_dir, run_name, best_state, metrics,
             history, norm_stats, report)

    # Cache validation logits so the ensemble can fuse without retraining
    out = Path(cfg.paths.output_dir) / "logits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{run_name}_val.npz",
             logits=logits, targets=targets, val_idx=val_idx)

    print(f"\nSaved checkpoint, report, and cached logits under "
          f"{Path(cfg.paths.output_dir).resolve()}")


if __name__ == "__main__":
    main()
