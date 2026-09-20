"""
Train / validation splitting.

THE SINGLE MOST IMPORTANT FILE IN THIS REPO for getting honest numbers.

Why patient-wise splitting is non-negotiable
--------------------------------------------
Each ICBHI recording contributes many breathing cycles. If you split at the
CYCLE level (the default `train_test_split` / `random_split` behaviour), cycles
from the same patient — same microphone, same room, same background hum, same
chest — appear in both training and validation.

The model then gets a free shortcut: instead of learning what a crackle sounds
like, it learns to recognise *this particular recording's fingerprint* and
recall the labels it memorised for it. Validation accuracy goes up. Real
performance does not. Published ICBHI results that look implausibly high are
very often this bug.

Splitting by PATIENT closes the shortcut. Expect your numbers to DROP when you
switch to this — that drop is the leakage being removed, not a regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def official_split(filenames: np.ndarray,
                   split_file: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Use the ICBHI challenge's own train/test list if it is available.

    The file `ICBHI_challenge_train_test.txt` has lines like:
        101_1b1_Al_sc_Meditron    train
        103_2b2_Ar_mc_LittC2SE    test

    Using it makes our results directly comparable to published papers. Returns
    None if the file is missing so the caller can fall back.
    """
    split_file = Path(split_file)
    if not split_file.exists():
        return None

    mapping = {}
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1].lower()

    if not mapping:
        return None

    stems = np.array([Path(f).stem for f in filenames])
    assign = np.array([mapping.get(s, "") for s in stems])

    train_idx = np.where(assign == "train")[0]
    test_idx = np.where(assign == "test")[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    unmatched = int(np.sum(assign == ""))
    if unmatched:
        print(f"  ! {unmatched} cycles had no entry in the official split list "
              f"and were dropped from both sets.")

    return train_idx, test_idx


def patient_wise_split(patient_ids: np.ndarray,
                       labels: np.ndarray,
                       val_fraction: float = 0.2,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback split: hold out whole patients, chosen so the validation set's
    class distribution stays as close as possible to the overall distribution.

    A naive random patient draw can produce a validation set with almost no
    'Both' cases (that class is only ~7% of data and clusters within patients),
    which makes the metric extremely noisy. We draw several candidate splits and
    keep the one whose class distribution best matches the full dataset.
    """
    rng = np.random.default_rng(seed)
    uniq_patients = np.unique(patient_ids)
    n_val_patients = max(1, int(round(len(uniq_patients) * val_fraction)))

    overall = np.bincount(labels, minlength=4).astype(float)
    overall /= overall.sum()

    best, best_cost = None, np.inf
    for _ in range(300):
        perm = rng.permutation(uniq_patients)
        val_patients = set(perm[:n_val_patients].tolist())

        val_mask = np.isin(patient_ids, list(val_patients))
        if val_mask.sum() == 0 or (~val_mask).sum() == 0:
            continue

        val_dist = np.bincount(labels[val_mask], minlength=4).astype(float)
        if val_dist.min() == 0:      # a class entirely missing -> unusable
            continue
        val_dist /= val_dist.sum()

        # Cost: distribution mismatch + deviation from target split size
        size_pen = abs(val_mask.mean() - val_fraction) * 2.0
        cost = np.abs(val_dist - overall).sum() + size_pen

        if cost < best_cost:
            best_cost, best = cost, val_mask

    if best is None:
        raise RuntimeError("Could not build a valid patient-wise split.")

    val_idx = np.where(best)[0]
    train_idx = np.where(~best)[0]
    return train_idx, val_idx


def get_split(filenames, patient_ids, labels, split_file,
              val_fraction=0.2, seed=42, force_patient_split=False):
    """
    Returns (train_idx, val_idx, split_name).
    Prefers the official ICBHI split; falls back to our own patient-wise split.
    """
    if not force_patient_split:
        res = official_split(filenames, split_file)
        if res is not None:
            print("  split: OFFICIAL ICBHI patient-wise train/test list")
            return res[0], res[1], "official"

    print("  split: generated patient-wise split "
          "(official list not found or overridden)")
    tr, va = patient_wise_split(patient_ids, labels, val_fraction, seed)
    return tr, va, "patient-wise"


def describe_split(train_idx, val_idx, labels, patient_ids, class_names):
    """Print a summary so leakage is obvious if it ever creeps back in."""
    tr_p = set(patient_ids[train_idx].tolist())
    va_p = set(patient_ids[val_idx].tolist())
    overlap = tr_p & va_p

    print(f"  train: {len(train_idx):5d} cycles from {len(tr_p):3d} patients")
    print(f"  val  : {len(val_idx):5d} cycles from {len(va_p):3d} patients")

    if overlap:
        print(f"  *** WARNING: {len(overlap)} patients appear in BOTH splits. "
              f"Results will be optimistic. ***")
    else:
        print("  patient overlap: none (good)")

    for name, idx in (("train", train_idx), ("val", val_idx)):
        cnt = np.bincount(labels[idx], minlength=len(class_names))
        dist = ", ".join(f"{class_names[i]}={cnt[i]}" for i in range(len(cnt)))
        print(f"  {name} distribution: {dist}")
