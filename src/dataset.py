"""
Datasets, feature normalisation, and augmentation.

THE MOST IMPORTANT FIX IN THIS FILE: feature normalisation.

Raw MFCCs are wildly non-uniform in scale. Coefficient 0 encodes overall
loudness and typically has values in the hundreds; coefficients 1..n sit in the
tens or below. Feeding these in raw means one input channel is an order of
magnitude larger than all the others, so it dominates the gradients and the
network burns capacity fighting the scale difference instead of learning
acoustic structure. That alone can cap accuracy and destabilise training.

We therefore standardise EVERY channel to zero mean / unit variance — using
statistics computed from the TRAINING SPLIT ONLY. Computing them over the whole
dataset would leak validation information into training and inflate results.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def compute_norm_stats(features: np.ndarray, train_idx: np.ndarray):
    """
    Per-channel mean/std over the training split.

    features: (N, C, T)  ->  returns mean/std of shape (1, C, 1)
    """
    train_feats = features[train_idx]
    mean = train_feats.mean(axis=(0, 2), keepdims=True)[0:1]
    std = train_feats.std(axis=(0, 2), keepdims=True)[0:1]
    std = np.maximum(std, 1e-5)     # guard against constant channels
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RespiratoryDataset(Dataset):
    """
    Serves normalised features plus a frame-validity mask.

    branch = "signal" -> (C, T) MFCC+deltas, for the 1D model
    branch = "image"  -> (1, M, T) log-mel, for the 2D model
    """

    def __init__(self, features, labels, valid_frames, indices,
                 mean, std, cfg, is_train: bool, branch: str = "signal"):
        self.features = features
        self.labels = labels
        self.valid_frames = valid_frames
        self.indices = indices
        self.mean = mean
        self.std = std
        self.cfg = cfg
        self.is_train = is_train
        self.branch = branch

    def __len__(self):
        return len(self.indices)

    def _augment(self, x, n_valid):
        """Augmentation operates on NORMALISED features, so magnitudes are sane."""
        t = self.cfg.train

        if np.random.rand() < t.noise_prob:
            x = x + np.random.normal(0, t.noise_std, x.shape).astype(np.float32)

        # Time masking — restricted to the VALID region so we don't
        # "mask" padding that carries no information anyway.
        if np.random.rand() < t.time_mask_prob and n_valid > 4:
            width = max(1, int(n_valid * t.time_mask_frac))
            start = np.random.randint(0, max(1, n_valid - width))
            x[..., start:start + width] = 0.0

        # Frequency / coefficient masking
        if np.random.rand() < t.freq_mask_prob:
            n_ch = x.shape[0]
            width = max(1, int(n_ch * t.freq_mask_frac))
            start = np.random.randint(0, max(1, n_ch - width))
            x[start:start + width, :] = 0.0

        return x

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.features[idx].astype(np.float32).copy()

        # Standardise using train-split statistics
        x = (x - self.mean[0]) / self.std[0]

        n_valid = int(self.valid_frames[idx])

        if self.is_train:
            x = self._augment(x, n_valid)

        # Frame validity mask: True where the frame is real signal
        mask = np.zeros(x.shape[-1], dtype=bool)
        mask[:n_valid] = True

        x_t = torch.from_numpy(x)
        if self.branch == "image":
            x_t = x_t.unsqueeze(0)          # (1, M, T)

        return (
            x_t,
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.from_numpy(mask),
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def make_balanced_sampler(labels, train_idx, power: float, num_classes=4):
    """
    WeightedRandomSampler with a tunable balancing strength.

    We put the imbalance correction HERE rather than stacking heavy weights in
    the loss on top of unbalanced batches. With batch_size=64 and 'Both' at ~7%
    of data, unbalanced batches contain wildly varying numbers of minority
    samples, so a heavily weighted loss produces a gradient that swings
    direction batch to batch. Balancing the batches fixes the cause instead of
    rescaling the symptom.

    power = 0.0 -> natural distribution
    power = 1.0 -> fully balanced
    power = 0.5 -> square-root balancing (a good default; keeps some real prior)
    """
    y = labels[train_idx]
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)

    class_w = (1.0 / counts) ** power
    class_w = class_w / class_w.sum()

    sample_w = class_w[y]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------
def build_loaders(features, labels, valid_frames, train_idx, val_idx,
                  cfg, branch: str):
    mean, std = compute_norm_stats(features, train_idx)

    train_ds = RespiratoryDataset(features, labels, valid_frames, train_idx,
                                  mean, std, cfg, is_train=True, branch=branch)
    val_ds = RespiratoryDataset(features, labels, valid_frames, val_idx,
                                mean, std, cfg, is_train=False, branch=branch)

    sampler = make_balanced_sampler(labels, train_idx,
                                    cfg.train.sampler_power,
                                    cfg.train.num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, (mean, std)
