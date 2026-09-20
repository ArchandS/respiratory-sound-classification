"""Loss functions and mixup."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss with optional class weights and label smoothing.

    gamma down-weights examples the model already classifies confidently, so
    training effort concentrates on hard cases. Note we keep gamma MILD
    (default 1.0, not the 2.0 used in the original runs): the batch sampler is
    already doing the imbalance correction, and stacking an aggressive gamma on
    top of balanced batches over-focuses on genuinely ambiguous / mislabelled
    cycles, of which ICBHI has plenty (annotator agreement on this dataset is
    far from perfect).
    """

    def __init__(self, alpha=None, gamma: float = 1.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        if self.gamma == 0:
            return ce.mean()
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        return (((1 - pt) ** self.gamma) * ce).mean()


class SoftTargetFocalLoss(nn.Module):
    """Focal loss variant that accepts soft targets (needed for mixup)."""

    def __init__(self, alpha=None, gamma: float = 1.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma

    def forward(self, logits, soft_targets):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        focal = (1 - p) ** self.gamma

        if self.alpha is not None:
            focal = focal * self.alpha.unsqueeze(0)

        loss = -(soft_targets * focal * logp).sum(dim=-1)
        return loss.mean()


def mixup_batch(x, y, num_classes, alpha: float, smoothing: float = 0.0):
    """
    Mixup: blend pairs of samples and their labels.

    On a small, noisily-labelled dataset like ICBHI this is one of the more
    reliable regularisers — it discourages the model from becoming overconfident
    about patient-specific artefacts, which is exactly our overfitting mode.

    Returns (mixed_x, soft_targets).
    """
    y_soft = F.one_hot(y, num_classes).float()
    if smoothing > 0:
        y_soft = y_soft * (1 - smoothing) + smoothing / num_classes

    if alpha <= 0:
        return x, y_soft

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)          # keep the dominant sample dominant

    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    mixed_y = lam * y_soft + (1 - lam) * y_soft[perm]
    return mixed_x, mixed_y


def make_class_weights(labels, train_idx, num_classes, power: float,
                       device) -> torch.Tensor:
    """
    MILD residual class weights for the loss.

    The sampler already balances batches, so these exist only to nudge the
    remaining imbalance. Using strong weights here as well is the
    double-correction that destabilised earlier runs.
    """
    counts = np.bincount(labels[train_idx], minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)
    w = (1.0 / counts) ** power
    w = w / w.mean()                 # centre around 1.0
    return torch.tensor(w, dtype=torch.float32, device=device)
