"""
Shared training / evaluation engine.

Both branches use the identical loop, so any improvement to optimisation
benefits both and the two branches stay directly comparable.

Stability measures baked in (these are the fixes for the oscillating-validation
problem seen in the original runs):
  * LR warmup then cosine decay — the attention/LSTM stack is fragile in the
    first few epochs while Q/K/V projections are still random.
  * Gradient clipping.
  * Imbalance correction in the SAMPLER, with only mild residual loss weights.
  * EMA of weights — evaluating a moving average of recent weights rather than
    the exact current point smooths out epoch-to-epoch noise considerably.
  * Model selection on ICBHI score, not accuracy.
"""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .losses import SoftTargetFocalLoss, mixup_batch
from .metrics import compute_all, format_metrics, full_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class EMA:
    """
    Exponential moving average of model weights.

    Cheap and reliably worth a point or two here: it damps the epoch-to-epoch
    validation swings that come from a small, noisy, imbalanced dataset.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)

    def module(self):
        return self.shadow


def make_scheduler(optimizer, warmup_epochs, total_epochs, min_factor=0.02):
    def fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cos = 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
        return min_factor + (1 - min_factor) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, class_names, return_logits=False):
    model.eval()
    all_logits, all_targets = [], []

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(x, mask)
        all_logits.append(logits.float().cpu())
        all_targets.append(y)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = logits.argmax(dim=1)

    metrics = compute_all(targets.numpy(), preds.numpy(), class_names)
    if return_logits:
        return metrics, logits.numpy(), targets.numpy()
    return metrics


def train_model(model, train_loader, val_loader, cfg, device,
                class_weights, run_name: str, log_fn=print):
    """
    Train one branch. Returns (best_metrics, best_state_dict, history).
    """
    t = cfg.train
    model = model.to(device)

    criterion = SoftTargetFocalLoss(alpha=class_weights,
                                    gamma=t.focal_gamma).to(device)

    # No weight decay on norms and biases — decaying them hurts more than helps
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in n.lower() or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.lr,
    )
    scheduler = make_scheduler(optimizer, t.warmup_epochs, t.epochs,
                               t.min_lr_factor)

    use_amp = bool(t.amp and device.type == "cuda")
    # torch.amp.* is the current API (torch.cuda.amp.* is deprecated)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=0.999)

    best_value, best_state, best_metrics = -np.inf, None, None
    no_improve = 0
    history = []

    log_fn(f"\n{'='*66}")
    log_fn(f"TRAINING: {run_name}")
    log_fn(f"  params      : {sum(p.numel() for p in model.parameters()):,}")
    log_fn(f"  device      : {device}   AMP: {use_amp}")
    log_fn(f"  epochs      : {t.epochs}  (early stop patience {t.patience})")
    log_fn(f"  selecting on: {t.monitor}")
    log_fn(f"{'='*66}")

    for epoch in range(t.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x, y_soft = mixup_batch(x, y, t.num_classes,
                                    t.mixup_alpha, t.label_smoothing)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x, mask)
                loss = criterion(logits, y_soft)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        raw_metrics = evaluate(model, val_loader, device, t.class_names)
        ema_metrics = evaluate(ema.module(), val_loader, device, t.class_names)

        # Keep whichever of raw / EMA weights is performing better
        if ema_metrics[t.monitor] >= raw_metrics[t.monitor]:
            metrics, which, state_src = ema_metrics, "ema", ema.module()
        else:
            metrics, which, state_src = raw_metrics, "raw", model

        lr_now = optimizer.param_groups[0]["lr"]
        log_fn(f"[{epoch+1:03d}/{t.epochs}] "
               f"loss={epoch_loss/max(1,n_batches):.4f} "
               f"lr={lr_now:.2e} | {format_metrics(metrics)} "
               f"({which}, {time.time()-t0:.0f}s)")

        history.append({"epoch": epoch + 1,
                        "loss": epoch_loss / max(1, n_batches),
                        **metrics})

        if metrics[t.monitor] > best_value:
            best_value = metrics[t.monitor]
            best_metrics = metrics
            best_state = copy.deepcopy(state_src.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= t.patience:
                log_fn(f"\nEarly stop: no {t.monitor} improvement "
                       f"for {t.patience} epochs.")
                break

    log_fn(f"\nBest {t.monitor}: {best_value:.4f}")
    return best_metrics, best_state, history


def save_run(out_dir: Path, run_name: str, state_dict, metrics, history,
             norm_stats, report_text: str):
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    mean, std = norm_stats
    torch.save({"state_dict": state_dict,
                "norm_mean": mean, "norm_std": std,
                "metrics": metrics},
               out_dir / "checkpoints" / f"{run_name}.pth")

    with open(out_dir / "reports" / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "reports" / f"{run_name}_report.txt", "w") as f:
        f.write(report_text)
