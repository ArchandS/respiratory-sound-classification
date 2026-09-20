"""
Signal branch: 1D CNN + BiLSTM + self-attention over MFCC sequences.

Design notes / fixes over the original version
----------------------------------------------
* GroupNorm instead of BatchNorm. BatchNorm statistics shift with whatever
  classes happen to be in a batch, which adds variance on top of an already
  imbalanced problem. GroupNorm normalises per-sample and is batch-independent.

* Pre-norm RESIDUAL attention (x = x + Attn(LN(x))). The original attention
  block had neither a residual path nor normalisation, so its output scale
  drifted with its input and fed straight into pooling. This is the standard
  transformer stabilisation.

* MASKED pooling. Padded frames are excluded from both mean and max pooling.
  Previously padding was averaged in as if it were signal, diluting short
  cycles (which is most of them).

* Squeeze-Excitation on the conv stack lets the network weight MFCC
  coefficient bands by usefulness rather than treating all equally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1d(nn.Module):
    """Channel attention: learn which feature channels matter."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):                    # (B, C, T)
        w = x.mean(dim=2)                    # (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, groups=8, pool=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.se = SEBlock1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.norm(self.conv(x)))
        x = self.se(x)
        x = self.pool(x)
        return self.drop(x)


class ResidualAttention(nn.Module):
    """Pre-norm residual multi-head self-attention with padding mask."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x


def masked_pool(x, mask):
    """
    Mean + max pooling that ignores padded frames.

    x    : (B, T, D)
    mask : (B, T) True where the frame is real signal
    """
    if mask is None:
        return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)

    m = mask.unsqueeze(-1).float()                     # (B, T, 1)
    denom = m.sum(dim=1).clamp(min=1.0)
    mean = (x * m).sum(dim=1) / denom

    neg_inf = torch.finfo(x.dtype).min
    mx = x.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
    # If a row had no valid frames at all, fall back to the mean
    mx = torch.where(torch.isfinite(mx), mx, mean)

    return torch.cat([mean, mx], dim=1)


class SignalBranch(nn.Module):
    def __init__(self, num_classes=4, in_channels=120, hidden=128,
                 dropout=0.3):
        super().__init__()

        # Multi-scale front end: kernel 3 catches short transients (crackles),
        # kernel 7 catches sustained patterns (wheezes), kernel 5 in between.
        self.stem3 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.stem5 = nn.Conv1d(in_channels, 64, 5, padding=2)
        self.stem7 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.stem_norm = nn.GroupNorm(8, 192)

        self.block1 = ConvBlock1d(192, 256, 3, pool=2, dropout=0.1)
        self.block2 = ConvBlock1d(256, 256, 3, pool=2, dropout=0.15)

        self.lstm = nn.LSTM(256, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)

        d = hidden * 2
        self.attn = ResidualAttention(d, heads=4, dropout=0.1)

        self.head = nn.Sequential(
            nn.Linear(d * 2, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.feature_dim = d * 2

    def _downsample_mask(self, mask, target_len):
        """Shrink the frame mask to match the conv stack's temporal downsampling."""
        if mask is None:
            return None
        m = mask.unsqueeze(1).float()
        m = F.adaptive_max_pool1d(m, target_len).squeeze(1)
        m = m > 0.5
        # Guarantee at least one valid frame per sample
        empty = ~m.any(dim=1)
        if empty.any():
            m[empty, 0] = True
        return m

    def forward(self, x, mask=None, return_features=False):
        # x: (B, C, T)
        h = torch.cat([self.stem3(x), self.stem5(x), self.stem7(x)], dim=1)
        h = F.gelu(self.stem_norm(h))

        h = self.block1(h)
        h = self.block2(h)

        h = h.transpose(1, 2)                          # (B, T', D)
        m = self._downsample_mask(mask, h.size(1))

        h, _ = self.lstm(h)
        h = self.attn(h, key_padding_mask=(~m) if m is not None else None)

        pooled = masked_pool(h, m)
        logits = self.head(pooled)

        if return_features:
            return logits, pooled
        return logits
