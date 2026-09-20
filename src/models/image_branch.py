"""
Image branch: 2D CNN over log-Mel spectrograms.

Two backbones are available:

1. `scratch` (default) — a compact residual CNN with SE blocks, trained from
   random init. No downloads required, so it works offline.

2. `resnet34` — an ImageNet-pretrained ResNet from torchvision.
   On a dataset this small (~6.9k cycles, and far fewer independent patients)
   transfer learning usually gives a real gain even though spectrograms are not
   natural images: the early filters still detect edges and textures, which map
   onto spectrogram structure fine. Use it when you have internet access.

   Enable with:  --backbone resnet34
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock2d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w[:, :, None, None]


class ResBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.se = SEBlock2d(out_ch)
        self.drop = nn.Dropout2d(dropout)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )

    def forward(self, x):
        idt = self.skip(x)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        return self.drop(F.gelu(h + idt))


class ScratchCNN(nn.Module):
    """Compact residual CNN trained from scratch."""

    def __init__(self, num_classes=4, width=32, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
        )
        self.layer1 = ResBlock2d(width, width * 2, stride=2, dropout=0.05)
        self.layer2 = ResBlock2d(width * 2, width * 4, stride=2, dropout=0.10)
        self.layer3 = ResBlock2d(width * 4, width * 8, stride=2, dropout=0.15)

        self.feature_dim = width * 8 * 2       # avg + max pooling
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, mask=None, return_features=False):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)

        avg = F.adaptive_avg_pool2d(h, 1).flatten(1)
        mx = F.adaptive_max_pool2d(h, 1).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)

        logits = self.head(pooled)
        if return_features:
            return logits, pooled
        return logits


class PretrainedCNN(nn.Module):
    """ImageNet-pretrained ResNet adapted to single-channel spectrograms."""

    def __init__(self, num_classes=4, arch="resnet34", dropout=0.3,
                 pretrained=True):
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError as e:
            raise ImportError(
                "torchvision is required for the pretrained backbone. "
                "Install it, or use --backbone scratch."
            ) from e

        weights = "DEFAULT" if pretrained else None
        net = getattr(tvm, arch)(weights=weights)

        # Adapt the 3-channel stem to 1 channel by summing the RGB filters.
        # This preserves the pretrained edge detectors rather than
        # reinitialising them randomly.
        old = net.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        net.conv1 = new

        self.feature_dim = net.fc.in_features * 2
        net.fc = nn.Identity()
        self.backbone = net

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _features(self, x):
        b = self.backbone
        h = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        h = b.layer1(h); h = b.layer2(h)
        h = b.layer3(h); h = b.layer4(h)
        avg = F.adaptive_avg_pool2d(h, 1).flatten(1)
        mx = F.adaptive_max_pool2d(h, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x, mask=None, return_features=False):
        pooled = self._features(x)
        logits = self.head(pooled)
        if return_features:
            return logits, pooled
        return logits


def build_image_branch(num_classes=4, backbone="scratch", dropout=0.3):
    if backbone == "scratch":
        return ScratchCNN(num_classes=num_classes, dropout=dropout)
    return PretrainedCNN(num_classes=num_classes, arch=backbone,
                         dropout=dropout, pretrained=True)
