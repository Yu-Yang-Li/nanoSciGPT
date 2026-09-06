"""nanoGPT blocks with continuous value+mask patches, not TimesFM-3 weights.

The patch interface is inspired by TimesFM-3. No TimesFM source is copied.
Source curves are already phase-folded/normalised by the dataset authors.
No full-curve sample statistics are fitted here, avoiding future-patch leakage.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .core.gpt import Block, GPTConfig


def patchify(x, patch_size):
    """Right-pad without resampling; return values and point-level validity."""
    if x.ndim != 2 or patch_size < 1 or x.shape[1] < 1:
        raise ValueError('Expected nonempty (batch, length) and positive patch size')
    valid = torch.isfinite(x)
    x = torch.where(valid, x, 0.)
    padding = (-x.shape[1]) % patch_size
    return (F.pad(x, (0, padding)).reshape(len(x), -1, patch_size),
            F.pad(valid, (0, padding), value=False).reshape(len(x), -1, patch_size))


def next_patch_loss(predictions, patches, valid):
    """Position i predicts i+1, scoring only real target observations."""
    target, keep = patches[:, 1:], valid[:, 1:]
    if not keep.any():
        raise ValueError('No observed next-patch targets')
    return (predictions - target).square()[keep].mean()


class LightCurveGPT(nn.Module):
    def __init__(self, patch_size=8, width=64, layers=2, heads=4, max_length=201):
        super().__init__()
        self.patch_size = patch_size
        self.config = dict(patch_size=patch_size, width=width, layers=layers,
                           heads=heads, max_length=max_length)
        n_positions = math.ceil(max_length / patch_size) + 1
        config = GPTConfig(vocab_size=1, block_size=n_positions, n_embd=width,
                           n_layer=layers, n_head=heads, causal=True)
        # Continuous residual embedding: values + observed flags, no vocabulary.
        self.patch_mlp = nn.Sequential(nn.Linear(2 * patch_size, width), nn.GELU(),
                                       nn.Linear(width, width))
        self.patch_skip = nn.Linear(2 * patch_size, width)
        self.position = nn.Embedding(n_positions, width)
        self.bos = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([Block(config) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.forecast_head = nn.Linear(width, patch_size)
        # Each view has its own positions. Pool only after separate encodings.
        self.classifier = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                         nn.Linear(width, 1))

    def encode(self, x):
        values, valid = patchify(x, self.patch_size)
        tokens = torch.cat([values, valid.to(values.dtype)], dim=-1)
        h = self.patch_mlp(tokens) + self.patch_skip(tokens)
        h = torch.cat([self.bos.expand(len(x), -1, -1), h], dim=1)
        h = h + self.position(torch.arange(h.shape[1], device=x.device))
        # BOS ensures even a missing first patch always has an attendable key.
        pad = F.pad(~valid.any(-1), (1, 0), value=False)
        for block in self.blocks:
            h = block(h, pad_mask=pad)
        return self.norm(h)[:, 1:], values, valid

    def predict_patches(self, x):
        h, _, _ = self.encode(x)
        return self.forecast_head(h[:, :-1])

    def pretraining_loss(self, x):
        h, values, valid = self.encode(x)
        return next_patch_loss(self.forecast_head(h[:, :-1]), values, valid)

    def view_features(self, x):
        h, _, valid = self.encode(x)
        keep = valid.any(-1)
        mean = (h * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp_min(1)
        last_idx = (keep * torch.arange(1, h.shape[1] + 1, device=x.device)).max(1).values - 1
        last = h[torch.arange(len(x), device=x.device), last_idx.clamp_min(0)]
        last = last * keep.any(1, keepdim=True)
        return torch.cat([mean, last], dim=-1)

    def forward(self, global_view, local_view):
        features = torch.cat([self.view_features(global_view), self.view_features(local_view)], -1)
        return self.classifier(features).squeeze(-1)
