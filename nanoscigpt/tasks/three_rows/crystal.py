"""Periodic graph encoder with auxiliary reconstruction of crystal descriptors."""

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nanoscigpt.scientific.models import CrystalGraphEncoder, masked_mean
from . import optimize, forest


class Density(nn.Module):
    def __init__(self, unlabeled):
        super().__init__()
        features = unlabeled["features"].float()
        self.register_buffer("mean", features.mean(0, keepdim=True))
        self.register_buffer(
            "scale", features.std(0, keepdim=True, unbiased=False).clamp_min(0.0001)
        )
        self.encoder = CrystalGraphEncoder(hidden_dim=16)
        self.aux = nn.Linear(16, features.shape[1])
        self.head = nn.Linear(16 + features.shape[1], 1)

    def represent(self, part, indices, corrupt=False):
        atoms, positions, lattice, mask = [
            part[k][indices] for k in ("x", "fractional", "lattice", "mask")
        ]
        if corrupt:
            atoms = atoms.masked_fill((torch.rand(atoms.shape) < 0.20) & mask, 0)
        hidden = masked_mean(self.encoder.encode(atoms, positions, lattice, mask), mask)
        descriptors = (
            (part["features"][indices].float() - self.mean) / self.scale
        ).clamp(-12, 12)
        return hidden, descriptors

    def forward(self, part, indices):
        hidden, descriptors = self.represent(part, indices)
        return self.head(torch.cat([hidden, descriptors], dim=1)).flatten()


def inputs(part):
    return np.column_stack(
        [
            part[key].reshape(len(part["x"]), -1).numpy()
            for key in ("x", "fractional", "lattice", "mask", "features")
        ]
    )


def fit(seed, settings, train, unlabeled, tests):
    targets = train["y"].float().ravel()
    predictions = [
        forest(
            inputs(train),
            targets,
            seed,
            settings["rf"]["max_features"],
            list(map(inputs, tests)),
        )
    ]
    center, scale = targets.mean(), targets.std(unbiased=False).clamp_min(1e-6)
    torch.manual_seed(seed)
    direct = Density(unlabeled)
    pretrained = copy.deepcopy(direct)
    torch.manual_seed(40000 + seed)
    optimizer = torch.optim.AdamW(pretrained.parameters(), lr=0.003)
    for _ in range(160):
        indices = torch.randint(len(unlabeled["x"]), (16,))
        hidden, desc = pretrained.represent(unlabeled, indices, corrupt=True)
        optimize(optimizer, F.mse_loss(pretrained.aux(hidden), desc))
    for method, model in (("direct", direct), ("pre", pretrained)):
        cfg = settings[method]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        gen = torch.Generator().manual_seed(seed + 12345)
        normalized = (targets - center) / scale
        for _ in range(cfg["supervised_updates"]):
            indices = torch.randint(len(targets), (16,), generator=gen)
            optimize(optimizer, F.mse_loss(model(train, indices), normalized[indices]))
        model.eval()
        with torch.no_grad():
            predictions.append(
                [
                    (
                        torch.cat(
                            [
                                model(test, ids)
                                for ids in torch.arange(len(test["x"])).split(32)
                            ]
                        )
                        * scale
                        + center
                    ).numpy()
                    for test in tests
                ]
            )
    return predictions
