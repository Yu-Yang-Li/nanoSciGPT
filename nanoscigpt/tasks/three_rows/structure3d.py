"""Precision-weighted helix pitch from independent position measurements."""

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nanoscigpt.scientific.models import PatchEncoder
from . import forest, optimize

theta = np.linspace(0, 4 * np.pi, 16)
B = np.column_stack([np.ones(16), np.cos(theta), np.sin(theta), theta / (2 * np.pi)])
TARGET = np.array([0.0, 0.0, 0.0, 1.0])


class PrecisionNano(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PatchEncoder(16, 16, hidden_dim=16)
        self.relative = nn.Parameter(torch.zeros(16))
        self.local = nn.Linear(16, 1)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.gain = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("basis", torch.from_numpy(B))
        self.register_buffer("target", torch.from_numpy(TARGET)[:, None])

    def logvar(self, x):
        h = self.encoder.encode(x.float() / 2.0)
        return (self.relative + 0.4 * torch.tanh(self.local(h).squeeze(-1))).clamp(
            -2.5, 5.0
        )

    def contrast(self, x):
        precision = torch.exp(-self.logvar(x)).double()
        wb = precision[:, :, None] * self.basis
        coeff = torch.linalg.solve(
            self.basis.T @ wb, self.target.expand(len(x), -1, -1)
        )
        return (wb @ coeff).squeeze(-1)

    def forward(self, x):
        w = self.contrast(x)
        q = -0.5 * torch.einsum("bi,bij,bj->b", w, x.double().square(), w)
        geom = q.clamp_min(1e-8).sqrt().float()
        return geom + 0.01 * self.offset + 0.01 * self.gain * (geom - 1.25)


def pretrain(base, aux, seed):
    net = copy.deepcopy(base)
    net.train()
    q = aux["q"]
    gen = torch.Generator().manual_seed(seed + 38123)
    opt = torch.optim.Adam(net.parameters(), lr=0.035)
    for step in range(160):
        ids = torch.randint(len(q), (8,), generator=gen)
        x = torch.cat([aux["a"][ids], aux["b"][ids]])
        v = net.logvar(x)
        ratio = q[ids].repeat(2, 1)
        optimize(opt, (v + ratio * torch.exp(-v)).mean())
    return net


@torch.no_grad()
def infer(net, x):
    net.eval()
    return torch.cat([net(batch) for batch in x.split(64)]).numpy()


def fit_fixed(start, x, y, seed, cfg):
    net = copy.deepcopy(start)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.001)
    gen = torch.Generator().manual_seed(12345 + seed)
    for _ in range(cfg["supervised_steps"]):
        net.train()
        batch = torch.randint(len(y), (16,), generator=gen)
        residual = (net(x[batch]) - y[batch].flatten()) / 0.01
        optimize(opt, F.smooth_l1_loss(residual, torch.zeros_like(residual)))
    return net


def fit(seed, settings, train, unlabeled, tests):
    x, y = train["x"], train["y"]
    predictions = [
        forest(
            train["geom"],
            y,
            seed,
            settings["rf"]["max_features"],
            [t["geom"] for t in tests],
        )
    ]
    torch.manual_seed(seed)
    base = PrecisionNano()
    direct = fit_fixed(base, x, y, seed, settings["direct"])
    pretrained = fit_fixed(pretrain(base, unlabeled, seed), x, y, seed, settings["pre"])
    for model in (direct, pretrained):
        predictions.append([infer(model, t["x"]) for t in tests])
    return predictions
