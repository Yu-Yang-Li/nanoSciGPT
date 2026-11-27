"""Shared patch encoder for spectral flux, DNA mappability and late diffusion."""

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nanoscigpt.scientific.models import PatchEncoder
from . import forest, optimize


class Nano(nn.Module):
    def __init__(self, u):
        super().__init__()
        x = u["x"]
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sd", x.std(0, unbiased=False).clamp_min(1e-5))
        self.register_buffer("rm", u["ref"].mean())
        self.register_buffer("rs", u["ref"].std(unbiased=False).clamp_min(1e-6))
        self.pad = (-x.shape[1]) % 8
        self.encoder = PatchEncoder(8, (x.shape[1] + self.pad) // 8, hidden_dim=16)
        width = 1 + x.shape[1] + 16
        self.proxy = nn.Linear(width, 1)
        nn.init.zeros_(self.proxy.weight)
        nn.init.zeros_(self.proxy.bias)
        self.head = nn.Linear(width + 1, 1)

    def blocks(self, x, ref):
        z = ((x - self.mu) / self.sd).clamp(-8, 8)
        r = ((ref - self.rm) / self.rs).clamp(-10, 10)[:, None]
        tokens = F.pad(z, (0, self.pad)).reshape(len(z), -1, 8)
        h = self.encoder.encode(tokens).mean(1)
        return r, torch.cat([r, z / np.sqrt(z.shape[1]), h / 4], 1)

    def auxval(self, x, ref):
        r, f = self.blocks(x, ref)
        return (r + self.proxy(f)).ravel()

    def design(self, x, ref):
        r, t = self.blocks(x, ref)
        return torch.cat([r + self.proxy(t), t], 1)

    def forward(self, x, ref):
        return self.head(self.design(x, ref)).ravel()

    @torch.no_grad()
    def initialize(self, x, r, y, lam):
        z = self.design(x, r).double()
        mc = z.mean(0)
        z = z - mc
        yy = y.double()
        penalty = np.full(z.shape[1], lam)
        penalty[:2] = 0.08
        w = torch.linalg.solve(
            z.T @ z + torch.diag(torch.from_numpy(penalty)), z.T @ (yy - yy.mean())
        )
        self.head.weight.copy_(w[None])
        self.head.bias.copy_(yy.mean() - mc @ w)


@torch.no_grad()
def predict(net, p, center, scale):
    q = torch.cat(
        [net(p["x"][i], p["ref"][i]) for i in torch.arange(len(p["x"])).split(128)]
    )
    return (center + scale * q).numpy()


def pretrain(base, u, seed):
    net = copy.deepcopy(base)
    q = u["aux"]
    q = (q - q.mean()) / q.std(unbiased=False).clamp_min(1e-5)
    opt = torch.optim.AdamW(net.parameters(), lr=0.004)
    gen = torch.Generator().manual_seed(seed + 21903)
    for i in range(160):
        ix = torch.randint(len(q), (16,), generator=gen)
        optimize(
            opt, F.smooth_l1_loss(net.auxval(u["x"][ix], u["ref"][ix]), q[ix], beta=0.4)
        )
    return net


def forest_x(p, form):
    ref = p["ref"].numpy()[:, None]
    x = p["x"].numpy()
    return ref if form == "reference" else np.column_stack([ref, x])


def calibrated_predictions(net, train, aux, tests, mu, sd, config):
    cols = config["cols"]
    scale = (
        np.maximum(aux["x"][:, cols].numpy().astype(float).std(0), 1e-5)
        * config["width"]
    )
    centers = train["x"][:, cols].numpy().astype(float) / scale

    def kernel(x):
        return np.exp(-0.5 * ((x[:, None] - centers[None, :]) ** 2).sum(2))

    residual = train["y"].numpy() - predict(net, train, mu, sd)
    weights = np.linalg.solve(kernel(centers) + 0.15 * np.eye(len(centers)), residual)
    return [
        predict(net, t, mu, sd)
        + kernel(t["x"][:, cols].numpy().astype(float) / scale) @ weights
        for t in tests
    ]


def fit(seed, settings, train, unlabeled, tests):
    cfg = settings["rf"]
    features = [forest_x(test, cfg["form"]) for test in tests]
    predictions = [
        forest(forest_x(train, cfg["form"]), train["y"], seed, cfg["frac"], features)
    ]
    torch.manual_seed(seed)
    base = Nano(unlabeled)
    for method in ("direct", "pre"):
        cfg = settings[method]
        net = (
            copy.deepcopy(base)
            if method == "direct"
            else pretrain(base, unlabeled, seed)
        )
        y = train["y"]
        mu = float(y.mean())
        sd = max(float(y.std(unbiased=False)), 1e-6)
        target = (y - mu) / sd
        net.initialize(train["x"], train["ref"], target, cfg["lam"])
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"])
        gen = torch.Generator().manual_seed(seed + 12345)
        for _ in range(cfg["steps"]):
            ids = torch.randint(len(y), (16,), generator=gen)
            pred = net(train["x"][ids], train["ref"][ids])
            optimize(opt, F.smooth_l1_loss(pred, target[ids], beta=0.45))
        extension = cfg.get("extension", {})
        if extension.get("extra_steps", 0):
            opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"] / 2)
            gen = torch.Generator().manual_seed(seed + 633227)
            for _ in range(extension["extra_steps"]):
                ids = torch.randint(len(y), (16,), generator=gen)
                pred = net(train["x"][ids], train["ref"][ids])
                optimize(opt, F.smooth_l1_loss(pred, target[ids], beta=0.45))
        if extension.get("cal"):
            predictions.append(
                calibrated_predictions(
                    net, train, unlabeled, tests, mu, sd, extension["cal"]
                )
            )
        else:
            predictions.append([predict(net, test, mu, sd) for test in tests])
    return predictions
