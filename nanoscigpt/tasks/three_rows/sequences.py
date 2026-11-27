"""Sequence-composition classifiers for yeast proteins and central Rfam fragments."""

import copy

import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
import torch.nn.functional as F

from nanoscigpt.core.gpt import GPT, GPTConfig
from . import optimize


class ProteinComposition(nn.Module):
    def __init__(self, aux):
        super().__init__()
        self.emb = GPT(GPTConfig(20, 32, 1, 1, 8)).transformer.wte
        self.readout = nn.Linear(8, 1, bias=False)
        self.register_buffer("prior", 2 * aux["chem"])
        self.register_buffer("mean", aux["x"][:, :20].mean(0))
        self.register_buffer(
            "scale", self.descriptor(aux["x"]).std(unbiased=False).clamp_min(0.005)
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.loggain = nn.Parameter(torch.zeros(()))

    def descriptor(self, x):
        return x[:, :20] @ self.prior.sigmoid()

    def forward(self, x):
        weights = (self.prior + self.readout(self.emb.weight).flatten()).sigmoid()
        z = ((x[:, :20] - self.mean) @ weights) / self.scale
        score = self.loggain.exp().clamp(max=12) * z + self.bias
        return torch.stack((-score, score), dim=1)

    @torch.no_grad()
    def initialize(self, x, y):
        scores = self(x)[:, 1]
        self.bias.sub_((scores[y == 0].median() + scores[y == 1].median()) / 2)


class RnaFamily(nn.Module):
    def __init__(self, aux):
        super().__init__()
        x = aux["x"]
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sd", x.std(0, unbiased=False).clamp_min(0.004))
        self.emb = GPT(GPTConfig(x.shape[1], 32, 1, 1, 24)).transformer.wte
        self.mid = nn.Linear(24, 24)
        self.aux = nn.Linear(24, 14)
        self.head = nn.Linear(24 + x.shape[1], 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def scaled(self, x):
        return ((x - self.mu) / self.sd).clamp(-7, 7)

    def latent(self, z):
        h = z @ self.emb.weight
        return h + F.gelu(self.mid(h))

    def design(self, z):
        return torch.cat(
            [z / z.shape[1] ** 0.5, F.normalize(self.latent(z), dim=1, eps=1e-6)], dim=1
        )

    def forward(self, x):
        score = self.head(self.design(self.scaled(x))).flatten()
        return torch.stack((-score, score), dim=1)

    @torch.no_grad()
    def initialize(self, x, y):
        h = self.design(self.scaled(x)).double()
        center = h.mean(0)
        h = h - center
        target = 2 * y.double() - 1 - self(x)[:, 1].double()
        w = h.T @ torch.linalg.solve(
            h @ h.T + 0.4 * torch.eye(len(x), dtype=h.dtype), target - target.mean()
        )
        self.head.weight.add_(w[None])
        self.head.bias.add_(target.mean() - center @ w)


def warm(base, aux, seed, protein):
    model = copy.deepcopy(base)
    x = aux["x"]
    if protein:
        halves = [aux["left"], aux["right"]]
        center = model.descriptor((halves[0] + halves[1]) / 2).mean()
        targets = [(model.descriptor(h) - center) / model.scale for h in halves]
        gen = torch.Generator().manual_seed(seed + 40000)
        lr = 0.006
    else:
        torch.manual_seed(40000 + seed)
        lr = 0.003
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(160):
        if protein:
            ids = torch.randint(len(x), (16,), generator=gen)
            pred = model(halves[step % 2][ids])[:, 1]
            loss = F.smooth_l1_loss(pred, targets[1 - step % 2][ids])
        else:
            ids = torch.randint(len(x), (16,))
            z = model.scaled(x[ids])
            z = z.masked_fill(torch.rand_like(z) < 0.25, 0.0)
            preds = model.aux(model.latent(z)).split((2, 4, 8), dim=1)
            targets = aux["target"][ids].split((2, 4, 8), dim=1)
            loss = (
                sum(
                    -(q * F.log_softmax(p, 1)).sum(1).mean()
                    for p, q in zip(preds, targets)
                )
                / 3
            )
        optimize(opt, loss)
    return model


def shifted(probs, offset):
    return expit(logit(np.clip(np.asarray(probs, float), 1e-7, 1 - 1e-7)) - offset)


def fit(seed, settings, train, aux, tests):
    protein = settings["domain"] == "protein"
    torch.manual_seed(seed)
    base = ProteinComposition(aux) if protein else RnaFamily(aux)
    x, y = train["x"], train["y"].long()
    cfg = settings["rf"]

    def features(x):
        if not protein:
            return x.numpy()
        desc = base.descriptor(x).numpy()[:, None]
        return (
            desc if cfg["form"] == "chem_single" else np.column_stack([x.numpy(), desc])
        )

    rf = RandomForestClassifier(
        n_estimators=200, max_features=cfg["max_features"], random_state=seed, n_jobs=1
    )
    rf.fit(features(x), y.numpy())
    predictions = [
        [
            shifted(rf.predict_proba(features(t["x"]))[:, 1], cfg["offset"])
            for t in tests
        ]
    ]
    classes = (0, 1) if protein else (1, 0)
    pools = [torch.nonzero(y == c).flatten() for c in classes]
    for method in ("direct", "pre"):
        model = (
            copy.deepcopy(base)
            if method == "direct"
            else warm(base, aux, seed, protein)
        )
        model.initialize(x, y)
        cfg = settings[method]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        gen = torch.Generator().manual_seed(seed + 12345)
        for _ in range(cfg["steps"]):
            ids = torch.cat(
                [pool[torch.randint(len(pool), (8,), generator=gen)] for pool in pools]
            )
            optimize(opt, F.cross_entropy(model(x[ids]), y[ids], label_smoothing=0.06))
        with torch.no_grad():
            predictions.append(
                [
                    shifted(model(t["x"]).softmax(1)[:, 1].numpy(), cfg["offset"])
                    for t in tests
                ]
            )
    return predictions
