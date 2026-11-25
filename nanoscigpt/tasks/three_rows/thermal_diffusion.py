"""Estimate warm-temperature diffusivity from four reference-temperature field snapshots."""

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from nanoscigpt.scientific.models import PatchEncoder

GAS = 0.008314462618


def features(part):
    obs = np.asarray(part["obs"], dtype=np.float64)
    modes = np.arange(1, 4)
    fourier = (obs @ np.sin(2 * np.pi * np.arange(64)[:, None] * modes / 64)) * (2 / 64)
    grid = np.linspace(0.005, 0.12, 345)
    design = np.exp(-grid[:, None, None] * np.arange(4)[None, :, None] * modes**2)
    norm = (design**2).sum(1)
    # Profile out the unknown initial amplitude of each Fourier mode.
    estimates = []
    for start in range(0, len(obs), 48):
        proj = np.einsum(
            "ntm,gtm->ngm", fourier[start : start + 48], design, optimize=False
        )
        score = (proj**2 / norm).sum(-1)
        estimates.append(grid[score.argmax(1)])
    ref = np.concatenate(estimates)
    q = (1 / 298.0 - 1 / np.asarray(part["temp"], dtype=np.float64)) / GAS
    return torch.tensor(
        np.column_stack((part["material"], ref, q)), dtype=torch.float32
    )


def smooth_basis(x):
    c = x[:, :2]
    axis = torch.linspace(0, 1, 5, dtype=x.dtype)
    centers = torch.cartesian_prod(axis, axis)
    radial = torch.exp(-((c[:, None, :] - centers) ** 2).sum(-1) / 0.065)
    return torch.cat((c * 2 - 1, radial), 1)


def barrier_target(x, y):
    return (torch.log(y / x[:, 2]) / x[:, 3] - 35) / 15


def diffusivity(x, barrier):
    return x[:, 2] * torch.exp(x[:, 3] * (35 + 15 * barrier))


def forest_features(x, mode):
    if mode == "raw":
        return torch.column_stack(
            (x, x[:, 2] * torch.exp(35 * x[:, 3]), smooth_basis(x))
        )
    return x[:, :2] if mode == "barrier" else smooth_basis(x)


def train_forest(x, y, seed, cfg):
    mode = cfg["mode"]
    target = y if mode == "raw" else barrier_target(x, y)
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=cfg["leaf"],
        max_features=cfg["fraction"],
        random_state=seed,
        n_jobs=1,
    ).fit(forest_features(x, mode), target)


def forest_predict(model, x, mode):
    p = model.predict(forest_features(x, mode))
    return p if mode == "raw" else diffusivity(x, torch.from_numpy(p)).numpy()


class ThermalNano(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = PatchEncoder(3, 9, 12)
        self.head = nn.Linear(12, 1)
        self.linear = nn.Linear(27, 1)
        nn.init.normal_(self.head.weight, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        b = smooth_basis(x)
        h = self.patch.encode(b.reshape(-1, 9, 3)).mean(1)
        return (self.linear(b) + self.head(h)).flatten()


def build(seed, u, msd, mode):
    torch.manual_seed(seed)
    m = ThermalNano()
    if mode == "direct":
        return m
    target = (
        torch.log(msd[:, 1] / msd[:, 0]) / ((1 / 298.0 - 1 / 318.0) / GAS) - 35
    ) / 15
    if mode == "shuffle":
        target = target[np.random.default_rng(seed + 8369).permutation(len(u))]
    opt = torch.optim.AdamW(m.parameters(), lr=0.0028)
    rng = np.random.default_rng(seed + 16539)
    for _ in range(360):
        i = rng.choice(len(u), 64, replace=False)
        err = m(u[i]) - target[i]
        loss = nn.functional.smooth_l1_loss(err, torch.zeros_like(err), beta=0.6)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def tune(m, x, y, steps):
    opt = torch.optim.AdamW(m.parameters(), lr=0.0006)
    target = barrier_target(x, y)
    for _ in range(steps):
        loss = nn.functional.smooth_l1_loss(m(x), target, beta=0.4)
        opt.zero_grad()
        loss.backward()
        opt.step()


def calibrate(m, x, y, alpha):
    with torch.no_grad():
        b = smooth_basis(x).double()
        r = (barrier_target(x, y) - m(x)).double()
        mu = b.mean(0)
        bc = b - mu
        w = torch.linalg.solve(
            bc.T @ bc + alpha * torch.eye(b.shape[1], dtype=b.dtype),
            bc.T @ (r - r.mean()),
        )
        return (w.float(), (r.mean() - mu @ w).float())


def calibrated(m, x, cal):
    with torch.no_grad():
        w, offset = cal
        return diffusivity(x, m(x) + smooth_basis(x) @ w + offset).numpy()


def fit(seed, settings, train, aux, tests):
    x, u = features(train), features(aux)
    y = train["y"]
    groups = [features(g) for g in tests]
    cfg = settings["rf"]
    rf = train_forest(x, y, seed, cfg)
    result = [[forest_predict(rf, g, cfg["mode"]) for g in groups]]
    for arm in ("direct", "pre"):
        cfg = settings[arm]
        model = build(seed, u, aux["msd"], arm)
        tune(model, x, y, cfg["steps"])
        cal = calibrate(model, x, y, cfg["alpha"])
        result.append([calibrated(model, g, cal) for g in groups])
    return result
