"""Three CPU regressors that all consume the same per-object feature matrix."""

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from nanoscigpt.scientific.models import PatchEncoder


class NumericNano(nn.Module):
    def __init__(self, ux, scale):
        super().__init__()
        self.register_buffer("center", ux.mean(0))
        self.register_buffer("spread", ux.std(0, unbiased=False).clamp_min(0.025))
        self.scale = scale
        n = ux.shape[1]
        self.enc = PatchEncoder(8, (n + 7) // 8, 16)
        self.mlp = nn.Sequential(
            nn.Linear(n + 16, 40), nn.SiLU(), nn.Linear(40, 20), nn.SiLU()
        )
        self.output = nn.Linear(20, 1)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.output.weight)

    def components(self, x):
        z = ((x - self.center) / self.spread).clamp(-8, 8)
        p = nn.functional.pad(z, (0, (-z.shape[1]) % 8)).reshape(len(x), -1, 8)
        h = self.mlp(torch.cat((z, self.enc.encode(p).mean(1)), 1))
        return z, h, x[:, 0] + self.scale * self.output(h).flatten()


def warm(seed, t, settings, kind):
    torch.manual_seed(seed)
    x = t["x"]
    net = NumericNano(x, settings["scale"])
    if kind == "direct":
        return net
    y = t["y"]
    if y.ndim == 2:
        y = y[:, 0] if settings["target"] == "single" else y.mean(1)
    if kind == "shuffle":
        y = y[np.random.default_rng(seed + 29471).permutation(len(y))]
    rng = np.random.default_rng(seed + 821)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0017)
    scale = settings["scale"]
    cuts = settings["cuts"]
    for _ in range(400):
        i = rng.choice(len(x), min(64, len(x)), replace=False)
        p = net.components(x[i])[2]
        if cuts:
            bounds = torch.tensor(cuts, dtype=x.dtype)
            loss = nn.functional.binary_cross_entropy_with_logits(
                (p[:, None] - bounds) / settings["width"],
                (y[i, None] > torch.arange(len(cuts))).to(x.dtype),
            )
        else:
            loss = nn.functional.smooth_l1_loss(
                (p - y[i]) / scale, torch.zeros_like(p), beta=0.4
            )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
    return net


def refine(m, x, y, steps):
    opt = torch.optim.AdamW(m.parameters(), lr=0.00065)
    for _ in range(steps):
        pred = m.components(x)[2]
        loss = nn.functional.smooth_l1_loss(
            (pred - y) / m.scale, torch.zeros_like(y), beta=0.35
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 2.0)
        opt.step()


def raw_kernel(a, b):
    return a @ b.T / a.shape[1]


@torch.no_grad()
def gram(m, x, t, shape):
    z, h, p = m.components(x)
    w, j, _ = m.components(t)
    z, w, h, j = [v.double() for v in (z, w, h, j)]
    k = raw_kernel(z, w) + 0.35 * raw_kernel(h, j)
    if shape == "smooth":
        # A smooth local residual can accompany the common linear basis.
        k = k + torch.exp(-torch.cdist(z, w).square() / (1.25 * z.shape[1]))
    return k + 2, p.double()


def readout(m, x, y, cfg):
    k, p = gram(m, x, x, cfg["shape"])
    return torch.linalg.solve(
        k + cfg["alpha"] * torch.eye(len(k), dtype=k.dtype), (y.double() - p)
    )


def prediction(m, x, t, coef, cfg):
    k, p = gram(m, x, t, cfg["shape"])
    return (p + k @ coef).numpy()


def forest(seed, x, y, cfg):
    label = y - x[:, 0] if cfg["residual"] else y
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=cfg["leaf"],
        max_features=cfg["fraction"],
        n_jobs=1,
        random_state=seed,
    ).fit(x, label)


def forest_predict(m, x, cfg):
    return m.predict(x) + (x[:, 0].numpy() if cfg["residual"] else 0.0)


def fit(seed, settings, train, aux, tests):
    x, y = train["x"], train["y"]
    cfg = settings["rf"]
    rf = forest(seed, x, y, cfg)
    result = [[forest_predict(rf, g["x"], cfg) for g in tests]]
    for arm in ("direct", "pre"):
        cfg = settings[arm]
        m = warm(seed, aux, settings["task"], arm)
        refine(m, x, y, cfg["steps"])
        coef = readout(m, x, y, cfg)
        result.append([prediction(m, g["x"], x, coef, cfg) for g in tests])
    return result
