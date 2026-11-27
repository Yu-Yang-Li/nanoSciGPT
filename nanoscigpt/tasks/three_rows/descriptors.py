"""Frozen descriptor and patch models for text, flow, solubility and source counts."""

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from nanoscigpt.scientific.models import PatchEncoder
from nanoscigpt.core.gpt import GPT, GPTConfig
from . import optimize


def scaled(x, mean, sd):
    return ((x - mean) / sd).clamp(-12, 12)


def with_flow_proxy(p):
    f = p["features"]
    return dict(p, features=torch.cat([f, f[:, -1:]], 1).float())


class Encoder(nn.Module):
    def __init__(self, domain, u, proxy=False):
        super().__init__()
        self.domain = domain
        self.proxy = proxy
        self.sequence = domain in ("text", "smiles")
        f = u["features"].float()
        self.register_buffer("fm", f.mean(0, keepdim=True))
        self.register_buffer(
            "fs", f.std(0, keepdim=True, unbiased=False).clamp_min(0.0001)
        )
        if self.sequence:
            self.emb = GPT(GPTConfig(f.shape[1], 32, 1, 1, 16)).transformer.wte
        else:
            x = u["x"].float()
            self.base = PatchEncoder(x.shape[2], x.shape[1], hidden_dim=16)
            self.register_buffer("xm", x.mean((0, 1), keepdim=True))
            self.register_buffer(
                "xs", x.std((0, 1), keepdim=True, unbiased=False).clamp_min(0.0001)
            )
        self.aux = nn.Linear(16, f.shape[1])
        self.head = nn.Linear(
            16 if self.sequence else 16 + f.shape[1], 2 if domain == "text" else 1
        )
        if proxy:
            self.skip = nn.Parameter(torch.tensor(0.5))
            self.head.weight.data.mul_(0.1)
            nn.init.zeros_(self.head.bias)

    def latents(self, part, ids):
        f = scaled(part["features"][ids].float(), self.fm, self.fs)
        if self.sequence:
            return f @ self.emb.weight, f
        x = scaled(part["x"][ids].float(), self.xm, self.xs)
        return self.base.encode(x).mean(1), f

    def forward(self, part, ids):
        h, f = self.latents(part, ids)
        out = self.head(h if self.sequence else torch.cat([h, f], dim=1))
        return out + self.skip * f[:, -1:] if self.proxy else out


class Calibrated(Encoder):
    def __init__(self, domain, u):
        super().__init__(domain, u)
        self.register_buffer("zm", u["z"].mean(0))
        self.register_buffer("zs", u["z"].std(0, unbiased=False).clamp_min(0.0001))
        dim = 16 + len(self.zm)
        self.correction = nn.Sequential(nn.Linear(dim, 24), nn.SiLU(), nn.Linear(24, 1))
        nn.init.zeros_(self.correction[-1].bias)
        self.correction[-1].weight.data.mul_(0.03)
        self.cal = nn.Linear(1 + dim, 1)

    def design(self, part, ids):
        z = scaled(part["z"][ids], self.zm, self.zs)
        h, _ = self.latents(part, ids)
        joint = torch.cat([z, h], 1)
        return torch.cat([z[:, :1] + self.correction(joint), joint], 1)

    def forward(self, part, ids):
        return self.cal(self.design(part, ids))

    @torch.no_grad()
    def initialize(self, train, normalized, penalty):
        self.eval()
        x = self.design(train, torch.arange(len(normalized))).double().numpy()
        cue = np.column_stack([np.ones(len(x)), x[:, 0]])
        z = x[:, 1:]
        zsd = z.std(0).clip(0.10)
        z = z / zsd
        projection = np.linalg.lstsq(cue, z, rcond=None)[0]
        residual = z - cue @ projection
        y = normalized.double().numpy()
        a = np.linalg.lstsq(cue, y, rcond=None)[0]
        beta = np.linalg.solve(
            residual.T @ residual + penalty * np.eye(z.shape[1]),
            residual.T @ (y - cue @ a),
        )
        a -= projection @ beta
        self.cal.weight.copy_(
            torch.tensor(np.r_[a[1], beta / zsd][None], dtype=torch.float32)
        )
        self.cal.bias.fill_(float(a[0]))


def warm(base, aux, seed):
    net = copy.deepcopy(base)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003)
    domain = base.domain
    if domain in ("smiles", "image"):
        pools = []
        for index in range(2 if domain == "smiles" else 1):
            prefix = str(index)
            part = {
                k[len(prefix) :]: value
                for k, value in aux.items()
                if k.startswith(prefix)
            }
            y = part["aux"]
            pools.append((part, (y - y.mean()) / y.std(unbiased=False)))
        gen = torch.Generator().manual_seed(42231 + seed)
        for step in range(160):
            part, y = pools[step % len(pools)]
            ids = torch.randint(len(y), (16,), generator=gen)
            optimize(opt, F.mse_loss(net.design(part, ids)[:, 0], y[ids]))
    else:
        torch.manual_seed(40000 + seed)
        if domain == "weather":
            y = aux["aux"]
            y = (y - y.mean()) / y.std(unbiased=False)
        for _ in range(160):
            ids = torch.randint(len(aux["x"]), (16,))
            if domain == "text":
                f = scaled(aux["features"][ids].float(), net.fm, net.fs)
                masked = f.masked_fill(torch.rand_like(f) < 0.30, 0)
                loss = F.mse_loss(net.aux(masked @ net.emb.weight), f)
            else:
                loss = F.mse_loss(net(aux, ids).flatten(), y[ids])
            optimize(opt, loss)
    return net


def fit(seed, settings, train, aux, tests):
    if settings.get("observation") == "crowded_exposure":
        from .crowded_image import fit as fit_source_count
        return fit_source_count(seed, settings, train, aux, tests)
    if settings.get("observation") == "two_cloud_layers":
        from .weather_flow import fit as fit_cloud_speed
        return fit_cloud_speed(seed, settings, train, aux, tests)
    domain = settings["domain"]
    cls = domain == "text"
    cfg = settings["rf"]

    def rf_features(p):
        if settings.get("rf_proxy"):
            p = with_flow_proxy(p)
        raw = p["features"].numpy()
        if domain in ("image", "weather"):
            raw = np.column_stack([p["x"].reshape(len(raw), -1).numpy(), raw])
        if "form" in cfg:
            z = p["z"].numpy()
            return z if cfg["form"] == "compact" else np.column_stack([raw, z])
        return raw

    rf = (RandomForestClassifier if cls else RandomForestRegressor)(
        n_estimators=200, max_features=cfg["max_features"], random_state=seed, n_jobs=1
    )
    rf.fit(rf_features(train), train["y"].numpy().ravel())
    predictions = [
        [
            rf.predict_proba(rf_features(t))[:, 1]
            if cls
            else rf.predict(rf_features(t))
            for t in tests
        ]
    ]
    if settings.get("proxy"):
        train, aux, *tests = [with_flow_proxy(p) for p in (train, aux, *tests)]
    torch.manual_seed(seed)
    calibrated = domain in ("smiles", "image")
    base = (
        Calibrated(domain, aux)
        if calibrated
        else Encoder(domain, aux, settings.get("proxy", False))
    )
    y = train["y"].ravel()
    mu, sd = (
        (0.0, 1.0)
        if cls
        else (float(y.float().mean()), max(float(y.float().std(unbiased=False)), 1e-6))
    )
    target = y.long() if cls else (y.float() - mu) / sd
    for method in ("direct", "pre"):
        cfg = settings[method]
        model = copy.deepcopy(base) if method == "direct" else warm(base, aux, seed)
        if calibrated:
            model.initialize(train, target, cfg["lambda_ridge"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        gen = torch.Generator().manual_seed(seed + 12345)
        model.train()
        for _ in range(cfg["supervised_updates"]):
            ids = torch.randint(len(y), (16,), generator=gen)
            logits = model(train, ids)
            loss = (
                F.cross_entropy(logits, target[ids])
                if cls
                else F.mse_loss(logits.flatten(), target[ids])
            )
            optimize(opt, loss)
        model.eval()
        with torch.no_grad():
            outputs = []
            for test in tests:
                value = torch.cat(
                    [
                        model(test, chunk)
                        for chunk in torch.arange(len(test["x"])).split(32)
                    ]
                )
                outputs.append(
                    (
                        value.softmax(1)[:, 1] if cls else value.flatten() * sd + mu
                    ).numpy()
                )
            predictions.append(outputs)
    return predictions
