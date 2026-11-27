"""Recover word counts from letters, with optional partially observed boundary supervision."""

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from nanoscigpt.core.gpt import GPT, GPTConfig


def features(tokens):
    x = np.asarray(tokens)
    length = (x > 0).sum(1)
    vowel = np.isin(x, [1, 5, 9, 15, 21, 25])
    starts = vowel & ~np.pad(vowel[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    letters = np.column_stack([(x == i).sum(1) for i in range(1, 27)])
    reps = ((x[:, 1:] == x[:, :-1]) & (x[:, 1:] > 0)).sum(1)
    return np.column_stack([length, vowel.sum(1), starts.sum(1), reps, letters]).astype(
        "float32"
    )


class BoundaryCounter(nn.Module):
    def __init__(self, mu, scale):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("scale", torch.as_tensor(scale))
        self.transformer = GPT(GPTConfig(27, 72, 1, 1, 16, causal=False)).transformer
        self.local = nn.Conv1d(16, 24, 6, padding=0)
        self.score = nn.Linear(24, 1)
        self.pair = nn.Embedding(27 * 27, 1)
        self.context = nn.Linear(16, 1)
        self.readout = nn.Linear(31, 1)
        nn.init.zeros_(self.pair.weight)
        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, -1.28)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)

    def boundaries(self, x):
        h = self.transformer.wte(x)
        position = self.transformer.wpe(torch.arange(x.shape[1], device=x.device))
        contextual = self.transformer.ln_f(
            self.transformer.h[0](h + position, pad_mask=x == 0)
        )
        # Each candidate boundary sees three letters on each side, including end padding.
        local = self.local(nn.functional.pad(h.transpose(1, 2), (2, 2))).transpose(1, 2)
        ids = 27 * x[:, :-1] + x[:, 1:]
        return (
            self.score(nn.functional.gelu(local)).squeeze(-1)
            + self.pair(ids).squeeze(-1)
            + self.context(contextual[:, :-1]).squeeze(-1)
        )

    def basis(self, x, f):
        count = 1 + (self.boundaries(x).sigmoid() * (x[:, 1:] > 0)).sum(1)
        return count, torch.cat([count[:, None] / 4.0, (f - self.mu) / self.scale], 1)

    def forward(self, x, f):
        p, b = self.basis(x, f)
        return p + self.readout(b).flatten()


def initialize(ux, uf, aux, seed, paired, shuffle=False):
    torch.manual_seed(seed)
    model = BoundaryCounter(uf.mean(0), uf.std(0).clip(0.15))
    if not paired:
        return model
    values = np.array(aux, copy=True)
    if shuffle:
        present = values >= 0
        values[present] = np.random.default_rng(seed + 513).permutation(values[present])
    target = torch.tensor(values, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed + 4182)
    for _ in range(700):
        ids = rng.integers(len(ux), size=32)
        p = model.boundaries(ux[ids])
        y = target[ids]
        mask = y >= 0
        loss = nn.functional.binary_cross_entropy_with_logits(p[mask], y[mask])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
    return model


def readout(model, x, f, y, alpha):
    with torch.no_grad():
        p, b = model.basis(x, f)
        b = b.double().numpy()
        r = (y - p).double().numpy()
        z = b - b.mean(0)
        w = np.zeros(b.shape[1])
        if alpha == "affine":
            c = z[:, 0]
            w[0] = c @ (r - r.mean()) / (c @ c + 1e-6)
        else:
            w = z.T @ np.linalg.solve(
                z @ z.T + float(alpha) * np.eye(len(z)), r - r.mean()
            )
        model.readout.weight.copy_(torch.as_tensor(w[None]))
        model.readout.bias.fill_(float(r.mean() - b.mean(0) @ w))


def step(model, x, f, y, opt):
    loss = nn.functional.smooth_l1_loss(model(x, f), y, beta=0.65)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()


def forest_input(x, f, form):
    if form == "compact":
        return f[:, :4]
    if form == "full":
        return np.column_stack([f, np.asarray(x)])
    return f


def fit(seed, settings, train, aux, tests):
    x = train["x"].long()
    y = train["y"].float()
    ux = aux["x"].long()
    uf = features(ux)
    tr = features(x)
    f = torch.tensor(tr)
    test = [(v["x"].long(), features(v["x"])) for v in tests]
    rf = settings["rf"]
    m = RandomForestRegressor(
        n_estimators=200, max_features=rf["fraction"], random_state=seed, n_jobs=1
    )
    m.fit(forest_input(x, tr, rf["form"]), y.numpy())
    pred = [[m.predict(forest_input(v, z, rf["form"])) for v, z in test]]
    for arm in ("direct", "pre"):
        c = settings[arm]
        model = initialize(ux, uf, aux["marked"].numpy(), seed, arm == "pre")
        readout(model, x, f, y, c["alpha"])
        opt = torch.optim.AdamW(model.parameters(), lr=0.00045)
        for _ in range(c["steps"]):
            step(model, x, f, y, opt)
        with torch.no_grad():
            pred.append([model(v, torch.tensor(z)).numpy() for v, z in test])
    return [
        [np.rint(p) if settings[arm]["round"] else p for p in group]
        for arm, group in zip(("rf", "direct", "pre"), pred)
    ]
