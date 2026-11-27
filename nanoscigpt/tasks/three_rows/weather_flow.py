"""Estimate advection speed in mixed cloud images using multiscale optical flow."""
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from nanoscigpt.scientific.models import PatchEncoder
from .cloud_motion import frames, descriptors


class CloudSpeedEncoder(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mean))
        self.register_buffer("sd", torch.as_tensor(scale))
        self.patch = PatchEncoder(5, 12, 16)
        self.linear = nn.Linear(60, 1)
        self.out = nn.Linear(16, 1)
        for layer in (self.linear, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.gain = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, flow):
        features = ((flow - self.mu) / self.sd).clamp(-6, 6)
        pooled = self.patch.encode(features).mean(1)
        correction = self.bias + self.linear(features.flatten(1)).flatten()
        correction = correction + self.out(pooled).flatten()
        return self.gain * flow[:, 4, 0] + .025 * correction


def update(model, x, y, optimizer):
    delta = (model(x) - y) * 40
    loss = nn.functional.smooth_l1_loss(delta, torch.zeros_like(y), beta=.5)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.)
    optimizer.step()


def flow_tensor(part):
    return torch.from_numpy(descriptors(frames({"x": part["x"].numpy()})))


def fit(seed, settings, train, aux, tests):
    x = flow_tensor(train)
    ux = flow_tensor(aux)
    test_flow = [flow_tensor(p) for p in tests]
    y = train["y"].float().flatten()
    rf_config = settings["forest"]
    def rf_inputs(flow, sample):
        if rf_config["subset"] == "speed":
            return flow[:, :, 0].numpy()
        flat = flow.flatten(1).numpy()
        if rf_config["subset"] == "full":
            return np.column_stack((flat, sample["features"].numpy(), sample["x"].flatten(1).numpy()))
        return flat
    forest = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1,
                                   max_features=rf_config["fraction"])
    forest.fit(rf_inputs(x, train), y.numpy())
    predictions = [[forest.predict(rf_inputs(flow, p)) for flow, p in zip(test_flow, tests)]]
    raw = ux.numpy()
    for method in ("direct", "pre"):
        torch.manual_seed(seed)
        model = CloudSpeedEncoder(raw.mean(0), raw.std(0).clip(.0003))
        if method == "pre":
            rng = np.random.default_rng(seed + 71409)
            target = aux["aux"].float().flatten()
            opt = torch.optim.AdamW(model.parameters(), lr=.0016)
            for _ in range(480):
                ids = rng.integers(len(ux), size=48)
                update(model, ux[ids], target[ids], opt)
        with torch.no_grad():
            prediction = model(x).numpy()
            centered = prediction - prediction.mean()
            slope = np.sum(centered * (y.numpy() - y.numpy().mean())) / max(np.sum(centered**2), 1e-9)
            slope = float(np.clip(slope, .4, 2.))
            model.gain.mul_(slope)
            model.linear.weight.mul_(slope)
            model.out.weight.mul_(slope)
            model.bias.add_((y - model(x)).mean() / .025)
        config = settings[method]
        opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        for _ in range(config["steps"]):
            update(model, x, y, opt)
        with torch.no_grad():
            predictions.append([model(test).numpy() for test in test_flow])
    return predictions
