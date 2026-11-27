"""Predict measured short RNA duplex formation free energy from stacking features."""

import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn


class StackNano(nn.Module):
    def __init__(self, auxiliary_features):
        super().__init__()
        width = auxiliary_features.shape[1]
        self.register_buffer("center", auxiliary_features.mean(0))
        self.register_buffer("spread", auxiliary_features.std(0, unbiased=False).clamp_min(0.8))
        self.encoder = nn.Linear(width, 16, dtype=torch.float64)
        self.output = nn.Linear(width + 16, 1, dtype=torch.float64)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def basis(self, features):
        z = (features - self.center) / self.spread
        return torch.cat([z, 0.25 * torch.tanh(self.encoder(z))], dim=1)

    def forward(self, features):
        return -9.0 + 4.0 * self.output(self.basis(features)).flatten()


def fit(seed, settings, train, aux, tests):
    features, target = train["x"].double(), train["y"].double()
    auxiliary = aux["x"].double()
    rf = RandomForestRegressor(
        n_estimators=200, max_features=settings["rf"][0],
        min_samples_leaf=settings["rf"][1], n_jobs=1, random_state=seed,
    ).fit(features, target)
    predictions = [[rf.predict(test["x"]) for test in tests]]
    for pretrain in (False, True):
        torch.manual_seed(seed)
        model = StackNano(auxiliary)
        if pretrain:
            cuts = auxiliary.new_tensor([-15, -12, -9, -6, -3])
            above_cut = (aux["y"][:, None] > torch.arange(len(cuts))).double()
            opt = torch.optim.AdamW(model.parameters(), lr=0.007, weight_decay=0.015)
            for _ in range(180):
                logits = (model(auxiliary)[:, None] - cuts) / 0.75
                loss = nn.functional.binary_cross_entropy_with_logits(logits, above_cut)
                loss = loss + 0.015 * model.output.weight.square().sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

        opt = torch.optim.AdamW(model.parameters(), lr=0.0007)
        for _ in range(settings["steps"]):
            loss = nn.functional.smooth_l1_loss(model(features), target, beta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Both arms fit the same regularized residual on the same 32 labels.
        with torch.no_grad():
            basis = model.basis(features)
            center = basis.mean(0)
            basis = basis - center
            residual = target - model(features)
            offset = residual.mean()
            penalty = settings["alpha"] * torch.eye(basis.shape[1], dtype=basis.dtype)
            weights = torch.linalg.solve(basis.T @ basis + penalty, basis.T @ (residual - offset))
            row = []
            for test in tests:
                x = test["x"].double()
                row.append((model(x) + (model.basis(x) - center) @ weights + offset).numpy())
            predictions.append(row)
    return predictions
