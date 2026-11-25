"""Count sources in a low-resolution crowded field, optionally using sharper exposures."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.ensemble import RandomForestRegressor
from torch import nn

from nanoscigpt.scientific.models import PatchEncoder


def image_features(images):
    images = np.asarray(images, dtype=np.float32)
    count = len(images)
    yy, xx = np.mgrid[-1:1:16j, -1:1:16j]
    border = (np.abs(xx) > .90) | (np.abs(yy) > .90)
    plane = np.stack([np.ones_like(xx), xx, yy], axis=-1)
    coefficients = np.linalg.lstsq(plane[border], images[:, border].T, rcond=None)[0]
    sky = (plane.reshape(-1, 3) @ coefficients).T.reshape(count, 16, 16)
    signal = images - sky
    summaries = []
    for width in (.45, .8, 1.25):
        smooth = gaussian_filter(signal, (0, width, width))
        contrast = smooth - gaussian_filter(signal, (0, 2, 2))
        peak = smooth == maximum_filter(smooth, size=(1, 3, 3))
        contrast_peak = contrast == maximum_filter(contrast, size=(1, 3, 3))
        for threshold in (.20, .45, .72):
            summaries.append((peak & (smooth > threshold)).sum((1, 2)))
            summaries.append((contrast_peak & (contrast > threshold * .42)).sum((1, 2)))
    for channel in (signal, np.maximum(signal, 0)**2, np.maximum(contrast, 0),
                    contrast_peak * (contrast > .12)):
        quadrants = channel.reshape(count, 2, 8, 2, 8).sum((2, 4)).reshape(count, 4)
        summaries.extend(quadrants.T)
    flux = signal.sum((1, 2))
    quantiles = [np.quantile(signal, q, axis=(1, 2)) for q in (.25, .5, .75, .9, .98, 1)]
    return np.column_stack([flux, summaries[0], *summaries, *quantiles,
                            images.std((1, 2))]).astype("float32")


class SourceCounter(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        tokens = (width + 7) // 8
        self.padding = tokens * 8 - width
        self.encoder = PatchEncoder(8, tokens, 16)
        self.dense = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 16))
        self.measurement = nn.Linear(32, outputs)
        self.head = nn.Linear(width + 33, 1)
        nn.init.zeros_(self.measurement.weight)
        nn.init.zeros_(self.measurement.bias)

    def basis(self, x, cue):
        patches = nn.functional.pad(x, (0, self.padding)).reshape(len(x), -1, 8)
        hidden = torch.cat((self.encoder.encode(patches).mean(1), self.dense(x)), 1)
        estimates = self.measurement(hidden) + cue[:, None]
        return torch.cat((estimates[:, :1], hidden, x), 1), estimates

    def forward(self, x, cue):
        basis, _ = self.basis(x, cue)
        return basis[:, 0] + self.head(basis).flatten()


def fit_readout(model, x, cue, y, alpha):
    with torch.no_grad():
        basis = model.basis(x, cue)[0].double().numpy()
        residual = y.double().numpy() - basis[:, 0]
        centered = basis - basis.mean(0)
        if alpha == "affine":
            first = centered[:, 0]
            weight = np.zeros(basis.shape[1])
            weight[0] = first @ (residual - residual.mean()) / (first @ first + 1e-8)
        else:
            gram = centered @ centered.T + float(alpha) * np.eye(len(basis))
            weight = centered.T @ np.linalg.solve(gram, residual - residual.mean())
        model.head.weight.copy_(torch.as_tensor(weight[None]))
        model.head.bias.fill_(float(residual.mean() - basis.mean(0) @ weight))


def gradient_step(model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.)
    optimizer.step()


def fit(seed, settings, train, aux, tests):
    arrays = [image_features(p["obs"]) for p in (train, aux, *tests)]
    labeled, unlabeled, *test_arrays = arrays
    cfg = settings["rf"]

    def forest_input(features, sample):
        if cfg["form"] == "full":
            return np.column_stack([features, np.asarray(sample["obs"]).reshape(len(features), -1)])
        if cfg["form"] == "compact":
            return features[:, :2]
        return features

    forest = RandomForestRegressor(n_estimators=200, max_features=cfg["fraction"],
                                   random_state=seed, n_jobs=1)
    forest.fit(forest_input(labeled, train), np.asarray(train["y"]))
    predictions = [[forest.predict(forest_input(z, p)) for z, p in zip(test_arrays, tests)]]
    mean = unlabeled.mean(0)
    scale = unlabeled.std(0).clip(.0001)

    def inputs(features):
        x = torch.as_tensor(np.clip((features - mean) / scale, -7, 7), dtype=torch.float32)
        flux_cue = features[:, 0] / (2 * np.pi)
        return x, torch.as_tensor(flux_cue / 5., dtype=torch.float32)

    x, cue = inputs(labeled)
    ux, uc = inputs(unlabeled)
    val = [inputs(z) for z in test_arrays]
    y = train["y"].float() / 5.
    for arm in ("direct", "pre"):
        cfg_arm = settings[arm]
        target = aux[cfg_arm["target"]].float() / 5.
        torch.manual_seed(seed)
        model = SourceCounter(x.shape[1], target.shape[1])
        if arm == "pre":
            rng = np.random.default_rng(seed + 17331)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
            for _ in range(680):
                indices = rng.integers(len(ux), size=64)
                _, estimate = model.basis(ux[indices], uc[indices])
                residual = estimate - target[indices]
                loss = nn.functional.smooth_l1_loss(residual[:, 0], torch.zeros(len(indices)), beta=.18)
                if residual.shape[1] > 1:
                    loss = loss + .6 * nn.functional.smooth_l1_loss(
                        residual[:, 1:], torch.zeros_like(residual[:, 1:]), beta=.12)
                gradient_step(model, optimizer, loss)
        fit_readout(model, x, cue, y, cfg_arm["alpha"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
        for _ in range(cfg_arm["steps"]):
            loss = nn.functional.smooth_l1_loss(model(x, cue), y, beta=.18)
            gradient_step(model, optimizer, loss)
        with torch.no_grad():
            predictions.append([model(v, t).numpy() * 5. for v, t in val])
    for i, arm in enumerate(("rf", "direct", "pre")):
        if settings[arm]["round"]:
            predictions[i] = [np.rint(p) for p in predictions[i]]
    return predictions
