"""Train the three 32-label CPU baselines on each bundled domain."""

import csv
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor

TRAINERS = {
    "spectrum": "spectrum",
    "protein": "sequences",
    "rna": "sequences",
    "dna": "spectrum",
    "smiles": "descriptors",
    "crystal": "crystal",
    "structure3d": "structure3d",
    "weather": "descriptors",
    "field": "spectrum",
    "image": "descriptors",
    "text": "descriptors",
}
DOMAINS = tuple(TRAINERS)
METHODS = ("随机森林", "Nano 直接微调", "Nano 预训练后微调")


def optimize(optimizer, loss):
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    optimizer.zero_grad()
    loss.backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    nn.utils.clip_grad_norm_(parameters, 2.0)
    optimizer.step()


def forest(features, target, seed, fraction, tests):
    model = RandomForestRegressor(
        n_estimators=200, max_features=fraction, random_state=seed, n_jobs=1
    )
    model.fit(features, target.numpy().ravel())
    return [model.predict(x) for x in tests]


def run(domain, data_root, out_root, overwrite=False):
    destination = Path(out_root) / domain
    receipt = destination / "three_rows_receipt.json"
    if receipt.exists() and not overwrite:
        raise FileExistsError(f"finished three-row example exists at {receipt}")
    folder = Path(data_root) / domain / "three_rows"
    config = json.loads((folder / "settings.json").read_text(encoding="utf8"))
    fixture = folder / "fixture.npz"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest != config["sha256"]:
        raise ValueError(f"fixture checksum mismatch: {fixture}")
    with np.load(fixture) as arrays:
        splits = {}
        for key in arrays.files:
            part, field = key.split("_", 1)
            splits.setdefault(part, {})[field] = torch.from_numpy(arrays[key])
    torch.set_num_threads(1)
    start = time.perf_counter()
    word_count = domain == "text" and config.get("variant") == "unspaced_word_count"
    module = "boundary_count" if word_count else TRAINERS[domain]
    if domain == "field" and config.get("variant") == "thermal_transfer":
        module = "thermal_diffusion"
    if domain == "smiles" and config.get("variant") == "scaffold_logp":
        module = "paired_numeric"
    measured = config.get("variant") == "measured_sequence"
    if measured:
        module = "measured_sequence"
    fit = import_module(f"{__name__}.{module}").fit
    classification = domain in ("text", "protein", "rna") and not (word_count or measured)
    metric = "balanced_accuracy" if classification else "MAE"
    tests = [splits["group1"], splits["group2"]]
    test_inputs = [{k: v for k, v in test.items() if k != "y"} for test in tests]
    rows, differences = [], []
    for key, settings in config["seeds"].items():
        train = splits[key]
        if len(train["y"]) != 32:
            raise ValueError("expected 32 labeled training objects")
        training_settings = {k: v for k, v in settings.items() if k != "expected"}
        predictions = fit(
            int(key), training_settings, train, splits["aux"], test_inputs
        )
        scores = []
        for group, test in enumerate(tests):
            y = test["y"].numpy().ravel()
            values = []
            for prediction in predictions:
                p = np.asarray(prediction[group])
                if p.shape != y.shape or not np.isfinite(p).all():
                    raise ValueError("invalid prediction shape or nonfinite value")
                if classification:
                    values.append(
                        np.mean([((p[y == i] > 0.5) == i).mean() for i in (0, 1)])
                    )
                else:
                    values.append(np.abs(p - y).mean())
            scores.append(values)
        scores = np.asarray(scores)
        reference = config.get("platform_expected", {}).get(sys.platform, {}).get(key, settings["expected"])
        diff = np.max(np.abs(scores - np.array(reference)))
        if not np.isfinite(diff) or diff > 2e-6:
            raise ValueError(
                f"{domain}, seed {key}: {metric} differs from reference by {diff}"
            )
        rows.append(scores)
        differences.append(float(diff))
    scores = np.array(rows)
    means = scores.mean(axis=0).T
    direction = 1 if classification else -1
    counts = (
        (direction * np.diff(scores, axis=2) > 1e-10).all(axis=2).sum(axis=0).tolist()
    )
    destination.mkdir(exist_ok=True, parents=True)
    with (destination / "three_rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["方法", f"确认组 1 {metric}", f"确认组 2 {metric}"])
        writer.writerows(zip(METHODS, *means.T))
    record = dict(
        domain=domain,
        device="cpu",
        fixture_sha256=digest,
        seeds=list(map(int, config["seeds"])),
        labeled_per_seed=32,
        test_sizes=[len(p["y"]) for p in tests],
        metric=metric,
        **{
            (
                "balanced_accuracy_by_seed" if classification else "mae_by_seed"
            ): scores.tolist()
        },
        paired_wins=counts,
        max_abs_deviation=max(differences),
        seconds=time.perf_counter() - start,
    )
    receipt.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf8"
    )
    for method, (first, second) in zip(METHODS, means):
        print(f"{method:18} {first:.6f}  {second:.6f}")
    print(f"paired wins: {counts[0]}/5, {counts[1]}/5")
    return record
