"""Three rows from one model. The nanoGPT blocks are shared, the encoder changes per field.

Row 1  random forest on the flattened input
Row 2  SciGPT trained from scratch on 32 labels
Row 3  SciGPT pretrained by next-token prediction on unlabeled objects,
       then fine-tuned on the same 32 labels

Every field uses the same model size, learning rate and number of steps.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch.nn import functional as F

from ..scientific.adapters import patchify_2d
from ..scientific.encoders import CrystalEncoder, PatchEncoder, SymbolEncoder
from ..scientific.scigpt import SciGPT

SEED = 53
PARTS = ("53", "aux", "group1", "group2")
DOMAINS = (
    "spectrum", "protein", "rna", "dna", "smiles", "crystal",
    "structure3d", "weather", "field", "image", "text",
)
METHODS = ("随机森林", "Nano 直接微调", "Nano 预训练后微调")
N_EMBD = 32
PRETRAIN_STEPS = 600
FINETUNE_STEPS = 200
LEARNING_RATE = 1e-3


def symbol_parts(domain, folder, arrays):
    """Text, protein, RNA, DNA and SMILES: one token per character."""
    if domain == "text":
        tokens = {part: arrays[f"{part}_x"].astype(np.int64) for part in PARTS}
        vocab_size = 27
    else:
        strings = {part: [] for part in PARTS}
        with (folder / "panel.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            next(rows)
            for row in rows:
                strings[row[0]].append(row[1])
        alphabet = sorted({c for items in strings.values() for s in items for c in s})
        token_of = {c: i + 1 for i, c in enumerate(alphabet)}
        length = max(len(s) for items in strings.values() for s in items)
        tokens = {}
        for part, items in strings.items():
            tokens[part] = np.zeros((len(items), length), dtype=np.int64)
            for row, string in enumerate(items):
                tokens[part][row, : len(string)] = [token_of[c] for c in string]
        vocab_size = len(alphabet) + 1
    parts = {}
    for part, ids in tokens.items():
        counts = np.stack([(ids == s).sum(1) for s in range(1, vocab_size)], axis=1)
        parts[part] = {
            "inputs": {"tokens": torch.from_numpy(ids)},
            "flat": np.hstack([counts, ids]),
        }
    block_size = tokens["aux"].shape[1]
    return (lambda: SymbolEncoder(vocab_size, N_EMBD)), block_size, parts


def patch_parts(domain, arrays):
    """Spectrum, field, image, weather and 3D structure: one token per numeric patch."""

    def patches(part):
        if domain == "spectrum":  # 32 flux samples -> 8 tokens of 4 neighbouring wavelengths
            return arrays[f"{part}_x"][:, -32:].reshape(-1, 8, 4)
        if domain == "field":  # 4 frames of 64 points -> 32 tokens of 8 neighbouring points
            return arrays[f"{part}_obs"].reshape(-1, 32, 8)
        if domain == "image":  # 16x16 pixels -> 16 tokens of 4x4 pixels
            image = torch.from_numpy(arrays[f"{part}_obs"])[:, None]
            return patchify_2d(image, 4).numpy()
        if domain == "structure3d":  # one token per point: its distances to all 16 points
            return arrays[f"{part}_a" if part == "aux" else f"{part}_x"]
        return arrays[f"{part}_x"]  # weather: 16 image regions, each seen in 4 frames

    def context(part):  # field only: porosity, crosslinking and working temperature
        return np.column_stack([arrays[f"{part}_material"], arrays[f"{part}_temp"]])

    has_context = domain == "field"
    reference = patches("aux")
    mean, scale = reference.mean((0, 1)), reference.std((0, 1)).clip(1e-6)
    if has_context:
        context_mean, context_scale = context("aux").mean(0), context("aux").std(0)
    parts = {}
    for part in PARTS:
        values = ((patches(part) - mean) / scale).astype(np.float32)
        inputs = {"patches": torch.from_numpy(values)}
        flat = values.reshape(len(values), -1)
        if has_context:
            extra = ((context(part) - context_mean) / context_scale).astype(np.float32)
            inputs["context"] = torch.from_numpy(extra)
            flat = np.hstack([flat, extra])
        parts[part] = {"inputs": inputs, "flat": flat}
    n_tokens, patch_dim = reference.shape[1:]
    context_dim = 3 if has_context else 0
    make_encoder = lambda: PatchEncoder(patch_dim, N_EMBD, context_dim)
    return make_encoder, n_tokens + has_context, parts


def crystal_parts(arrays):
    """Crystal: the lattice token followed by one token per atom."""
    lattice_scale = arrays["aux_lattice"].max()
    parts = {}
    for part in PARTS:
        elements = arrays[f"{part}_x"]
        fractional = arrays[f"{part}_fractional"]
        lattice = arrays[f"{part}_lattice"] / lattice_scale
        parts[part] = {
            "inputs": {
                "elements": torch.from_numpy(elements),
                "fractional": torch.from_numpy(fractional),
                "lattice": torch.from_numpy(lattice),
            },
            "flat": np.hstack([a.reshape(len(a), -1) for a in (elements, fractional, lattice)]),
        }
    return (lambda: CrystalEncoder(N_EMBD)), 1 + elements.shape[1], parts


def load(domain, data_root):
    folder = Path(data_root) / domain / "three_rows"
    with np.load(folder / "fixture.npz") as source:
        arrays = {key: source[key] for key in source.files}
    if domain in ("text", "protein", "rna", "dna", "smiles"):
        make_encoder, block_size, parts = symbol_parts(domain, folder, arrays)
    elif domain == "crystal":
        make_encoder, block_size, parts = crystal_parts(arrays)
    else:
        make_encoder, block_size, parts = patch_parts(domain, arrays)
    for part in ("53", "group1", "group2"):
        parts[part]["y"] = arrays[f"{part}_y"].astype(np.float32).ravel()
    return make_encoder, block_size, parts


def select(inputs, rows):
    return {name: values[rows] for name, values in inputs.items()}


def pretrain(model, unlabeled):
    """GPT pretraining: predict each token from the tokens before it. No labels are used."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)
    n_objects = len(next(iter(unlabeled.values())))
    losses = []
    for _ in range(PRETRAIN_STEPS):
        batch = select(unlabeled, torch.randint(n_objects, (64,), generator=generator))
        hidden, _ = model.hidden(batch)
        loss = model.encoder.next_token_loss(hidden, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def finetune(model, inputs, targets):
    """Supervised training on the 32 labeled objects. All weights are updated."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(FINETUNE_STEPS):
        loss = F.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, inputs):
    n_objects = len(next(iter(inputs.values())))
    chunks = [model(select(inputs, rows)) for rows in torch.arange(n_objects).split(256)]
    return torch.cat(chunks).numpy()


def run(domain, data_root, out_root):
    start = time.perf_counter()
    make_encoder, block_size, parts = load(domain, data_root)
    train, tests = parts["53"], (parts["group1"], parts["group2"])
    center, spread = train["y"].mean(), train["y"].std()
    targets = torch.from_numpy((train["y"] - center) / spread)

    forest = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1)
    forest.fit(train["flat"], train["y"])
    predictions = [[forest.predict(test["flat"]) for test in tests]]

    pretrain_losses = []
    for use_pretraining in (False, True):
        torch.manual_seed(SEED)  # both models start from the same random weights
        model = SciGPT(make_encoder(), block_size, n_embd=N_EMBD)
        if use_pretraining:
            pretrain_losses = pretrain(model, parts["aux"]["inputs"])
        finetune(model, train["inputs"], targets)
        predictions.append(
            [predict(model, test["inputs"]) * spread + center for test in tests]
        )

    errors = [
        [float(np.abs(group - test["y"]).mean()) for group, test in zip(method, tests)]
        for method in predictions
    ]
    destination = Path(out_root) / domain
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "three_rows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["方法", "确认组 1 MAE", "确认组 2 MAE"])
        writer.writerows([name, *row] for name, row in zip(METHODS, errors))
    record = {
        "domain": domain,
        "seed": SEED,
        "labeled_objects": len(train["y"]),
        "unlabeled_objects": len(next(iter(parts["aux"]["inputs"].values()))),
        "test_sizes": [len(test["y"]) for test in tests],
        "tokens_per_object": block_size,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "mae": dict(zip(METHODS, errors)),
        "pretrain_loss_first_and_last_20_steps": [
            float(np.mean(pretrain_losses[:20])), float(np.mean(pretrain_losses[-20:])),
        ],
        "seconds": time.perf_counter() - start,
    }
    (destination / "receipt.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(domain)
    for name, (first, second) in zip(METHODS, errors):
        print(f"  {name:14} {first:.5f}  {second:.5f}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", choices=(*DOMAINS, "all"), required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_root", default="out/one-backbone")
    args = parser.parse_args()
    for domain in DOMAINS if args.domain == "all" else (args.domain,):
        run(domain, args.data_root, args.out_root)


if __name__ == "__main__":
    main()
