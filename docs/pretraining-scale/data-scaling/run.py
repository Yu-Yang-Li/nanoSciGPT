"""Reproduce a CPU pretraining curve from a prepared nanoSciGPT input panel."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from nanoscigpt.scientific.encoders import CrystalEncoder, PatchEncoder, SymbolEncoder
from nanoscigpt.scientific.scigpt import SciGPT
from nanoscigpt.tasks import one_backbone as train

SIZES = (200, 800, 3200, 12800)
SEEDS = (53, 71, 89, 107, 131)


def load_panel(directory, domain):
    meta = json.loads((directory / "manifest.json").read_text(encoding="utf8"))[domain]
    path = directory / f"{domain}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["panel_sha256"]:
        raise ValueError(f"Panel checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        groups = {g: {k.split("_", 1)[1]: torch.from_numpy(z[k]) for k in z if k.startswith(g + "_")}
                  for g in ("unlabeled", "pool", "A", "B")}
    return meta, groups


def normalized(groups, kind, seed):
    if kind == "symbol":
        return groups
    inputs = groups["pool"]
    rows = np.random.default_rng(seed).choice(len(next(iter(inputs.values()))), 32, replace=False)
    if kind == "crystal":
        unit = inputs["lattice"][rows].abs().max()
        return {g: {**values, "lattice": values["lattice"] / unit} for g, values in groups.items()}
    calibration = inputs["patches"][rows]
    center = calibration.mean((0, 1))
    scale = calibration.std((0, 1), unbiased=False).clamp_min(1e-6)
    return {g: {"patches": values["patches"].sub(center).div(scale)} for g, values in groups.items()}


def network(meta):
    if meta["kind"] == "symbol":
        encoder = SymbolEncoder(meta["vocab"], train.N_EMBD)
    elif meta["kind"] == "crystal":
        encoder = CrystalEncoder(train.N_EMBD)
    else:
        encoder = PatchEncoder(meta["width"], train.N_EMBD)
    return SciGPT(encoder, meta["positions"], n_embd=train.N_EMBD)


@torch.no_grad()
def held_out_loss(net, group):
    net.eval()
    n = len(next(iter(group.values())))
    loss_sum = 0.0
    targets = 0
    for start in range(0, n, 128):
        chunk = {k: x[start:start + 128] for k, x in group.items()}
        hidden = net.hidden(chunk)[0][:, :-1]
        if "patches" in chunk:
            expected = chunk["patches"][:, 1:]
            loss = F.mse_loss(net.encoder.to_patch(hidden), expected, reduction="sum")
            targets += expected.numel()
        else:
            if "elements" in chunk:
                expected = chunk["elements"]
                logits = net.encoder.to_element(hidden)
            else:
                expected = chunk["tokens"][:, 1:]
                logits = net.encoder.to_symbol(hidden)
            loss = F.cross_entropy(logits.flatten(0, 1), expected.flatten(), ignore_index=0, reduction="sum")
            targets += int((expected != 0).sum())
        loss_sum += float(loss)
    return loss_sum / targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--domain", required=True, choices=("spectrum", "protein", "dna", "rna", "smiles",
                        "crystal", "structure3d", "weather", "field", "image", "text"))
    parser.add_argument("--full", action="store_true", help="run all five seeds (default: seed 53)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    meta, groups = load_panel(args.data_dir, args.domain)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    labels = ("C", "D") if args.domain == "structure3d" else ("A", "B")
    writer.writerow(["domain", "seed", "unlabeled_objects", "steps", *labels])
    for seed in SEEDS if args.full else SEEDS[:1]:
        inputs = normalized(groups, meta["kind"], seed)
        order = torch.from_numpy(np.random.default_rng(seed).permutation(12800))
        train.SEED = seed
        train.PRETRAIN_STEPS = 1600
        for n in SIZES:
            torch.manual_seed(seed)
            net = network(meta)
            train.pretrain(net, train.select(inputs["unlabeled"], order[:n]))
            losses = [held_out_loss(net, inputs[g]) for g in ("A", "B")]
            writer.writerow([args.domain, seed, n, 1600, *losses])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
