"""Reproduce RNA scaling from the four bundled Rfam token panels on CPU."""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys

import numpy as np
import torch
from nanoscigpt.scientific.encoders import SymbolEncoder
from nanoscigpt.scientific.scigpt import SciGPT
from nanoscigpt.tasks import one_backbone as training


def triplet_entropy(tokens):
    triples = zip(tokens, tokens[1:], tokens[2:])
    n = len(tokens) - 2
    return -sum((count / n) * math.log2(count / n) for count in Counter(triples).values())


def targets(panel):
    return np.array([triplet_entropy(row.tolist()) for row in panel], dtype=np.float32)


@torch.no_grad()
def token_loss(model, panel):
    total = 0.0
    for rows in panel.split(128):
        batch = {"tokens": rows}
        hidden, _ = model.hidden(batch)
        total += float(model.encoder.next_token_loss(hidden, batch)) * len(rows)
    return total / len(panel)  # Every test fragment has exactly 31 valid prediction targets.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("next-token", "entropy"), default="next-token")
    parser.add_argument("--full", action="store_true", help="repeat all four scales with the five registered seeds")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with np.load(Path(__file__).with_name("rna-tokens.npz")) as raw:
        panels = {name: torch.from_numpy(raw[name].astype(np.int64)) for name in raw.files}
    seeds = (53, 71, 89, 107, 131) if args.full else (53,)
    metric = "cross_entropy" if args.objective == "next-token" else "MAE_bits"
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["seed", "unlabeled_objects", "steps", "A_" + metric, "B_" + metric])
    groups = [panels["A"], panels["B"]]
    if args.objective == "entropy":
        truth = [targets(test) for test in groups]
    for seed in seeds:
        training.SEED = seed
        permutation = np.random.default_rng(seed).permutation(len(panels["unlabeled"]))
        if args.objective == "entropy":
            rows = np.random.default_rng(seed).choice(len(panels["pool"]), 32, replace=False)
            x = panels["pool"][rows]
            y = targets(x)
            center, spread = y.mean(), y.std()
            standard = torch.from_numpy((y - center) / spread)
        for size in (200, 800, 3200, 12800):
            torch.manual_seed(seed)
            model = SciGPT(SymbolEncoder(6, training.N_EMBD), 32, n_embd=training.N_EMBD)
            training.PRETRAIN_STEPS = 1600 if args.objective == "next-token" else size // 2
            training.pretrain(model, {"tokens": panels["unlabeled"][permutation[:size]]})
            if args.objective == "next-token":
                model.eval()
                scores = [token_loss(model, test) for test in groups]
            else:
                training.finetune(model, {"tokens": x}, standard)
                model.eval()
                scores = [np.abs(training.predict(model, {"tokens": test}) * spread + center - label).mean()
                          for test, label in zip(groups, truth)]
            writer.writerow([seed, size, training.PRETRAIN_STEPS, *scores])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
