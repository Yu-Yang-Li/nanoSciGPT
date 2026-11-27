"""CPU demonstration of pretraining data scale on simulated chain end-to-end distance."""

import argparse
import csv
import sys

import numpy as np
import torch

from ..scientific.encoders import PatchEncoder
from ..scientific.scigpt import SciGPT
from . import one_backbone as training


def chain_panel(count, stream):
    """Generate unit-bond chains. Withhold both blocks containing the target distance."""
    rng = np.random.default_rng(stream)
    persistence = rng.uniform(0.05, 0.93, (count, 1))
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = [np.zeros_like(direction)]
    for _ in range(15):
        noise = rng.normal(size=direction.shape)
        direction = persistence * direction + np.sqrt(1 - persistence**2) * noise / np.sqrt(3)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        points.append(points[-1] + direction)
    xyz = np.stack(points, axis=1)
    distance = np.linalg.norm(xyz[:, :, None] - xyz[:, None], axis=-1)
    target = distance[:, 0, -1].astype(np.float32)
    blocks = distance.reshape(count, 4, 4, 4, 4).transpose(0, 1, 3, 2, 4).reshape(count, 16, 16)
    visible = [i for i in range(16) if i not in (3, 12)]
    return torch.from_numpy(np.ascontiguousarray(blocks[:, visible], dtype=np.float32)), target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use five seeds and all five sizes, including zero")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    unlabeled, _ = chain_panel(12800, [264130, 0])
    pool, targets = chain_panel(256, [264130, 1])
    holdouts = [chain_panel(480, [26112607, k]) for k in (31, 47)]
    seeds = (53, 71, 89, 107, 131) if args.full else (53,)
    sizes = (0, 200, 800, 3200, 12800) if args.full else (200, 800, 3200)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["seed", "unlabeled_objects", "C_MAE", "D_MAE"])
    for seed in seeds:
        training.SEED = seed
        rows = np.random.default_rng(seed).choice(len(targets), 32, replace=False)
        order = np.random.default_rng(seed).permutation(len(unlabeled))
        train_x = pool[rows]
        center = train_x.mean((0, 1))
        scale = train_x.std((0, 1), unbiased=False).clamp_min(1e-6)
        train_x = {"patches": (train_x - center) / scale}
        train_y = targets[rows]
        mean, std = train_y.mean(), train_y.std()
        normalized_y = torch.from_numpy((train_y - mean) / std)
        for n in sizes:
            torch.manual_seed(seed)
            model = SciGPT(PatchEncoder(16, training.N_EMBD), 16, n_embd=training.N_EMBD)
            if n:
                training.PRETRAIN_STEPS = n // 2
                training.pretrain(model, {"patches": (unlabeled[order[:n]] - center) / scale})
            training.finetune(model, train_x, normalized_y)
            model.eval()
            errors = []
            for inputs, truth in holdouts:
                predicted = training.predict(model, {"patches": (inputs - center) / scale})
                errors.append(np.abs(predicted * std + mean - truth).mean())
            writer.writerow([seed, n, *errors])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
