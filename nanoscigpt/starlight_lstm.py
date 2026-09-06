"""Supervised patch-LSTM baseline for the same StarLightCurves classroom split."""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .starlight import load_data, probabilities, scores


class LightCurveLSTM(nn.Module):
    def __init__(self, patch_size=32, width=64, layers=2):
        super().__init__()
        self.config = dict(patch_size=patch_size, width=width, layers=layers)
        self.lstm = nn.LSTM(patch_size, width, num_layers=layers, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))

    def forward(self, curve):
        patch = self.config['patch_size']
        if curve.ndim != 2 or curve.shape[1] % patch:
            raise ValueError('Curve length must be divisible by patch size')
        _, (hidden, _) = self.lstm(curve.reshape(len(curve), -1, patch))
        return self.classifier(hidden[-1])


@dataclass
class Config:
    data: str
    output: str
    seed: int = 45
    steps: int = 800
    eval_every: int = 50
    batch_size: int = 32
    patch_size: int = 32
    width: int = 64
    layers: int = 2
    threads: int = 2
    learning_rate: float = .001


def run(config):
    if min(config.steps, config.eval_every, config.batch_size, config.patch_size,
           config.width, config.layers, config.threads, config.learning_rate) <= 0:
        raise ValueError('Sizes and budgets must be positive')
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    start = time.perf_counter()
    data = load_data(config.data)
    x, y = data['fit']
    xv, yv = data['val']
    model = LightCurveLSTM(config.patch_size, config.width, config.layers)
    classes = [torch.where(y == c)[0].numpy() for c in range(3)]
    rng = np.random.default_rng(config.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    initial = copy.deepcopy(model.state_dict())
    history, best, selected = [], -1., 0
    chosen = initial

    def record(**entry):
        history.append(entry)
        line = json.dumps(entry)
        print(line, flush=True)
        with (out / 'run.log').open('a', encoding='utf-8') as log:
            log.write(line + '\n')

    record(stage='start', config=asdict(config), device='cpu',
           parameters=sum(p.numel() for p in model.parameters()))
    for step in range(1, config.steps + 1):
        model.train()
        lr = config.learning_rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step - 1) / max(1, config.steps - 1))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        idx = np.array([rng.choice(classes[c]) for c in rng.integers(0, 3, config.batch_size)])
        loss = F.cross_entropy(model(x[idx]), y[idx])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % config.eval_every == 0 or step == config.steps:
            metric = scores(yv.numpy(), probabilities(model, xv, config.batch_size))
            if metric['macro_f1'] > best:
                best, selected, chosen = metric['macro_f1'], step, copy.deepcopy(model.state_dict())
            record(stage='classification', step=step, train_loss=float(loss.detach()),
                   validation_macro_f1=metric['macro_f1'], validation_accuracy=metric['accuracy'], learning_rate=lr)

    model.load_state_dict(chosen)
    torch.save(dict(config=model.config, state=chosen, step=selected), out / 'lstm.pt')
    xt, yt = data['test']
    prediction = probabilities(model, xt, config.batch_size)
    metric = scores(yt.numpy(), prediction)
    np.savez_compressed(out / 'predictions.npz', y_true=yt.numpy() + 1, lstm=prediction)
    report = dict(config=asdict(config), fit_samples=len(x), validation_samples=len(xv), test_samples=len(xt),
        parameters=sum(p.numel() for p in model.parameters()), selected_step=selected,
        validation=scores(yv.numpy(), probabilities(model, xv, config.batch_size)), metrics=metric,
        elapsed_seconds=round(time.perf_counter() - start, 3),
        data_sha256=hashlib.sha256(Path(config.data).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        boundary='Supervised patch-LSTM; no pretraining. Same split and 32-point patches as classroom GPT. Validation selects checkpoint. Architecture and optimization differ, so this is not an isolated pretraining ablation.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    record(stage='test', metrics=metric, selected_step=selected, elapsed_seconds=report['elapsed_seconds'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    for name in ('seed', 'steps', 'eval_every', 'batch_size', 'patch_size', 'width', 'layers', 'threads'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=Config.__dataclass_fields__[name].default)
    parser.add_argument('--learning-rate', type=float, default=.001)
    run(Config(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
