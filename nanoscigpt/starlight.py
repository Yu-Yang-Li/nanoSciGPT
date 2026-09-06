"""Fixed CPU StarLightCurves experiment using nanoGPT continuous patches.

Not TimesFM weights or a paper reproduction. Official preprocessing is retained.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .lightcurve_model import LightCurveGPT, patchify


class StarLightGPT(LightCurveGPT):
    """One continuous curve; same causal backbone for forecasting and classification."""
    def __init__(self, patch_size=32, width=64, layers=2, heads=4, max_length=1024, head_mode='mean_last'):
        super().__init__(patch_size, width, layers, heads, max_length)
        if head_mode not in ('mean_last', 'flat'):
            raise ValueError('Unknown classification head')
        self.config['head_mode'] = head_mode
        features = math.ceil(max_length / patch_size) * width if head_mode == 'flat' else 2 * width
        self.classifier = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, 3))

    def forward(self, curve):
        if self.config['head_mode'] == 'flat':
            h, _, _ = self.encode(curve)
            return self.classifier(h.flatten(1))
        return self.classifier(self.view_features(curve))


@dataclass
class Config:
    data: str
    output: str
    seed: int = 42
    pretrain_steps: int = 300
    finetune_steps: int = 400
    eval_every: int = 50
    batch_size: int = 32
    patch_size: int = 32
    width: int = 64
    layers: int = 2
    threads: int = 2
    head_mode: str = 'mean_last'
    cosine: bool = False
    validation_only: bool = False


def load_data(path, include_test=True):
    with np.load(path, allow_pickle=False) as a:
        x, y = a['x_train'].astype('float32'), a['y_train']
        fit, val = a['fit_indices'], a['validation_indices']
        xt, yt = (a['x_test'].astype('float32'), a['y_test']) if include_test else (None, None)
    if x.ndim != 2 or (include_test and (xt.ndim != 2 or x.shape[1] != xt.shape[1])):
        raise ValueError('Expected curve matrices of equal width')
    if not np.isfinite(x).all() or (include_test and not np.isfinite(xt).all()):
        raise ValueError('Unexpected non-finite values in packaged benchmark')
    if set(y) != {1, 2, 3} or (include_test and set(yt) != {1, 2, 3}):
        raise ValueError('Expected original labels 1, 2, 3')
    if len(x) != len(y) or (include_test and len(xt) != len(yt)):
        raise ValueError('Curve and label counts differ')
    if not np.array_equal(np.sort(np.concatenate([fit, val])), np.arange(len(x))):
        raise ValueError('Fit and validation must partition original training rows exactly')
    splits = [('fit', x[fit], y[fit]), ('val', x[val], y[val])]
    if include_test:
        splits.append(('test', xt, yt))
    return {name: (torch.from_numpy(values), torch.tensor(labels - 1, dtype=torch.long))
            for name, values, labels in splits}


def scores(y, probability):
    predicted = probability.argmax(-1)
    return dict(accuracy=float(accuracy_score(y, predicted)), macro_f1=float(f1_score(y, predicted, average='macro')),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                per_class=classification_report(y, predicted, labels=[0, 1, 2], target_names=['1', '2', '3'],
                                                output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist())


@torch.no_grad()
def probabilities(model, x, batch):
    model.eval()
    return torch.cat([model(part).softmax(-1) for part in x.split(batch)]).numpy()


@torch.no_grad()
def forecasting_mse(model, x, batch):
    model.eval()
    total, count = 0., 0
    for part in x.split(batch):
        values, valid = patchify(part, model.patch_size)
        keep = valid[:, 1:]
        total += ((model.predict_patches(part) - values[:, 1:]).square()[keep]).sum().item()
        count += keep.sum().item()
    return total / count


def run(config):
    if min(config.pretrain_steps, config.finetune_steps, config.eval_every, config.batch_size, config.threads) < 1:
        raise ValueError('Budgets must be positive')
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    start = time.perf_counter()
    data = load_data(config.data, include_test=not config.validation_only)
    x, y = data['fit']
    xv, yv = data['val']
    model = StarLightGPT(config.patch_size, config.width, config.layers, max_length=x.shape[1], head_mode=config.head_mode)
    initial = copy.deepcopy(model.state_dict())
    history, selected, states, validation = [], {}, {}, {}

    def record(**entry):
        entry['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        history.append(entry)
        line = json.dumps(entry)
        print(line, flush=True)
        with (out / 'run.log').open('a', encoding='utf-8') as log:
            log.write(line + '\n')

    def save(name, state, step):
        torch.save(dict(config=model.config, state=state, step=step), out / (name + '.pt'))

    def update(loss, optimizer):
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    parameters = sum(p.numel() for p in model.parameters())
    record(stage='start', rows={s: len(v[0]) for s, v in data.items()}, parameters=parameters,
           device='cpu', threads=config.threads, config=asdict(config))
    save('initial', initial, 0)
    initial_mse = forecasting_mse(model, xv, config.batch_size)
    best_mse, pretrained, selected['pretrained'] = initial_mse, copy.deepcopy(initial), 0
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    record(stage='pretraining', step=0, validation_mse=initial_mse)
    for step in range(1, config.pretrain_steps + 1):
        model.train()
        idx = rng.integers(0, len(x), config.batch_size)
        loss = model.pretraining_loss(x[idx])
        update(loss, optimizer)
        if step % config.eval_every == 0 or step == config.pretrain_steps:
            mse = forecasting_mse(model, xv, config.batch_size)
            if mse < best_mse:
                best_mse, pretrained, selected['pretrained'] = mse, copy.deepcopy(model.state_dict()), step
            record(stage='pretraining', step=step, train_loss=float(loss.detach()), validation_mse=mse)
    save('pretrained', pretrained, selected['pretrained'])
    classes = [torch.where(y == i)[0].numpy() for i in range(3)]
    branches = [('scratch', initial, config.pretrain_steps + config.finetune_steps),
                ('finetuned', pretrained, config.finetune_steps)]
    if config.cosine:
        # A long schedule's prefix is not the same optimizer schedule as a short
        # run. Train independent controls when learning rate depends on budget.
        branches = [('scratch', initial, config.finetune_steps),
                    ('scratch_long', initial, config.pretrain_steps + config.finetune_steps),
                    ('finetuned', pretrained, config.finetune_steps)]
    for branch, state, steps in branches:
        model.load_state_dict(state)
        # Identical classifier initialization and sampled labels in both branches.
        for key in initial:
            if key.startswith('classifier.'):
                model.state_dict()[key].copy_(initial[key])
        rng = np.random.default_rng(config.seed + 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
        best_f1, chosen, chosen_step, chosen_metrics = -1., None, 0, None
        for step in range(1, steps + 1):
            model.train()
            if config.cosine:
                lr = 5e-4 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step - 1) / max(1, steps - 1))))
                for group in optimizer.param_groups:
                    group['lr'] = lr
            # Uniform class sampling, then uniform example sampling within class.
            idx = np.array([rng.choice(classes[c]) for c in rng.integers(0, 3, config.batch_size)])
            loss = F.cross_entropy(model(x[idx]), y[idx])
            update(loss, optimizer)
            if step % config.eval_every == 0 or step in (config.finetune_steps, steps):
                metric = scores(yv.numpy(), probabilities(model, xv, config.batch_size))
                if metric['macro_f1'] > best_f1:
                    best_f1, chosen, chosen_step, chosen_metrics = metric['macro_f1'], copy.deepcopy(model.state_dict()), step, metric
                record(stage=branch, step=step, train_loss=float(loss.detach()),
                       learning_rate=optimizer.param_groups[0]['lr'],
                       validation_macro_f1=metric['macro_f1'], validation_accuracy=metric['accuracy'])
                if step == steps or (not config.cosine and step == config.finetune_steps):
                    name = 'scratch_long' if branch == 'scratch' and step > config.finetune_steps else branch
                    states[name], selected[name], validation[name] = copy.deepcopy(chosen), chosen_step, chosen_metrics
                    save(name, chosen, chosen_step)

    # All model selection is finished. Test set is evaluated only now.
    predictions, metrics = {}, {}
    if not config.validation_only:
        xt, yt = data['test']
        for name, state in states.items():
            model.load_state_dict(state)
            predictions[name] = probabilities(model, xt, config.batch_size)
            metrics[name] = scores(yt.numpy(), predictions[name])
        np.savez_compressed(out / 'predictions.npz', y_true=yt.numpy() + 1, **predictions)
    values, valid = patchify(xv, config.patch_size)
    target, keep = values[:, 1:], valid[:, 1:]
    naive = {name: float((prediction - target).square()[keep].mean()) for name, prediction in
             [('zero', torch.zeros_like(target)), ('last_value', values[:, :-1, -1:]), ('repeat_patch', values[:, :-1])]}
    # Fixed first validation example from each class, not selected by model quality.
    examples = torch.stack([torch.where(yv == c)[0][0] for c in range(3)])
    with torch.no_grad():
        model.load_state_dict(initial)
        before = model.predict_patches(xv[examples]).numpy()
        model.load_state_dict(pretrained)
        after = model.predict_patches(xv[examples]).numpy()
    np.savez_compressed(out / 'patch_examples.npz', curves=xv[examples].numpy(), labels=yv[examples].numpy() + 1,
                        validation_positions=examples.numpy(), before=before, after=after, patch_size=config.patch_size)
    record(stage='validation_complete' if config.validation_only else 'test',
           metrics={n: {k: s[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy')} for n, s in metrics.items()})
    sources = [Path(__file__), Path(__file__).with_name('lightcurve_model.py'), Path(__file__).parent / 'core' / 'gpt.py']
    result = dict(config=asdict(config), parameters=parameters, selected_steps=selected, metrics=metrics,
                  validation=validation, pretraining=dict(initial_validation_mse=initial_mse, best_validation_mse=best_mse, naive_validation_mse=naive),
                  budget=dict(pretraining=config.pretrain_steps, scratch=config.finetune_steps,
                              finetuned=config.finetune_steps, scratch_long=config.pretrain_steps + config.finetune_steps),
                  elapsed_seconds=round(time.perf_counter() - start, 3),
                  versions=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__),
                  data_sha256=hashlib.sha256(Path(config.data).read_bytes()).hexdigest(),
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  boundary='Preprocessed UCR benchmark; not raw survey classification or TimesFM reproduction. Pretraining uses only 800 fit curves, without labels. Validation selects checkpoints. Patch predictions use observed prefixes, not free-running forecasts. No guarantee of physical-time alignment. Long scratch matches update count, not measured FLOPs.')
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--head-mode', choices=['mean_last', 'flat'], default='mean_last')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    for name in ('seed', 'pretrain_steps', 'finetune_steps', 'eval_every', 'batch_size', 'patch_size', 'width', 'layers', 'threads'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=Config.__dataclass_fields__[name].default)
    run(Config(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
