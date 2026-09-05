"""Task-specific v1 template asset; run only from its prepared research folder.

The GPT class comes from the unchanged upstream nanoGPT template. The task head,
labels and normalization come from the student's fine-tuned checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from native_gpt import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="run_0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    root = Path(__file__).resolve().parent
    setup = json.loads((root / "task_setup.json").read_text(encoding="utf-8"))
    initial = torch.load(root / "initial_model.pt", map_location="cpu", weights_only=False)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {key: initial["model_args"][key] for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}
    encoder = GPT(GPTConfig(**config, bias=True, dropout=0.0))
    encoder.load_state_dict({key: value for key, value in initial["model"].items() if key in encoder.state_dict()}, strict=True)
    head = nn.Linear(config["n_embd"], initial["head"]["weight"].size(0))
    head.load_state_dict(initial["head"])
    with np.load(root / "task_data.npz", allow_pickle=False) as archive:
        data = {key: torch.from_numpy(archive[key]) for key in archive.files}
    regression = setup["task"]["task_type"] == "regression"
    mean, scale = initial["target_mean"], initial["target_scale"]

    def predict(x, padding):
        hidden = encoder.transformer.wte(x) + encoder.transformer.wpe(torch.arange(x.size(1)))
        for block in encoder.transformer.h:
            hidden = block(hidden)
        hidden = encoder.transformer.ln_f(hidden)
        keep = (~padding).float().unsqueeze(-1)
        return head((hidden * keep).sum(1) / keep.sum(1).clamp(min=1))

    def evaluate():
        encoder.eval()
        with torch.no_grad():
            predicted = torch.cat([predict(x, mask) for x, mask in
                                   zip(data["val_x"].split(8), data["val_pad"].split(8))])
            if regression:
                return float((predicted.flatten() * scale + mean - data["val_y"]).abs().mean())
            return float((predicted.argmax(-1) == data["val_y"]).float().mean())

    metric = "val_mae" if regression else "val_accuracy"
    before = evaluate()
    # These task-training choices are visible in experiment.py for the native
    # research agent to edit. Data, task_setup.json and the initial model stay fixed.
    learning_rate = 3e-4
    batch_size = 8
    parameters = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    losses = []
    for _ in range(setup["experiment_steps"]):
        encoder.train()
        indices = torch.randint(len(data["train_y"]), (min(batch_size, len(data["train_y"])),))
        predicted = predict(data["train_x"][indices], data["train_pad"][indices])
        target = data["train_y"][indices]
        loss = nn.functional.mse_loss(predicted.flatten(), (target - mean) / scale) if regression else nn.functional.cross_entropy(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    after = evaluate()
    # Retain fixed masks required by the course implementation, while replacing
    # every trained tensor with the native model's actual updated value.
    weights = {**initial["model"], **encoder.state_dict()}
    torch.save({**initial, "model": weights, "head": head.state_dict(),
                "parent_checkpoint_sha256": hashlib.sha256((root / "initial_model.pt").read_bytes()).hexdigest()},
               output / "checkpoint.pt")
    result = {"task": {"means": {"initial_" + metric: before, metric: after},
                       "stderrs": {}, "task": setup["task"], "runs": 1}}
    (output / "final_info.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "training_losses.json").write_text(json.dumps(losses), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
