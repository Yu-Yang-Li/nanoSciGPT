"""Fine-tune the existing structured encoders without replacing pretraining files."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..scientific.models import CrystalGraphEncoder, PatchEncoder, masked_mean
from .structured_demo import array_tokens, load_meta, validate_structured_fixture


def run(domain, ckpt_path, data_root, out_dir, epochs, max_samples, seed):
    from .downstream_demo import task_data_fingerprint

    validate_structured_fixture(domain, data_root)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint["domain"] != domain:
        raise ValueError("checkpoint domain does not match the selected data")
    meta = load_meta(data_root, domain)
    provenance = {
        "parent_checkpoint_sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
        "task_sampling": {"max_samples": max_samples},
        "data_fingerprint": task_data_fingerprint(Path(data_root) / domain),
    }
    continuing_task = "head" in checkpoint and "target_mean" in checkpoint
    if continuing_task:
        if checkpoint.get("task_sampling") != provenance["task_sampling"]:
            raise ValueError("task sampling changed; keep the same sample limit when continuing this task")
        if checkpoint.get("data_fingerprint") != provenance["data_fingerprint"]:
            raise ValueError("task data changed or this checkpoint lacks data provenance")
    with np.load(Path(data_root) / domain / "fixture.npz") as source:
        data = {key: source[key] for key in source.files}
    train_idx = np.linspace(0, len(data["train_y"]) - 1, min(max_samples, len(data["train_y"])), dtype=int)
    val_idx = np.linspace(0, len(data["val_y"]) - 1, min(max(16, max_samples // 4), len(data["val_y"])), dtype=int)
    train_y = torch.as_tensor(data["train_y"][train_idx], dtype=torch.float32)
    val_y = torch.as_tensor(data["val_y"][val_idx], dtype=torch.float32)
    torch.manual_seed(seed)
    weights = checkpoint["model"]
    if domain == "crystal":
        hidden_dim = weights["atom_embedding.weight"].size(1)
        model = CrystalGraphEncoder(hidden_dim, weights["radial_centers"].numel())

        def inputs(split, indices):
            return tuple(torch.as_tensor(data[f"{split}_{name}"][indices]) for name in
                         ("atomic_numbers", "fractional", "lattice", "mask"))

        train_x, val_x = inputs("train", train_idx), inputs("val", val_idx)

        def features(values):
            return masked_mean(model.encode(*values), values[-1])
    else:
        hidden_dim, input_dim = weights["input_projection.weight"].shape
        model = PatchEncoder(input_dim, weights["position"].size(1), hidden_dim)

        def inputs(split, indices):
            tokens = array_tokens(domain, data[f"{split}_x"][indices], meta.get("patch_size", 1))
            return ((tokens - checkpoint["token_mean"]) / checkpoint["token_scale"],)

        train_x, val_x = inputs("train", train_idx), inputs("val", val_idx)

        def features(values):
            return model.encode(values[0]).mean(dim=1)

    model.load_state_dict(weights)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    head = nn.Linear(hidden_dim, 1)
    mean, scale = train_y.mean(), train_y.std().clamp(min=1e-6)
    if continuing_task:
        head.load_state_dict(checkpoint["head"])
        mean, scale = checkpoint["target_mean"], checkpoint["target_scale"]
    targets = (train_y - mean) / scale
    parameters = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4)

    def evaluate():
        model.eval()
        with torch.no_grad():
            prediction = head(features(val_x)).flatten() * scale + mean
            return float((prediction - val_y).abs().mean())

    before = evaluate()
    for _ in range(epochs):
        model.train()
        for indices in torch.randperm(len(train_y)).split(8):
            prediction = head(features(tuple(value[indices] for value in train_x))).flatten()
            loss = nn.functional.mse_loss(prediction, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    after = evaluate()
    delta = sum(float((value.detach() - original[name]).square().sum()) for name, value in model.named_parameters()) ** 0.5
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "finetuned.pt"
    torch.save({**checkpoint, **provenance, "model": model.state_dict(), "head": head.state_dict(),
                "target_mean": mean, "target_scale": scale}, destination)
    result = {"status": "completed", "domain": domain, "task_name": meta["task_name"],
              "task_type": "regression", "label_source": "recorded parameter from deterministic teaching generator",
              "metric_name": "mae", "metric_value": after, "metric_before_finetune": before,
              "target_unit": meta["target_unit"], "train_samples": len(train_y), "val_samples": len(val_y),
              "encoder_frozen": False, "pretrained_parameters_updated": delta > 0, "encoder_delta_l2": delta,
              "adaptation": "finetune", "task_checkpoint": str(destination.resolve()), "teaching_only": True}
    result["head_initialization"] = ("continued with saved target normalization" if continuing_task else
                                      "new task head; source checkpoint has no saved target normalization")
    (out_dir / "downstream_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result
