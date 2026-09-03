"""CPU pretraining-to-task flow for six structured scientific objects."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..scientific.adapters import patchify_1d, patchify_2d, pairwise_distance_tokens
from ..scientific.models import CrystalGraphEncoder, PatchEncoder, masked_mean
from ..domains.registry import STRUCTURED_DOMAINS

def load_meta(data_root, domain):
    path = Path(data_root) / domain / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_structured_fixture(domain, data_root="data", require_labels=True):
    if domain not in STRUCTURED_DOMAINS:
        raise ValueError(f"unsupported structured domain={domain}")
    data_dir = Path(data_root) / domain
    meta = load_meta(data_root, domain)
    fixture = data_dir / "fixture.npz"
    if not fixture.is_file():
        raise FileNotFoundError(f"missing structured fixture: {fixture}")
    with np.load(fixture) as values:
        if domain == "crystal":
            required = {
                "train_atomic_numbers",
                "val_atomic_numbers",
                "train_fractional",
                "val_fractional",
                "train_mask",
                "val_mask",
                "train_lattice",
                "val_lattice",
            }
        else:
            required = {"train_x", "val_x"}
        if require_labels:
            required |= {"train_y", "val_y"}
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"{domain} fixture missing arrays: {sorted(missing)}")
        if domain == "crystal":
            train_items = len(values["train_atomic_numbers"])
            val_items = len(values["val_atomic_numbers"])
        else:
            train_items = len(values["train_x"])
            val_items = len(values["val_x"])
        if require_labels and (
            len(values["train_y"]) != train_items or len(values["val_y"]) != val_items
        ):
            raise ValueError(f"{domain} inputs and labels are not aligned")
    if train_items == 0 or val_items == 0:
        raise ValueError(f"{domain} has an empty train/validation split")
    return {
        "status": "ready",
        "domain": domain,
        "mode": meta["mode"],
        "representation": meta["representation"],
        "train_items": train_items,
        "val_items": val_items,
        "task_name": meta.get("task_name"),
        "data_dir": str(data_dir.resolve()),
        "source_name": meta.get("source"),
        "source_kind": meta.get("source_kind", "generated_fixture"),
        "teaching_only": bool(meta.get("teaching_only", True)),
    }


def array_tokens(domain, values, patch_size):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if domain in ("weather", "image"):
        return patchify_2d(tensor, patch_size)
    if domain in ("spectrum", "field"):
        return patchify_1d(tensor, patch_size)
    if domain == "structure3d":
        return pairwise_distance_tokens(tensor)
    raise ValueError(f"no array adapter for domain={domain}")


def normalize_tokens(train_tokens, val_tokens):
    mean = train_tokens.mean(dim=(0, 1), keepdim=True)
    scale = train_tokens.std(dim=(0, 1), keepdim=True).clamp(min=1e-5)
    return (train_tokens - mean) / scale, (val_tokens - mean) / scale, mean, scale


def fit_regression_head(train_features, train_y, val_features, val_y, steps, seed):
    torch.manual_seed(seed)
    mean = train_y.mean()
    scale = train_y.std().clamp(min=1e-6)
    target = ((train_y - mean) / scale).unsqueeze(-1)
    head = nn.Linear(train_features.size(-1), 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.04)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(head(train_features), target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = head(val_features).squeeze(-1) * scale + mean
        mae = torch.mean(torch.abs(prediction - val_y)).item()
    return head, round(mae, 6)


def run_patch_domain(
    domain, fixture, meta, pretrain_steps, task_steps, seed, skip_downstream=False
):
    train_tokens = array_tokens(domain, fixture["train_x"], meta.get("patch_size", 1))
    val_tokens = array_tokens(domain, fixture["val_x"], meta.get("patch_size", 1))
    train_tokens, val_tokens, token_mean, token_scale = normalize_tokens(train_tokens, val_tokens)

    torch.manual_seed(seed)
    model = PatchEncoder(train_tokens.size(-1), train_tokens.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(pretrain_steps):
        masked = torch.rand(train_tokens.shape[:2], generator=generator) < 0.25
        masked[:, 0] = True
        optimizer.zero_grad()
        reconstruction = model.reconstruct(train_tokens, masked)
        loss = nn.functional.mse_loss(reconstruction[masked], train_tokens[masked])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    validation_generator = torch.Generator().manual_seed(seed + 1)
    validation_masked = torch.rand(val_tokens.shape[:2], generator=validation_generator) < 0.25
    validation_masked[:, 0] = True
    with torch.no_grad():
        validation_reconstruction = model.reconstruct(val_tokens, validation_masked)
        validation_loss = float(
            nn.functional.mse_loss(
                validation_reconstruction[validation_masked], val_tokens[validation_masked]
            )
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        train_features = model.encode(train_tokens).mean(dim=1)
        val_features = model.encode(val_tokens).mean(dim=1)
    head, mae = None, None
    if not skip_downstream:
        train_y = torch.as_tensor(fixture["train_y"], dtype=torch.float32)
        val_y = torch.as_tensor(fixture["val_y"], dtype=torch.float32)
        head, mae = fit_regression_head(
            train_features, train_y, val_features, val_y, task_steps, seed
        )
    preview = {
        "raw_shape": list(fixture["train_x"].shape[1:]),
        "token_shape": list(train_tokens.shape[1:]),
        "representation": meta["representation"],
    }
    checkpoint = {
        "domain": domain,
        "model": model.state_dict(),
        "token_mean": token_mean,
        "token_scale": token_scale,
        "representation": meta["representation"],
    }
    if head is not None:
        checkpoint["head"] = head.state_dict()
    return checkpoint, losses, validation_loss, mae, preview


def crystal_mask_positions(node_mask):
    positions = torch.zeros_like(node_mask)
    for row in range(node_mask.size(0)):
        index = int(torch.nonzero(node_mask[row], as_tuple=False)[0, 0])
        positions[row, index] = True
    return positions


def run_crystal_domain(
    fixture, meta, pretrain_steps, task_steps, seed, skip_downstream=False
):
    train_z = torch.as_tensor(fixture["train_atomic_numbers"], dtype=torch.long)
    val_z = torch.as_tensor(fixture["val_atomic_numbers"], dtype=torch.long)
    train_frac = torch.as_tensor(fixture["train_fractional"], dtype=torch.float32)
    val_frac = torch.as_tensor(fixture["val_fractional"], dtype=torch.float32)
    train_mask = torch.as_tensor(fixture["train_mask"], dtype=torch.bool)
    val_mask = torch.as_tensor(fixture["val_mask"], dtype=torch.bool)
    train_lattice = torch.as_tensor(fixture["train_lattice"], dtype=torch.float32)
    val_lattice = torch.as_tensor(fixture["val_lattice"], dtype=torch.float32)

    torch.manual_seed(seed)
    model = CrystalGraphEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    masked = crystal_mask_positions(train_mask)
    masked_z = train_z.clone()
    masked_z[masked] = 0
    losses = []
    for _ in range(pretrain_steps):
        optimizer.zero_grad()
        logits = model.classify_atoms(masked_z, train_frac, train_lattice, train_mask)
        loss = nn.functional.cross_entropy(logits[masked], train_z[masked])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    val_masked = crystal_mask_positions(val_mask)
    masked_val_z = val_z.clone()
    masked_val_z[val_masked] = 0
    with torch.no_grad():
        validation_logits = model.classify_atoms(masked_val_z, val_frac, val_lattice, val_mask)
        validation_loss = float(
            nn.functional.cross_entropy(validation_logits[val_masked], val_z[val_masked])
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        train_features = masked_mean(model.encode(train_z, train_frac, train_lattice, train_mask), train_mask)
        val_features = masked_mean(model.encode(val_z, val_frac, val_lattice, val_mask), val_mask)
    head, mae = None, None
    if not skip_downstream:
        train_y = torch.as_tensor(fixture["train_y"], dtype=torch.float32)
        val_y = torch.as_tensor(fixture["val_y"], dtype=torch.float32)
        head, mae = fit_regression_head(
            train_features, train_y, val_features, val_y, task_steps, seed
        )
    preview = {
        "raw_shape": list(fixture["train_fractional"].shape[1:]),
        "max_nodes": int(fixture["train_atomic_numbers"].shape[1]),
        "representation": meta["representation"],
    }
    checkpoint = {
        "domain": "crystal",
        "model": model.state_dict(),
        "representation": meta["representation"],
    }
    if head is not None:
        checkpoint["head"] = head.state_dict()
    return checkpoint, losses, validation_loss, mae, preview


def run_structured(
    domain,
    data_root,
    out_dir,
    pretrain_steps=20,
    task_steps=20,
    seed=1337,
    skip_downstream=False,
):
    validate_structured_fixture(domain, data_root, require_labels=not skip_downstream)
    data_dir = Path(data_root) / domain
    meta = load_meta(data_root, domain)
    with np.load(data_dir / "fixture.npz") as source:
        fixture = {key: source[key] for key in source.files}
    if domain == "crystal":
        checkpoint, losses, validation_loss, mae, preview = run_crystal_domain(
            fixture, meta, pretrain_steps, task_steps, seed, skip_downstream
        )
    else:
        checkpoint, losses, validation_loss, mae, preview = run_patch_domain(
            domain, fixture, meta, pretrain_steps, task_steps, seed, skip_downstream
        )

    out_dir = Path(out_dir)
    model_dir = out_dir / "model"
    downstream_dir = out_dir / "downstream"
    model_dir.mkdir(parents=True, exist_ok=True)
    if not skip_downstream:
        downstream_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_dir / "ckpt.pt")
    train_log = {
        "domain": domain,
        "pretraining": meta["pretraining"],
        "pretrain_steps": pretrain_steps,
        "pretrain_loss_start": round(losses[0], 6),
        "pretrain_loss_end": round(losses[-1], 6),
        "pretrain_val_loss": round(validation_loss, 6),
    }
    (model_dir / "train_log.json").write_text(
        json.dumps(train_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "representation_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"domain={domain} representation={meta['representation']}")
    print(f"pretraining: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"fixed validation loss: {validation_loss:.4f}")
    if skip_downstream:
        print("downstream task: not requested")
        return None
    result = {
        "status": "completed",
        "domain": domain,
        "task_name": meta["task_name"],
        "task_type": meta["task_type"],
        "label_source": meta.get(
            "label_source", "recorded parameter from deterministic teaching generator"
        ),
        "metric_name": "mae",
        "metric_value": mae,
        "target_unit": meta["target_unit"],
        "train_samples": int(meta["train_samples"]),
        "val_samples": int(meta["val_samples"]),
        "encoder_frozen": True,
        "pretrained_parameters_updated": False,
        "teaching_only": bool(meta.get("teaching_only", True)),
    }
    result_path = downstream_dir / "downstream_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("downstream task: completed")
    print(f"task: {meta['task_name']}")
    print(f"result saved: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=STRUCTURED_DOMAINS)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--pretrain_steps", type=int, default=20)
    parser.add_argument("--task_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir or Path("out") / "classroom" / args.domain
    run_structured(
        args.domain,
        args.data_root,
        out_dir,
        pretrain_steps=args.pretrain_steps,
        task_steps=args.task_steps,
        seed=args.seed,
        skip_downstream=args.skip_downstream,
    )


if __name__ == "__main__":
    main()
