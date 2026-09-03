"""Prepare a user NPZ for the structured nanoSciGPT classroom path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .domains.registry import get_domain_spec


SUPPORTED_DOMAINS = ("weather", "crystal", "image", "spectrum", "field", "structure3d")
PRETRAINING_NAMES = {
    "weather": "masked patch reconstruction",
    "crystal": "masked atomic-number reconstruction",
    "image": "masked image-patch reconstruction",
    "spectrum": "masked wavelength-patch reconstruction",
    "field": "masked field-patch reconstruction",
    "structure3d": "masked distance-row reconstruction",
}

CRYSTAL_ARRAYS = {
    "train_atomic_numbers",
    "val_atomic_numbers",
    "train_fractional",
    "val_fractional",
    "train_mask",
    "val_mask",
    "train_lattice",
    "val_lattice",
    "train_y",
    "val_y",
}


def validate_crystal_arrays(
    arrays: dict[str, np.ndarray], require_labels: bool = True
) -> None:
    required = CRYSTAL_ARRAYS if require_labels else CRYSTAL_ARRAYS - {"train_y", "val_y"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"crystal NPZ missing arrays: {sorted(missing)}")
    label_keys = {"train_y", "val_y"}
    if not require_labels and label_keys & set(arrays) and not label_keys <= set(arrays):
        raise ValueError("crystal NPZ must provide both train_y and val_y or neither")

    for split in ("train", "val"):
        atomic_numbers = arrays[f"{split}_atomic_numbers"]
        fractional = arrays[f"{split}_fractional"]
        mask = arrays[f"{split}_mask"]
        lattice = arrays[f"{split}_lattice"]
        target = arrays.get(f"{split}_y")
        if atomic_numbers.ndim != 2:
            raise ValueError(f"{split}_atomic_numbers expects (samples, max_atoms)")
        samples, max_atoms = atomic_numbers.shape
        if fractional.shape != (samples, max_atoms, 3):
            raise ValueError(f"{split}_fractional must match atomic numbers with xyz coordinates")
        if mask.shape != atomic_numbers.shape or mask.dtype != np.bool_:
            raise ValueError(f"{split}_mask must be a boolean array matching atomic numbers")
        if lattice.shape != (samples, 3, 3):
            raise ValueError(f"{split}_lattice expects (samples, 3, 3)")
        if target is not None and target.shape != (samples,):
            raise ValueError(f"{split}_y must be a one-dimensional regression target")
        if samples == 0 or max_atoms == 0 or not mask.any(axis=1).all():
            raise ValueError(f"{split} must contain samples with at least one active atom")
        if not np.issubdtype(atomic_numbers.dtype, np.integer):
            raise ValueError(f"{split}_atomic_numbers must use an integer dtype")
        active_numbers = atomic_numbers[mask]
        if np.any((active_numbers < 1) | (active_numbers > 118)):
            raise ValueError("active atomic numbers must be 1 through 118")
        if np.any(atomic_numbers[~mask] != 0):
            raise ValueError("padded atomic-number positions must be zero")
        numeric_values = [("fractional", fractional), ("lattice", lattice)]
        if target is not None:
            numeric_values.append(("y", target))
        for name, value in numeric_values:
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"{split}_{name} must contain finite numeric values")
        if np.any(np.abs(np.linalg.det(lattice)) < 1e-8):
            raise ValueError(f"{split}_lattice contains a singular cell")

    if arrays["train_atomic_numbers"].shape[1] != arrays["val_atomic_numbers"].shape[1]:
        raise ValueError("train and validation crystals must use the same max_atoms width")


def validate_arrays(
    domain: str,
    arrays: dict[str, np.ndarray],
    patch_size: int,
    require_labels: bool = True,
) -> None:
    if domain == "crystal":
        validate_crystal_arrays(arrays, require_labels=require_labels)
        return
    required = {"train_x", "val_x"}
    if require_labels:
        required |= {"train_y", "val_y"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"NPZ missing arrays: {sorted(missing)}")

    train_x, val_x = arrays["train_x"], arrays["val_x"]
    label_keys = {"train_y", "val_y"}
    if not require_labels and label_keys & set(arrays) and not label_keys <= set(arrays):
        raise ValueError("NPZ must provide both train_y and val_y or neither")
    train_y, val_y = arrays.get("train_y"), arrays.get("val_y")
    if train_y is not None:
        if len(train_x) != len(train_y) or len(val_x) != len(val_y):
            raise ValueError("train_x/train_y and val_x/val_y must have matching lengths")
        if train_y.ndim != 1 or val_y.ndim != 1:
            raise ValueError("train_y and val_y must be one-dimensional regression targets")
    if not len(train_x) or not len(val_x):
        raise ValueError("training and validation arrays must both be non-empty")
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("train_x and val_x must have the same sample shape")
    if not all(np.issubdtype(value.dtype, np.number) for value in arrays.values()):
        raise ValueError("NPZ arrays must use numeric dtypes")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("NPZ arrays must contain only finite numeric values")

    if domain in {"weather", "image"}:
        if train_x.ndim != 4:
            raise ValueError(f"{domain} expects (samples, channels, height, width)")
        if train_x.shape[-2] % patch_size or train_x.shape[-1] % patch_size:
            raise ValueError("height and width must be divisible by patch size")
    elif domain in {"spectrum", "field"}:
        if train_x.ndim != 3:
            raise ValueError(f"{domain} expects (samples, channels, length)")
        if train_x.shape[-1] % patch_size:
            raise ValueError("sequence length must be divisible by patch size")
    elif domain == "structure3d":
        if train_x.ndim != 3 or train_x.shape[-1] != 3:
            raise ValueError("structure3d expects (samples, points, 3)")


def prepare(
    domain: str,
    source: Path,
    out_dir: Path,
    patch_size: int,
    task_name: str | None,
    sample_unit: str,
    target_unit: str | None,
    skip_downstream: bool = False,
    overwrite: bool = False,
) -> Path:
    source = source.resolve()
    out_dir = out_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    fixture_path = out_dir / "fixture.npz"
    meta_path = out_dir / "meta.json"
    if not overwrite and (fixture_path.exists() or meta_path.exists()):
        raise FileExistsError(f"prepared data already exists under {out_dir}")

    with np.load(source) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_arrays(domain, arrays, patch_size, require_labels=not skip_downstream)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixture_path, **arrays)
    spec = get_domain_spec(domain)
    train_inputs = arrays["train_atomic_numbers"] if domain == "crystal" else arrays["train_x"]
    val_inputs = arrays["val_atomic_numbers"] if domain == "crystal" else arrays["val_x"]
    has_labels = "train_y" in arrays and "val_y" in arrays
    meta = {
        "mode": "structured",
        "teaching_only": False,
        "source": str(source),
        "source_kind": "user_file",
        "has_labels": has_labels,
        "train_samples": int(len(train_inputs)),
        "val_samples": int(len(val_inputs)),
        "representation": spec.representation,
        "pretraining": PRETRAINING_NAMES[domain],
        "sample_unit": sample_unit,
        "patch_size": patch_size,
    }
    if has_labels:
        meta["label_source"] = "user_provided"
    if not skip_downstream:
        meta.update(
            {
                "task_name": task_name,
                "task_type": "regression",
                "target_unit": target_unit,
            }
        )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare labeled user arrays for a structured nanoSciGPT lesson."
    )
    parser.add_argument("--domain", required=True, choices=SUPPORTED_DOMAINS)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--task-name")
    parser.add_argument("--sample-unit", required=True)
    parser.add_argument("--target-unit")
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.patch_size < 1:
        parser.error("--patch-size must be at least 1")
    if not args.skip_downstream and (not args.task_name or not args.target_unit):
        parser.error("--task-name and --target-unit are required unless --skip-downstream is used")
    try:
        path = prepare(
            args.domain,
            args.npz,
            args.out_dir,
            args.patch_size,
            args.task_name,
            args.sample_unit,
            args.target_unit,
            args.skip_downstream,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(f"prepared {args.domain} user data -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
