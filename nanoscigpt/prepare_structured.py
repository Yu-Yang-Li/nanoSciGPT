"""Prepare a user NPZ for the structured nanoSciGPT classroom path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .domains.registry import get_domain_spec


SUPPORTED_DOMAINS = ("weather", "image", "spectrum", "field", "structure3d")
PRETRAINING_NAMES = {
    "weather": "masked patch reconstruction",
    "image": "masked image-patch reconstruction",
    "spectrum": "masked wavelength-patch reconstruction",
    "field": "masked field-patch reconstruction",
    "structure3d": "masked distance-row reconstruction",
}


def validate_arrays(domain: str, arrays: dict[str, np.ndarray], patch_size: int) -> None:
    required = {"train_x", "val_x", "train_y", "val_y"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"NPZ missing arrays: {sorted(missing)}")

    train_x, val_x = arrays["train_x"], arrays["val_x"]
    train_y, val_y = arrays["train_y"], arrays["val_y"]
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("train_x/train_y and val_x/val_y must have matching lengths")
    if not len(train_x) or not len(val_x):
        raise ValueError("training and validation arrays must both be non-empty")
    if train_y.ndim != 1 or val_y.ndim != 1:
        raise ValueError("train_y and val_y must be one-dimensional regression targets")
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
    task_name: str,
    sample_unit: str,
    target_unit: str,
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
    validate_arrays(domain, arrays, patch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixture_path, **arrays)
    spec = get_domain_spec(domain)
    meta = {
        "mode": "structured",
        "teaching_only": False,
        "source": str(source),
        "source_kind": "user_file",
        "label_source": "user_provided",
        "train_samples": int(len(arrays["train_y"])),
        "val_samples": int(len(arrays["val_y"])),
        "representation": spec.representation,
        "pretraining": PRETRAINING_NAMES[domain],
        "task_name": task_name,
        "task_type": "regression",
        "sample_unit": sample_unit,
        "target_unit": target_unit,
        "patch_size": patch_size,
    }
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
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--sample-unit", required=True)
    parser.add_argument("--target-unit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.patch_size < 1:
        parser.error("--patch-size must be at least 1")
    try:
        path = prepare(
            args.domain,
            args.npz,
            args.out_dir,
            args.patch_size,
            args.task_name,
            args.sample_unit,
            args.target_unit,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(f"prepared {args.domain} user data -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
