"""Prepare a student's protein CSV without replacing bundled course data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core.tokenizer import CharTokenizer
from .domains.protein.prepare import AA


def prepare(source, data_root, sequence_column, target_column=None, split_column=None):
    source = Path(source).resolve()
    destination = Path(data_root).resolve() / "protein"
    if destination.exists():
        raise FileExistsError("choose a new data_root; previous data is preserved")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = [name for name in (sequence_column, target_column, split_column) if name]
        if not all(name in (reader.fieldnames or []) for name in required):
            raise ValueError("a requested CSV column is missing")
        rows = list(reader)
    sequences = [row[sequence_column].strip().upper() for row in rows]
    if not sequences or any(not value or set(value) - set(AA) for value in sequences):
        raise ValueError("protein input requires nonempty canonical amino-acid sequences; no rows were silently removed")
    if split_column:
        splits = [row[split_column].strip() for row in rows]
        if set(splits) != {"train", "val"}:
            raise ValueError("split column must contain train and val")
    else:
        groups = sorted(set(sequences))
        if len(groups) < 2:
            raise ValueError("need at least two distinct sequences to hold out data")
        order = np.random.default_rng(1337).permutation(len(groups))
        held_out = {groups[i] for i in order[:max(2, len(groups) // 5)]}
        splits = ["val" if sequence in held_out else "train" for sequence in sequences]
    if {s for s, split in zip(sequences, splits) if split == "train"} & {s for s, split in zip(sequences, splits) if split == "val"}:
        raise ValueError("the same sequence occurs in both train and val")
    labels = None
    if target_column:
        labels = np.asarray([float(row[target_column]) for row in rows], dtype=np.float32)
        if not np.isfinite(labels).all():
            raise ValueError("target values must be finite numbers; missing labels are not replaced")
    if any(splits.count(split) < 2 for split in ("train", "val")):
        raise ValueError("need at least two train and two val samples for this small lesson")
    tokenizer = CharTokenizer(set(AA) | {"<pad>", "<eos>"})
    meta = {"mode": "independent", "vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.stoi["<pad>"],
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "student_data": True, "target_column": target_column,
            "split": "supplied column" if split_column else "seeded exact-sequence-group holdout; not a homology split",
            "sequence_column": sequence_column, "task_type": "regression" if target_column else None}
    destination.mkdir(parents=True)
    tokenizer.save(destination / "tokenizer.json")
    for split in ("train", "val"):
        indices = [i for i, value in enumerate(splits) if value == split]
        encoded = np.empty(len(indices), dtype=object)
        encoded[:] = [np.asarray(tokenizer.encode(sequences[i]) + [tokenizer.stoi["<eos>"]], dtype=np.uint16) for i in indices]
        np.save(destination / f"{split}_seqs.npy", encoded)
        if labels is not None:
            np.save(destination / f"{split}_y.npy", labels[indices])
    (destination / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sequence-column", required=True)
    parser.add_argument("--target-column")
    parser.add_argument("--split-column")
    args = parser.parse_args()
    print(json.dumps(prepare(args.csv, args.data_root, args.sequence_column, args.target_column, args.split_column), ensure_ascii=False, indent=2))
