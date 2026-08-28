"""Prepare DNA sequences from a local FASTA file (stream mode)."""

import argparse
import json
from pathlib import Path

import numpy as np

from ...core.tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fasta", required=True)
    p.add_argument("--num_bases", type=int, default=500_000)
    args = p.parse_args()

    domain_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / "dna"
    domain_dir.mkdir(parents=True, exist_ok=True)

    seq = []
    for line in Path(args.fasta).read_text(encoding="utf-8").splitlines():
        if not line.startswith(">"):
            seq.append(line.strip().upper())
    seq = "".join(seq)
    # skip leading N-run (unsequenced telomere region) to get real bases
    first_real = next((i for i, c in enumerate(seq) if c in "ACGT"), len(seq))
    seq = seq[first_real : first_real + args.num_bases]
    seq = "".join(c for c in seq if c in "ACGT")
    print(f"loaded {len(seq):,} bases from {args.fasta}")

    tok = CharTokenizer(set("ACGT"))
    tok.save(domain_dir / "tokenizer.json")
    n = len(seq)
    train_ids = np.array(tok.encode(seq[: int(n * 0.9)]), dtype=np.uint16)
    val_ids = np.array(tok.encode(seq[int(n * 0.9) :]), dtype=np.uint16)
    train_ids.tofile(domain_dir / "train.bin")
    val_ids.tofile(domain_dir / "val.bin")
    with open(domain_dir / "meta.json", "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "mode": "stream"}, f)
    print(f"vocab={tok.vocab_size} train={len(train_ids):,} val={len(val_ids):,} -> {domain_dir}")


if __name__ == "__main__":
    main()
