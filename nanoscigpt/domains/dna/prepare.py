"""Prepare DNA sequences from a local FASTA file (stream mode)."""

import argparse
import json
from pathlib import Path

import numpy as np

from ...core.tokenizer import CharTokenizer

SLICE_NAME = "chr21_slice.fa"


def synthetic_dna_fallback(n: int, seed: int = 7) -> str:
    """Offline teaching fallback: a deterministic synthetic chromosome.

    Not real genomic data. It only lets the pipeline run end-to-end when no
    FASTA is available, so students can observe loss decreasing and sampling.
    """
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fasta", default=None, help="local FASTA; if omitted, uses the bundled chr21 teaching slice")
    p.add_argument("--num_bases", type=int, default=500_000)
    p.add_argument("--allow_synthetic_fallback", action="store_true",
                   help="if no FASTA at all, use deterministic synthetic DNA (teaching only)")
    args = p.parse_args()

    domain_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / "dna"
    domain_dir.mkdir(parents=True, exist_ok=True)

    if args.fasta:
        fasta_path = Path(args.fasta).resolve()
        if not fasta_path.exists():
            raise SystemExit(f"FASTA not found: {fasta_path}")
        source = str(fasta_path)
        source_kind = "user_file"
    else:
        fasta_path = domain_dir / SLICE_NAME
        source = str(fasta_path.resolve())
        source_kind = "bundled_file"

    synthetic = False
    if fasta_path.exists():
        seq = []
        for line in fasta_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(">"):
                seq.append(line.strip().upper())
        seq = "".join(seq)
        # skip leading N-run (unsequenced telomere region) to get real bases
        first_real = next((i for i, c in enumerate(seq) if c in "ACGT"), len(seq))
        seq = seq[first_real : first_real + args.num_bases]
        seq = "".join(c for c in seq if c in "ACGT")
    elif args.allow_synthetic_fallback:
        seq = synthetic_dna_fallback(args.num_bases)
        synthetic = True
        source = "deterministic_synthetic_dna"
        source_kind = "synthetic_fixture"
        print("WARNING: no FASTA found; using deterministic SYNTHETIC DNA (teaching only)")
    else:
        raise SystemExit(
            f"no FASTA found at {fasta_path}; bundle {SLICE_NAME} or pass --fasta / --allow_synthetic_fallback"
        )

    print(f"loaded {len(seq):,} bases from {fasta_path}" + (" (synthetic)" if synthetic else ""))

    tok = CharTokenizer(set("ACGT"))
    tok.save(domain_dir / "tokenizer.json")
    n = len(seq)
    train_ids = np.array(tok.encode(seq[: int(n * 0.9)]), dtype=np.uint16)
    val_ids = np.array(tok.encode(seq[int(n * 0.9) :]), dtype=np.uint16)
    train_ids.tofile(domain_dir / "train.bin")
    val_ids.tofile(domain_dir / "val.bin")
    with open(domain_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": tok.vocab_size,
                "mode": "stream",
                "synthetic": synthetic,
                "source": source,
                "source_kind": source_kind,
            },
            f,
        )
    print(f"vocab={tok.vocab_size} train={len(train_ids):,} val={len(val_ids):,} -> {domain_dir}")


if __name__ == "__main__":
    main()
