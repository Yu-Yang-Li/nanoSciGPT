"""Prepare protein sequences from UniProt reviewed entries (teaching fixture)."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from ...core.tokenizer import CharTokenizer

URL = "https://rest.uniprot.org/uniprotkb/search?query=reviewed:true&format=fasta&size={size}"
AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_fasta_sequences(path, max_len=0):
    sequences = []
    rejected = 0
    current = []

    def keep_current():
        nonlocal rejected
        if not current:
            return
        sequence = "".join(current).upper()
        if not sequence or set(sequence) - set(AA):
            rejected += 1
            return
        sequences.append(sequence[:max_len] if max_len else sequence)

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            keep_current()
            current = []
        else:
            current.append(line.strip())
    keep_current()
    return sequences, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fasta", type=Path, help="local protein FASTA; avoids network access")
    p.add_argument("--size", type=int, default=500)
    p.add_argument(
        "--max_len",
        type=int,
        default=0,
        help="optional preprocessing truncation; 0 keeps full sequences and trainer samples windows",
    )
    args = p.parse_args()

    domain_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / "protein"
    domain_dir.mkdir(parents=True, exist_ok=True)
    if args.fasta:
        raw = args.fasta.resolve()
        if not raw.is_file():
            raise SystemExit(f"FASTA not found: {raw}")
        source = str(raw)
        source_kind = "user_file"
    else:
        raw = domain_dir / "uniprot.fasta"
        if not raw.exists():
            print(f"downloading {args.size} reviewed UniProt entries...")
            urllib.request.urlretrieve(URL.format(size=args.size), raw)
        source = URL.format(size=args.size)
        source_kind = "public_source"

    seqs, rejected = parse_fasta_sequences(raw, max_len=args.max_len)
    print(
        f"parsed {len(seqs)} canonical protein sequences "
        f"(rejected={rejected}, max_len={args.max_len or 'full'})"
    )

    tok = CharTokenizer(set(AA) | {"<pad>", "<eos>"})
    tok.save(domain_dir / "tokenizer.json")
    pad_id = tok.stoi["<pad>"]

    def encode_seqs(lst):
        return [np.array(tok.encode(s) + [tok.stoi["<eos>"]], dtype=np.uint16) for s in lst]

    n = len(seqs)
    if n < 2:
        raise SystemExit("protein FASTA must contain at least two canonical sequences")
    split = int(n * 0.9)
    train_arr = np.empty(split, dtype=object)
    val_arr = np.empty(n - split, dtype=object)
    train_arr[:] = encode_seqs(seqs[:split])
    val_arr[:] = encode_seqs(seqs[split:])
    np.save(domain_dir / "train_seqs.npy", train_arr, allow_pickle=True)
    np.save(domain_dir / "val_seqs.npy", val_arr, allow_pickle=True)
    with open(domain_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": tok.vocab_size,
                "mode": "independent",
                "pad_id": pad_id,
                "source": source,
                "source_kind": source_kind,
                "raw_sequences": len(seqs) + rejected,
                "accepted_sequences": len(seqs),
                "rejected_noncanonical": rejected,
                "split": "source_order_90_10_teaching_only",
                "long_sequence_handling": "random_window_at_training_time",
            },
            f,
        )
    print(f"vocab={tok.vocab_size} train={split} val={n - split} -> {domain_dir}")


if __name__ == "__main__":
    main()
