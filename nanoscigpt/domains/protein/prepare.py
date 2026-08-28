"""Prepare protein sequences from UniProt reviewed entries (teaching fixture)."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from ...core.tokenizer import CharTokenizer

URL = "https://rest.uniprot.org/uniprotkb/search?query=reviewed:true&format=fasta&size={size}"
AA = "ACDEFGHIKLMNPQRSTVWY"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--max_len", type=int, default=128)
    args = p.parse_args()

    domain_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / "protein"
    domain_dir.mkdir(parents=True, exist_ok=True)
    raw = domain_dir / "uniprot.fasta"
    if not raw.exists():
        print(f"downloading {args.size} reviewed UniProt entries...")
        urllib.request.urlretrieve(URL.format(size=args.size), raw)

    seqs = []
    current = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if current and 0 < len(current) <= args.max_len:
                seqs.append("".join(current))
            current = []
        else:
            current.append(line.strip())
    if current and 0 < len(current) <= args.max_len:
        seqs.append("".join(current))
    print(f"parsed {len(seqs)} protein sequences (max_len={args.max_len})")

    tok = CharTokenizer(set(AA) | {"<pad>", "<eos>"})
    tok.save(domain_dir / "tokenizer.json")
    pad_id = tok.stoi["<pad>"]

    def encode_seqs(lst):
        return [np.array(tok.encode(s) + [tok.stoi["<eos>"]], dtype=np.uint16) for s in lst]

    n = len(seqs)
    split = int(n * 0.9)
    train_arr = np.empty(split, dtype=object)
    val_arr = np.empty(n - split, dtype=object)
    train_arr[:] = encode_seqs(seqs[:split])
    val_arr[:] = encode_seqs(seqs[split:])
    np.save(domain_dir / "train_seqs.npy", train_arr, allow_pickle=True)
    np.save(domain_dir / "val_seqs.npy", val_arr, allow_pickle=True)
    with open(domain_dir / "meta.json", "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "mode": "independent", "pad_id": pad_id}, f)
    print(f"vocab={tok.vocab_size} train={split} val={n - split} -> {domain_dir}")


if __name__ == "__main__":
    main()
