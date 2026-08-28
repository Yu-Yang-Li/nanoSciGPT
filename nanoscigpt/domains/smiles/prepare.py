"""Prepare SMILES strings from DeepChem ESOL (teaching fixture)."""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import numpy as np

from ...core.tokenizer import CharTokenizer

URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    domain_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / "smiles"
    domain_dir.mkdir(parents=True, exist_ok=True)
    raw = domain_dir / "delaney-processed.csv"
    if not raw.exists():
        print("downloading DeepChem ESOL...")
        urllib.request.urlretrieve(URL, raw)

    smiles = []
    with open(raw, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row.get("smiles", "").strip()
            if s:
                smiles.append(s)
    print(f"parsed {len(smiles)} SMILES")

    tok = CharTokenizer(set("".join(smiles)) | {"<pad>", "<eos>"})
    tok.save(domain_dir / "tokenizer.json")
    pad_id = tok.stoi["<pad>"]

    def encode_seqs(lst):
        return [np.array(tok.encode(s) + [tok.stoi["<eos>"]], dtype=np.uint16) for s in lst]

    n = len(smiles)
    split = int(n * 0.9)
    train_arr = np.empty(split, dtype=object)
    val_arr = np.empty(n - split, dtype=object)
    train_arr[:] = encode_seqs(smiles[:split])
    val_arr[:] = encode_seqs(smiles[split:])
    np.save(domain_dir / "train_seqs.npy", train_arr, allow_pickle=True)
    np.save(domain_dir / "val_seqs.npy", val_arr, allow_pickle=True)
    with open(domain_dir / "meta.json", "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "mode": "independent", "pad_id": pad_id}, f)
    print(f"vocab={tok.vocab_size} train={split} val={n - split} -> {domain_dir}")


if __name__ == "__main__":
    main()
