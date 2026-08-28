"""Prepare tiny Shakespeare: 1MB download, character-level stream."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from ...core.tokenizer import CharTokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    domain_dir = Path(args.out_dir) if args.out_dir else Path(args.data_root) / "text"
    domain_dir.mkdir(parents=True, exist_ok=True)
    raw = domain_dir / "input.txt"
    if not raw.exists():
        print("downloading tiny shakespeare...")
        urllib.request.urlretrieve(URL, raw)
    text = raw.read_text(encoding="utf-8")
    print(f"loaded {len(text):,} chars")

    tok = CharTokenizer(set(text))
    tok.save(domain_dir / "tokenizer.json")
    n = len(text)
    train_ids = np.array(tok.encode(text[: int(n * 0.9)]), dtype=np.uint16)
    val_ids = np.array(tok.encode(text[int(n * 0.9) :]), dtype=np.uint16)
    train_ids.tofile(domain_dir / "train.bin")
    val_ids.tofile(domain_dir / "val.bin")
    with open(domain_dir / "meta.json", "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "mode": "stream"}, f)
    print(f"vocab={tok.vocab_size} train={len(train_ids):,} val={len(val_ids):,} -> {domain_dir}")


if __name__ == "__main__":
    main()
