"""A2: does OUR pretrained encoder transfer? Three-way comparison.

The nanoGPT-series teaching point: train your own protein model in A1,
then ask whether its representations actually transfer. We compare:

1. one-hot baseline (no learning in the encoder)
2. random-init encoder + probe (architecture alone)
3. our pretrained encoder + probe (A1's checkpoint)

Expected honest outcome on ~450 sequences: gains are small or absent.
That IS the lesson - real foundation models need orders of magnitude more
data than a teaching fixture provides.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..core.dataset import IndependentSequenceDataset
from ..core.gpt import GPT, GPTConfig
from ..domains.protein.prepare import AA
from ..core.tokenizer import CharTokenizer


def make_probe_task(n=300, seed=42):
    """Synthetic localization-style task: hydrophobic N-terminal signal."""
    rng = np.random.default_rng(seed)
    aas = list(AA)
    hydro = "AILMFWV"
    seqs, labels = [], []
    for i in range(n):
        length = int(rng.integers(40, 90))
        if i % 2 == 0:
            head = "".join(rng.choice(list(hydro), 18))
            body = "".join(rng.choice(aas, max(0, length - 19)))
            labels.append(1)
        else:
            body = "".join(rng.choice(aas, length))
            labels.append(0)
        seqs.append("M" + head + body if i % 2 == 0 else "M" + body)
    return seqs, labels


def encode_batch(seqs, tok, block_size=90):
    """Pad to fixed length, return tensor + pad_mask."""
    x = torch.full((len(seqs), block_size), tok.stoi.get("<pad>", 0), dtype=torch.long)
    pad = torch.ones((len(seqs), block_size), dtype=torch.bool)
    for j, s in enumerate(seqs):
        ids = tok.encode(s)[:block_size]
        x[j, : len(ids)] = torch.tensor(ids)
        pad[j, : len(ids)] = False
    return x, pad


def extract_features(model, x, pad, mode="pretrained"):
    """Mean-pool the final hidden state (pre-lm_head) as sequence representation."""
    with torch.no_grad():
        pos = torch.arange(0, x.size(1))
        h = model.transformer.wte(x) + model.transformer.wpe(pos)
        for block in model.transformer.h:
            h = block(h, pad)
        h = model.transformer.ln_f(h)
        # mean over non-pad positions
        m = (~pad).float().unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def onehot_features(seqs, tok, max_len=90):
    V = tok.vocab_size
    out = np.zeros((len(seqs), max_len * V), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:max_len]):
            if c in tok.stoi:
                out[i, j * V + tok.stoi[c]] = 1.0
    return out


def train_probe(X, y, epochs=200, lr=0.01):
    torch.manual_seed(0)
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.tensor(y)
    n_tr = int(len(X) * 0.8)
    idx = torch.randperm(len(X))
    tr, va = idx[:n_tr], idx[n_tr:]
    head = nn.Linear(X.shape[1], 2)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(X[tr]), y[tr]).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(X[va]).argmax(1) == y[va]).float().mean().item()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/protein/ckpt.pt")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="out/transfer_probe")
    p.add_argument("--size", type=int, default=300)
    args = p.parse_args()

    seqs, labels = make_probe_task(args.size)
    tok = CharTokenizer(set(AA) | {"<pad>", "<eos>"})
    # remap: data/protein's tokenizer used same alphabet; load from meta for safety
    tok = CharTokenizer.load(Path(args.data_root) / "protein" / "tokenizer.json")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**{k: ckpt["model_args"][k] for k in ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]})
    # probe sequences must fit the trained model's context window
    x, pad = encode_batch(seqs, tok, block_size=cfg.block_size)

    results = {}

    # 1. one-hot
    X_oh = onehot_features(seqs, tok, max_len=cfg.block_size)
    results["onehot"] = round(train_probe(X_oh, labels), 3)

    # 2. random-init encoder
    torch.manual_seed(999)
    rand_model = GPT(cfg)
    X_rand = extract_features(rand_model, x, pad).numpy()
    results["random_encoder"] = round(train_probe(X_rand, labels), 3)

    # 3. our pretrained encoder
    if not Path(args.ckpt).exists():
        print(f"checkpoint {args.ckpt} not found; run trainer --domain protein first")
        return
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    X_pre = extract_features(model, x, pad).numpy()
    results["pretrained_encoder"] = round(train_probe(X_pre, labels), 3)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"one-hot:            {results['onehot']:.3f}")
    print(f"random encoder:     {results['random_encoder']:.3f}")
    print(f"pretrained encoder: {results['pretrained_encoder']:.3f}")
    delta = results["pretrained_encoder"] - results["onehot"]
    print(f"transfer delta: {delta:+.3f} -> {out/'probe_results.json'}")
    if delta < 0.05:
        print("NOTE: transfer gain is small - this IS the honest lesson: 450 sequences")
        print("cannot support a foundation-model claim. Real ones need orders more data.")


if __name__ == "__main__":
    main()
