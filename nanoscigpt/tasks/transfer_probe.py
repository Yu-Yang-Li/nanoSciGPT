"""A2: load the classroom checkpoint and run one downstream task.

This module is intentionally a process demonstration: pretrain first, then
attach a small task head and produce a saved result. It does not use the
teaching fixture to make a claim about model quality.
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


def downstream_demo_summary(score, result_path):
    """Keep the classroom output focused on completing the downstream step."""
    return [
        "downstream task: completed",
        f"result saved: {result_path}",
    ]


def downstream_demo_result(score):
    return {"status": "completed", "downstream_score": round(score, 3)}


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

    # Load the classroom checkpoint and attach one small task head.
    if not Path(args.ckpt).exists():
        print(f"checkpoint {args.ckpt} not found; run trainer --domain protein first")
        return
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    X_pre = extract_features(model, x, pad).numpy()
    score = train_probe(X_pre, labels)
    results = downstream_demo_result(score)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    for line in downstream_demo_summary(
        results["downstream_score"], out / "probe_results.json"
    ):
        print(line)


if __name__ == "__main__":
    main()
