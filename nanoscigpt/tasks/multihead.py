"""A3: one shared encoder, multiple task heads (classification + regression).

Teaching point: "基座能力统一" - the encoder is shared, task-specific
behavior comes only from lightweight heads. This is the minimal structural
metaphor for a foundation model: one representation, many consumers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..core.gpt import GPT, GPTConfig


class MultiTaskModel(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_layer=2, n_head=2, block_size=64, n_classes=2):
        super().__init__()
        self.encoder = GPT(GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd, causal=False))
        self.cls_head = nn.Linear(n_embd, n_classes)
        self.reg_head = nn.Linear(n_embd, 1)

    def forward(self, x, task="cls"):
        # take pre-lm_head representation: run blocks manually
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        h = self.encoder.transformer.wte(x) + self.encoder.transformer.wpe(pos)
        for block in self.encoder.transformer.h:
            h = block(h)
        h = self.encoder.transformer.ln_f(h)
        pooled = h.mean(dim=1)  # (B, n_embd)
        if task == "cls":
            return self.cls_head(pooled)
        return self.reg_head(pooled).squeeze(-1)


def make_synthetic_tasks(n=200, seq_len=40, seed=0):
    rng = np.random.default_rng(seed)
    vocab = 8
    X = torch.from_numpy(rng.integers(1, vocab, size=(n, seq_len))).long()
    # task A (classification): does token id 5 appear in the first 5 positions
    y_cls = (X[:, :5] == 5).any(dim=1).long()
    # task B (regression): mean token value
    y_reg = X.float().mean(dim=1)
    return X, y_cls, y_reg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out/multihead")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    X, y_cls, y_reg = make_synthetic_tasks()
    n_train = int(len(X) * 0.8)
    model = MultiTaskModel(vocab_size=10)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    cls_lossf = nn.CrossEntropyLoss()
    reg_lossf = nn.MSELoss()

    for ep in range(args.epochs):
        opt.zero_grad()
        loss = cls_lossf(model(X[:n_train], "cls"), y_cls[:n_train])
        loss = loss + reg_lossf(model(X[:n_train], "reg"), y_reg[:n_train])
        loss.backward()
        opt.step()

    with torch.no_grad():
        cls_acc = (model(X[n_train:], "cls").argmax(1) == y_cls[n_train:]).float().mean().item()
        reg_mae = (model(X[n_train:], "reg") - y_reg[n_train:]).abs().mean().item()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {"cls_acc": round(cls_acc, 3), "reg_mae": round(reg_mae, 4), "epochs": args.epochs}
    with open(out / "multihead_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"shared encoder multi-task: cls_acc={cls_acc:.3f} reg_mae={reg_mae:.4f} -> {out/'multihead_results.json'}")


if __name__ == "__main__":
    main()
