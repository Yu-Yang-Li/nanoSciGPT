"""A2: train the same encoder with CLM vs MLM on the same protein data.

Teaching point: "换预训练目标" - architecture stays identical, only the
objective and attention mask change. Students see both losses decrease but
at different rates and with different downstream behavior.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..core.dataset import IndependentSequenceDataset
from ..core.gpt import GPT, GPTConfig
from ..core.objectives import mask_tokens


def train(model, ds, objective, iters, batch_size, block_size, lr=1e-3, mask_prob=0.15, seed=1337):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    losses = []
    vocab = ds.vocab_size
    for it in range(iters):
        x, y, pad = ds.get_batch(batch_size, block_size, "cpu")
        if objective == "clm":
            _, loss = model(x, y, pad)
        else:
            x_m, y_m = mask_tokens(x, vocab, mask_prob)
            _, loss = model(x_m, y_m, pad)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--domain", default="protein")
    p.add_argument("--out_dir", default="out/objective_contrast")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=64)
    args = p.parse_args()

    ds = IndependentSequenceDataset(Path(args.data_root) / args.domain, "train")
    results = {}
    for objective in ["clm", "mlm"]:
        causal = objective == "clm"
        cfg = GPTConfig(vocab_size=ds.vocab_size + 1, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, causal=causal)
        model = GPT(cfg)
        losses = train(model, ds, objective, args.iters, args.batch_size, args.block_size)
        results[objective] = {
            "first_10": round(float(np.mean(losses[:10])), 4),
            "last_10": round(float(np.mean(losses[-10:])), 4),
            "params_M": round(model.num_params() / 1e6, 2),
        }
        print(f"{objective.upper()}: loss {results[objective]['first_10']} -> {results[objective]['last_10']} ({results[objective]['params_M']}M params)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "contrast_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {out/'contrast_results.json'}")


if __name__ == "__main__":
    main()
