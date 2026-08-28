"""Unified trainer for all domains."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import IndependentSequenceDataset, TokenStreamDataset
from .gpt import GPT, GPTConfig

DOMAIN_DATASET_MODES = {
    "text": "stream",
    "dna": "stream",
    "protein": "independent",
    "smiles": "independent",
}


def get_dataset(domain, data_root, split, mode):
    data_dir = Path(data_root) / domain
    return TokenStreamDataset(data_dir, split) if mode == "stream" else IndependentSequenceDataset(data_dir, split)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=list(DOMAIN_DATASET_MODES))
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mode = DOMAIN_DATASET_MODES[args.domain]
    train_ds = get_dataset(args.domain, args.data_root, "train", mode)
    val_ds = get_dataset(args.domain, args.data_root, "val", mode)
    config = GPTConfig(vocab_size=train_ds.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(config).to(args.device)
    print(f"domain={args.domain} mode={mode} vocab={config.vocab_size} params={model.num_params()/1e6:.2f}M")

    out_dir = Path(args.out_dir) if args.out_dir else Path("out") / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            model.eval()
            with torch.no_grad():
                losses = []
                for ds in (train_ds, val_ds):
                    ls = []
                    for _ in range(args.eval_iters):
                        x, y, pad = ds.get_batch(args.batch_size, args.block_size, args.device)
                        _, loss = model(x, y, pad)
                        ls.append(loss.item())
                    losses.append(sum(ls) / len(ls))
            model.train()
            print(f"iter {it}: train {losses[0]:.4f} val {losses[1]:.4f} ({time.time()-t0:.1f}s)")
            if losses[1] < best_val:
                best_val = losses[1]
                torch.save({"model": model.state_dict(), "model_args": vars(config), "iter_num": it, "best_val_loss": best_val, "domain": args.domain}, out_dir / "ckpt.pt")
        if it == args.max_iters:
            break
        x, y, pad = train_ds.get_batch(args.batch_size, args.block_size, args.device)
        _, loss = model(x, y, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with open(out_dir / "train_log.json", "w") as f:
        json.dump({"best_val_loss": best_val, "iters": args.max_iters, "domain": args.domain}, f)
    print(f"done: best val loss {best_val:.4f} -> {out_dir/'ckpt.pt'}")


if __name__ == "__main__":
    main()
