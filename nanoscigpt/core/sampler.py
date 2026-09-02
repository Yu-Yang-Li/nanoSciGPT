"""Sample from a trained checkpoint."""

import argparse
from pathlib import Path

import torch

from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer


def trim_at_token(token_ids, stop_token_id):
    if stop_token_id is None or stop_token_id not in token_ids:
        return token_ids
    return token_ids[: token_ids.index(stop_token_id)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["text", "protein", "dna", "smiles"])
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tok = CharTokenizer.load(Path(args.data_root) / args.domain / "tokenizer.json")
    out_dir = Path(args.out_dir) if args.out_dir else Path("out") / args.domain
    ckpt = torch.load(out_dir / "ckpt.pt", map_location=args.device)
    cfg = GPTConfig(**{k: ckpt["model_args"][k] for k in ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]})
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    defaults = {"text": "\n", "protein": "M", "dna": "A", "smiles": "C"}
    start = args.start or defaults[args.domain]
    x = torch.tensor([tok.encode(start)], dtype=torch.long, device=args.device)
    stop_token_id = tok.stoi.get("<eos>")
    forbidden_token_ids = [tok.stoi["<pad>"]] if "<pad>" in tok.stoi else None
    for i in range(args.num_samples):
        y = model.generate(
            x,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_token_id=stop_token_id,
            forbidden_token_ids=forbidden_token_ids,
        )
        print(f"--- sample {i+1} ---")
        print(tok.decode(trim_at_token(y[0].tolist(), stop_token_id)))


if __name__ == "__main__":
    main()
