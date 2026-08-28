"""A2: frozen ESM-2 representations + linear probe classification.

Teaching point: "别人预训练好的表征可以直接取用" - we never touch ESM
weights; we only train a tiny logistic head on top of frozen per-residue
embeddings, then compare against a one-hot baseline to show what transfer
actually buys.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_localization_dataset(out_dir, size=300):
    """Synthetic teaching fixture: sequences with a hydrophobic N-terminal
    signal vs. soluble sequences. Labels come from a simple rule we control,
    so probe accuracy is interpretable rather than a scientific claim."""
    rng = np.random.default_rng(42)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    hydro = "AILMFWV"
    seqs, labels = [], []
    for i in range(size):
        n = rng.integers(40, 90)
        if i % 2 == 0:
            head = "".join(rng.choice(list(hydro), 18))
            body = "".join(rng.choice(list(aas), max(0, n - 19)))
            labels.append(1)  # signal-peptide-like
        else:
            body = "".join(rng.choice(list(aas), n))
            labels.append(0)  # soluble-like
        seqs.append("M" + head + body if i % 2 == 0 else "M" + body)
    return seqs, labels


def embed_esm(seqs, batch_size=8, weights_dir=None):
    import esm

    if weights_dir and (Path(weights_dir) / "esm2_t6_8M_UR50D.pt").exists():
        # offline path: load from local weights/ instead of downloading
        # weights_only=False required for fair-esm's legacy checkpoint format (torch>=2.6 default changed)
        model_data = torch.load(Path(weights_dir) / "esm2_t6_8M_UR50D.pt", map_location="cpu", weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core("esm2_t6_8M_UR50D", model_data)
        model.eval()
    else:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    converter = alphabet.get_batch_converter()
    reps = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [(f"s{j}", seqs[j]) for j in range(i, min(i + batch_size, len(seqs)))]
            _, _, tokens = converter(batch)
            out = model(tokens, repr_layers=[6], return_contacts=False)
            r = out["representations"][6]  # (B, T, 320)
            # mean-pool over residues (excluding BOS/EOS/pad)
            reps.append(r.mean(dim=1).cpu().numpy())
    return np.concatenate(reps, axis=0)  # (N, 320)


def onehot_embed(seqs, max_len=90):
    aas = "ACDEFGHIKLMNPQRSTVWY"
    stoi = {c: i for i, c in enumerate(aas)}
    out = np.zeros((len(seqs), max_len * len(aas)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:max_len]):
            if c in stoi:
                out[i, j * len(aas) + stoi[c]] = 1.0
    return out


def train_probe(X, y, epochs=200, lr=0.01, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(np.array(y)).long()
    n_train = int(len(X) * 0.8)
    idx = torch.randperm(len(X))
    tr, va = idx[:n_train], idx[n_train:]
    head = nn.Linear(X.shape[1], 2)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(X[tr]), y[tr])
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(X[va]).argmax(1) == y[va]).float().mean().item()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out/esm_probe")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--weights_dir", default="weights", help="local ESM weights dir; falls back to online download")
    args = p.parse_args()

    seqs, labels = load_localization_dataset(args.out_dir, args.size)
    print(f"dataset: {len(seqs)} sequences, {sum(labels)} positive")

    X_esm = embed_esm(seqs, weights_dir=args.weights_dir)
    acc_esm = train_probe(X_esm, labels)
    print(f"ESM frozen + linear probe val acc: {acc_esm:.3f}")

    X_oh = onehot_embed(seqs)
    acc_oh = train_probe(X_oh, labels)
    print(f"one-hot + linear probe val acc:     {acc_oh:.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe_results.json", "w") as f:
        json.dump({"esm_acc": acc_esm, "onehot_acc": acc_oh, "n": len(seqs)}, f, indent=2)
    print(f"delta = {acc_esm - acc_oh:+.3f}  -> {out/'probe_results.json'}")


if __name__ == "__main__":
    main()
