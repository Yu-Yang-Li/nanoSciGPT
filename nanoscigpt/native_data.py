"""Data/checkpoint adapter copied beside the original AI Scientist v1 template."""

from pathlib import Path

import numpy as np
import torch


def get_batch(split, data_dir, batch_size, block_size, device):
    data_dir = Path(data_dir)
    sequence_file = data_dir / f"{split}_seqs.npy"
    if sequence_file.exists():
        sequences = np.load(sequence_file, allow_pickle=True)
        x = torch.zeros((batch_size, block_size), dtype=torch.long)
        y = torch.full_like(x, -1)
        for row, index in enumerate(torch.randint(len(sequences), (batch_size,)).tolist()):
            sequence = sequences[index]
            if len(sequence) < 2:
                raise ValueError("an independent sequence must contain at least two tokens")
            start = int(torch.randint(max(1, len(sequence) - block_size), ()).item())
            tokens = torch.as_tensor(np.asarray(sequence[start:start + block_size + 1], dtype=np.int64))
            length = len(tokens) - 1
            x[row, :length] = tokens[:-1]
            y[row, :length] = tokens[1:]
    else:
        stream = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        starts = torch.randint(len(stream) - block_size, (batch_size,)).tolist()
        x = torch.stack([torch.as_tensor(stream[i:i + block_size].astype(np.int64)) for i in starts])
        y = torch.stack([torch.as_tensor(stream[i + 1:i + block_size + 1].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


def load_initial_model(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint["model_args"].get("causal", True):
        raise ValueError("the original v1 GPT template requires a causal checkpoint")
    keys = model.state_dict().keys()
    state = {key: value for key, value in checkpoint["model"].items()
             if not (key.endswith(".attn.bias") and key not in keys)}
    model.load_state_dict(state, strict=True)
