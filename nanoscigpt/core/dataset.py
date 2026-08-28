"""Dataset layer: concatenated-stream and independent-sequence modes."""

import json
from pathlib import Path

import numpy as np
import torch


class TokenStreamDataset:
    """nanoGPT-style: all data concatenated into one stream, random windows."""

    def __init__(self, data_dir, split):
        self.data = np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint16, mode="r")
        with open(Path(data_dir) / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.vocab_size = self.meta["vocab_size"]

    def get_batch(self, batch_size, block_size, device):
        ix = np.random.randint(0, len(self.data) - block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(self.data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device), None


class IndependentSequenceDataset:
    """Protein/SMILES style: each sample is a separate sequence, padded."""

    def __init__(self, data_dir, split):
        self.sequences = np.load(Path(data_dir) / f"{split}_seqs.npy", allow_pickle=True)
        with open(Path(data_dir) / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.vocab_size = self.meta["vocab_size"]
        self.pad_id = self.meta["pad_id"]

    def get_batch(self, batch_size, block_size, device):
        idxs = np.random.randint(0, len(self.sequences), size=batch_size)
        batch = [self.sequences[i] for i in idxs]
        T = min(block_size, max(len(s) for s in batch))
        x = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
        y = torch.full((len(batch), T), -1, dtype=torch.long)
        pad_mask = torch.ones((len(batch), T), dtype=torch.bool)
        for j, seq in enumerate(batch):
            n = min(len(seq), T)
            x[j, :n] = torch.from_numpy(seq[:n].astype(np.int64))
            y[j, : n - 1] = torch.from_numpy(seq[1:n].astype(np.int64))
            pad_mask[j, :n] = False
        return x.to(device), y.to(device), pad_mask.to(device)
