"""nanoSciGPT: the nanoGPT blocks stay the same, only the encoder changes."""

import torch
from torch import nn

from ..core.gpt import GPT, GPTConfig


class SciGPT(nn.Module):
    """encoder -> position embedding -> nanoGPT blocks -> one predicted number."""

    def __init__(self, encoder, block_size, n_layer=2, n_head=2, n_embd=32):
        super().__init__()
        self.encoder = encoder
        # The position embedding, attention blocks and final LayerNorm are taken
        # unchanged from core/gpt.py. Its word embedding is replaced by `encoder`.
        gpt = GPT(GPTConfig(1, block_size, n_layer, n_head, n_embd))
        self.wpe = gpt.transformer.wpe
        self.blocks = gpt.transformer.h
        self.ln_f = gpt.transformer.ln_f
        self.encoder.apply(gpt._init_weights)  # same small initial weights as nanoGPT
        self.head = nn.Linear(n_embd, 1)

    def hidden(self, batch):
        embedded, padding = self.encoder(batch)
        positions = torch.arange(embedded.size(1))
        x = embedded + self.wpe(positions)
        for block in self.blocks:
            x = block(x, padding)
        return self.ln_f(x), padding

    def pooled(self, hidden, padding):
        if padding is None:
            return hidden.mean(dim=1)
        real = (~padding).unsqueeze(-1).float()
        return (hidden * real).sum(dim=1) / real.sum(dim=1)

    def forward(self, batch):
        hidden, padding = self.hidden(batch)
        return self.head(self.pooled(hidden, padding)).flatten()
