"""Encoders: the only model part that changes between scientific fields.

An encoder turns one object (a sentence, a protein, a spectrum, an image, a
crystal) into a sequence of vectors. The shared nanoGPT blocks in
``scigpt.py`` read that sequence. Each encoder also scores the pretraining
task used by GPT: predict the next token from the tokens before it.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SymbolEncoder(nn.Module):
    """Characters of text, protein, DNA, RNA or SMILES. Token 0 is padding."""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.to_symbol = nn.Linear(n_embd, vocab_size, bias=False)
        self.to_symbol.weight = self.embedding.weight  # tied, as in nanoGPT

    def forward(self, batch):
        tokens = batch["tokens"]
        return self.embedding(tokens), tokens == 0

    def next_token_loss(self, hidden, batch):
        logits = self.to_symbol(hidden[:, :-1])
        targets = batch["tokens"][:, 1:]
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)


class PatchEncoder(nn.Module):
    """Numeric patches of a spectrum, a field, an image or a distance matrix.

    ``context`` holds per-object conditions such as temperature. When present
    it becomes the first token, so every later token can attend to it.
    """

    def __init__(self, patch_dim, n_embd, context_dim=0):
        super().__init__()
        self.project = nn.Linear(patch_dim, n_embd)
        self.context = nn.Linear(context_dim, n_embd) if context_dim else None
        self.to_patch = nn.Linear(n_embd, patch_dim)

    def forward(self, batch):
        embedded = self.project(batch["patches"])
        if self.context is not None:
            first = self.context(batch["context"])[:, None]
            embedded = torch.cat([first, embedded], dim=1)
        return embedded, None

    def next_token_loss(self, hidden, batch):
        patches = batch["patches"]
        targets = patches if self.context is not None else patches[:, 1:]
        return F.mse_loss(self.to_patch(hidden[:, :-1]), targets)


class CrystalEncoder(nn.Module):
    """A unit cell: the lattice is the first token, then one token per atom.

    Element 0 marks an empty atom slot.
    """

    def __init__(self, n_embd, n_elements=119):
        super().__init__()
        self.element = nn.Embedding(n_elements, n_embd, padding_idx=0)
        self.position = nn.Linear(3, n_embd)
        self.lattice = nn.Linear(9, n_embd)
        self.to_element = nn.Linear(n_embd, n_elements)

    def forward(self, batch):
        elements = batch["elements"]
        atoms = self.element(elements) + self.position(batch["fractional"])
        cell = self.lattice(batch["lattice"].flatten(1))[:, None]
        empty = elements == 0
        cell_is_real = empty.new_zeros(len(empty), 1)
        return torch.cat([cell, atoms], dim=1), torch.cat([cell_is_real, empty], dim=1)

    def next_token_loss(self, hidden, batch):
        logits = self.to_element(hidden[:, :-1])
        targets = batch["elements"]
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)
