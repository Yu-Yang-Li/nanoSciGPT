"""Small CPU models that keep shared and domain-specific structure separate."""

import torch
import torch.nn as nn

from .adapters import periodic_distances


class PatchEncoder(nn.Module):
    """Transformer over numeric patches with a reconstruction head."""

    def __init__(self, input_dim, max_tokens, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.reconstruction = nn.Linear(hidden_dim, input_dim)

    def encode(self, tokens, masked=None):
        hidden = self.input_projection(tokens) + self.position[:, : tokens.size(1)]
        if masked is not None:
            hidden = torch.where(masked.unsqueeze(-1), self.mask_token, hidden)
        return self.encoder(hidden)

    def reconstruct(self, tokens, masked):
        return self.reconstruction(self.encode(tokens, masked))


class CrystalGraphEncoder(nn.Module):
    """Periodic message-passing encoder for small crystal cells."""

    def __init__(self, hidden_dim=32, radial_dim=8):
        super().__init__()
        self.atom_embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.register_buffer("radial_centers", torch.linspace(0.0, 6.0, radial_dim))
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.norm = nn.LayerNorm(hidden_dim)
        self.atom_head = nn.Linear(hidden_dim, 119)

    def encode(self, atomic_numbers, fractional, lattice, node_mask):
        hidden = self.atom_embedding(atomic_numbers)
        distances = periodic_distances(fractional, lattice)
        radial = torch.exp(-((distances.unsqueeze(-1) - self.radial_centers) ** 2))
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        identity = torch.eye(node_mask.size(1), dtype=torch.bool, device=node_mask.device)
        pair_mask = pair_mask & ~identity.unsqueeze(0)
        for _ in range(2):
            neighbour = hidden[:, None, :, :].expand(-1, hidden.size(1), -1, -1)
            messages = self.message(torch.cat([neighbour, radial], dim=-1))
            messages = messages * pair_mask.unsqueeze(-1)
            count = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
            aggregate = messages.sum(dim=2) / count
            hidden = self.norm(hidden + self.update(aggregate))
            hidden = hidden * node_mask.unsqueeze(-1)
        return hidden

    def classify_atoms(self, atomic_numbers, fractional, lattice, node_mask):
        return self.atom_head(self.encode(atomic_numbers, fractional, lattice, node_mask))


def masked_mean(hidden, mask):
    weights = mask.float().unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
