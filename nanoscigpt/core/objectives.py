"""Pretraining objectives: causal LM vs masked LM (A2 teaching contrast)."""

import torch
import torch.nn.functional as F


def causal_loss(model, x, y, pad_mask=None):
    _, loss = model(x, y, pad_mask)
    return loss


def mask_tokens(x, vocab_size, mask_prob=0.15, mask_token_id=None, special_ids=()):
    """Mask 15% of positions. Returns (masked_x, targets); targets==-1 ignored."""
    if mask_token_id is None:
        mask_token_id = vocab_size
    labels = x.clone()
    prob = torch.full(x.shape, mask_prob)
    for sid in special_ids:
        prob[x == sid] = 0.0
    mask = torch.bernoulli(prob).bool()
    labels[~mask] = -1
    x_masked = x.clone()
    x_masked[mask] = mask_token_id
    return x_masked, labels


def masked_loss(model, x, y, pad_mask=None):
    logits, _ = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)
