import numpy as np
import torch

from nanoscigpt.core.gpt import GPT, GPTConfig
from nanoscigpt.core.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    tok = CharTokenizer(set("ACGT"))
    s = "ACGTACGT"
    ids = tok.encode(s)
    assert tok.decode(ids) == s


def test_gpt_forward_stream():
    cfg = GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 20, (2, 16))
    y = torch.randint(0, 20, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 20)
    assert loss.item() > 0


def test_gpt_forward_padding():
    cfg = GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 20, (2, 16))
    y = torch.randint(0, 20, (2, 16))
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[0, 8:] = True
    logits, loss = model(x, y, pad)
    assert loss.item() > 0


def test_generate():
    cfg = GPTConfig(vocab_size=10, block_size=32, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg)
    x = torch.randint(0, 10, (1, 4))
    out = model.generate(x, 8)
    assert out.shape == (1, 12)
