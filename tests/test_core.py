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


def test_mask_tokens():
    from nanoscigpt.core.objectives import mask_tokens

    x = torch.tensor([[1, 2, 3, 4, 5]])
    x_m, y = mask_tokens(x, vocab_size=6, mask_prob=1.0)  # mask everything
    assert (y == x).all()  # all positions are targets
    assert (x_m == 6).all()  # all replaced by mask token id


def test_route_decision():
    from nanoscigpt.tasks.route_decision import decide

    r = decide({"data_scale": False, "task_sharing": True, "transfer_evidence": True, "multi_task_gain": True, "budget": True})
    assert r["route"] == "use_specialized_model"
    r2 = decide({"data_scale": True, "task_sharing": True, "transfer_evidence": True, "multi_task_gain": True, "budget": True})
    assert r2["route"] == "train_new_foundation"


def test_bidirectional_attention():
    from nanoscigpt.core.gpt import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=10, block_size=16, n_layer=1, n_head=1, n_embd=16, causal=False)
    model = GPT(cfg)
    x = torch.randint(0, 10, (1, 8))
    # bidirectional: perturbing a later token changes earlier representations
    x2 = x.clone()
    x2[0, 4] = (x2[0, 4] + 1) % 10
    pos = torch.arange(8)
    h1 = model.transformer.wte(x) + model.transformer.wpe(pos)
    h2 = model.transformer.wte(x2) + model.transformer.wpe(pos)
    b1 = model.transformer.h[0](h1)
    b2 = model.transformer.h[0](h2)
    assert not torch.allclose(b1[0, 0], b2[0, 0])  # position 0 sees position 4
