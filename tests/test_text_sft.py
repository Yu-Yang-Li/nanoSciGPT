import hashlib
from pathlib import Path

import torch

from nanoscigpt.core.gpt import GPT, GPTConfig
from nanoscigpt.core.tokenizer import CharTokenizer
from nanoscigpt.tasks import text_sft
from nanoscigpt.tasks.downstream_demo import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_loading_a_masked_encoder_does_not_silently_make_it_causal(tmp_path):
    config = GPTConfig(65, block_size=64, n_layer=1, n_head=1, n_embd=16, causal=False)
    model = GPT(config)
    path = tmp_path / "masked.pt"
    torch.save({"model": model.state_dict(), "model_args": vars(config), "domain": "text"}, path)
    loaded, _, _ = load_checkpoint(path)
    assert loaded.config.causal is False


def test_sft_ignores_question_targets_and_trains_the_answer():
    tok = CharTokenizer.load(ROOT / "data/text/tokenizer.json")
    x, y = text_sft.encode_pair("What is DNA?", "A sequence.", tok, 64)
    prefix = "Q:What is DNA?\nA:"
    assert y[:len(prefix) - 1].eq(-1).all()
    assert y[len(prefix) - 1].item() == tok.stoi["A"]
    assert y[y != -1].tolist() == tok.encode("A sequence.\n")
    assert len(x) == len(y)


def test_text_sft_uses_pretrained_weights_and_preserves_source(tmp_path):
    torch.set_num_threads(1)
    config = GPTConfig(65, block_size=64, n_layer=1, n_head=1, n_embd=16)
    model = GPT(config)
    checkpoint = tmp_path / "input.pt"
    torch.save({"model": model.state_dict(), "model_args": vars(config), "domain": "text"}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = text_sft.run(checkpoint, ROOT / "data", tmp_path / "sft", steps=2)
    assert result["pretrained_parameters_updated"]
    assert result["encoder_delta_l2"] > 0
    assert result["loss_scope"] == "answer_tokens_only"
    assert len(result["samples"]) > 0
    assert result["samples"][0]["before"] is not None
    assert result["samples"][0]["after"] is not None
    assert len(result["training_samples"]) == len(text_sft.TRAIN_PAIRS)
    assert result["training_samples"][0]["question"] == text_sft.TRAIN_PAIRS[0][0]
    assert result["training_samples"][0]["after"] is not None
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
