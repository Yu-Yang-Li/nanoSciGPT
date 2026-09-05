import hashlib
from pathlib import Path

import pytest
import torch

from nanoscigpt.core.gpt import GPT, GPTConfig
from nanoscigpt.core.tokenizer import CharTokenizer
from nanoscigpt.tasks.downstream_demo import run_downstream


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("domain", ["text", "protein", "dna", "smiles"])
def test_finetune_updates_encoder_and_saves_model_without_replacing_input(tmp_path, domain):
    torch.set_num_threads(1)
    tokenizer = CharTokenizer.load(ROOT / "data" / domain / "tokenizer.json")
    config = GPTConfig(vocab_size=len(tokenizer.stoi), block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = GPT(config)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({"domain": domain, "model_args": vars(config), "model": original}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = run_downstream(domain, checkpoint, ROOT / "data", tmp_path / "task", epochs=1,
                            max_samples=16, adaptation="finetune")
    assert result["encoder_frozen"] is False
    assert result["pretrained_parameters_updated"] is True
    updated = torch.load(result["task_checkpoint"], weights_only=False)
    assert any(not torch.equal(original[key], value) for key, value in updated["model"].items())
    assert updated["head"]
    assert updated["domain"] == domain
    assert updated["task_sampling"]["max_samples"] == 16
    assert result["encoder_delta_l2"] > 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    assert result["metric_name"] == ("mae" if domain == "smiles" else "accuracy")
    resumed = run_downstream(domain, result["task_checkpoint"], ROOT / "data", tmp_path / "continued",
                             epochs=1, max_samples=16, adaptation="finetune")
    assert resumed["metric_before_finetune"] == result["metric_value"]
    with pytest.raises(ValueError, match="sampling"):
        run_downstream(domain, result["task_checkpoint"], ROOT / "data", tmp_path / "changed_subset",
                       epochs=1, max_samples=8, adaptation="finetune")


def test_finetune_refuses_to_replace_an_existing_result(tmp_path):
    output = tmp_path / "task"
    output.mkdir()
    checkpoint = output / "finetuned.pt"
    checkpoint.write_bytes(b"previous student experiment")
    with pytest.raises(FileExistsError):
        run_downstream("text", tmp_path / "not_loaded.pt", ROOT / "data", output,
                       epochs=1, adaptation="finetune")
    assert checkpoint.read_bytes() == b"previous student experiment"


def test_teaching_cli_limits_cpu_threads(monkeypatch):
    import sys
    from nanoscigpt.tasks import downstream_demo

    previous = torch.get_num_threads()
    observed = []
    try:
        torch.set_num_threads(2)
        monkeypatch.setattr(sys, "argv", ["downstream_demo", "--domain", "protein", "--ckpt", "unused.pt"])
        monkeypatch.setattr(downstream_demo, "run_downstream", lambda *a, **kw: observed.append(torch.get_num_threads()))
        downstream_demo.main()
        assert observed == [1]
    finally:
        torch.set_num_threads(previous)
