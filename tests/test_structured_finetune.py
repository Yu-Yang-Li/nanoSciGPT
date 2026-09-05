import hashlib
from pathlib import Path

import pytest
import torch

from nanoscigpt.tasks.downstream_demo import run_downstream
from nanoscigpt.tasks.structured_demo import STRUCTURED_DOMAINS, run_structured


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("domain", STRUCTURED_DOMAINS)
def test_pretrained_structured_encoder_can_be_finetuned_and_reloaded(tmp_path, domain):
    torch.set_num_threads(1)
    run_structured(domain, ROOT / "data", tmp_path / "pretrain", pretrain_steps=1, task_steps=1)
    checkpoint = tmp_path / "pretrain/model/ckpt.pt"
    before = torch.load(checkpoint, weights_only=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = run_downstream(domain, checkpoint, ROOT / "data", tmp_path / "finetune",
                            adaptation="finetune", epochs=1, max_samples=16)
    after = torch.load(result["task_checkpoint"], weights_only=False)
    assert result["encoder_frozen"] is False
    assert result["pretrained_parameters_updated"] is True
    assert result["encoder_delta_l2"] > 0
    assert any(not torch.equal(value, after["model"][key]) for key, value in before["model"].items())
    assert "target_mean" in after and "target_scale" in after
    assert after["domain"] == domain
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    continued = run_downstream(domain, result["task_checkpoint"], ROOT / "data", tmp_path / "continued",
                               adaptation="finetune", epochs=1, max_samples=16)
    assert continued["metric_before_finetune"] == pytest.approx(result["metric_value"], abs=1e-6)
    with pytest.raises(ValueError, match="sampling"):
        run_downstream(domain, result["task_checkpoint"], ROOT / "data", tmp_path / "other_subset",
                       adaptation="finetune", epochs=1, max_samples=8)
