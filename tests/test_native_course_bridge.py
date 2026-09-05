import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoscigpt import native_data
from nanoscigpt.core.gpt import GPT, GPTConfig


def test_native_independent_batch_never_joins_samples_and_ignores_padding(tmp_path):
    sequences = np.empty(2, dtype=object)
    sequences[0] = np.array([2, 2, 2, 1])
    sequences[1] = np.array([3, 3, 3, 1])
    np.save(tmp_path / "train_seqs.npy", sequences)
    x, y = native_data.get_batch("train", tmp_path, 16, 8, "cpu")
    for inputs, targets in zip(x, y):
        assert not (2 in inputs and 3 in inputs)
        assert targets[:3].tolist() in ([2, 2, 1], [3, 3, 1])
        assert targets[3:].eq(-1).all()


def test_conversion_preserves_logits_in_the_original_v1_model(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / "out/upstream/v1/templates/nanoGPT/experiment.py"
    if not source.exists():
        pytest.skip("prepare the pinned v1 source to run the original-model integration test")
    monkeypatch.setattr(sys, "argv", [str(source)])
    spec = importlib.util.spec_from_file_location("original_v1_model", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    config = GPTConfig(22, block_size=16, n_layer=1, n_head=2, n_embd=16)
    course = GPT(config).eval()
    path = tmp_path / "initial.pt"
    torch.save({"model": course.state_dict(), "model_args": vars(config), "domain": "protein"}, path)
    original = module.GPT(module.GPTConfig(vocab_size=22, block_size=16, n_layer=1, n_head=2, n_embd=16,
                                          bias=True, dropout=0.0)).eval()
    native_data.load_initial_model(original, path)
    x = torch.randint(1, 22, (2, 16))
    with torch.no_grad():
        expected = course(x)[0][:, -1]
        actual = original(x)[0][:, -1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_native_bridge_runs_original_training_on_the_course_checkpoint(tmp_path):
    import json
    import os
    import subprocess
    from nanoscigpt import native_v1

    root = Path(__file__).resolve().parents[1]
    cached = root / "out/upstream/v1"
    if not cached.exists():
        pytest.skip("prepare the pinned v1 source for this integration test")
    checkout = tmp_path / "original"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(cached), str(checkout)], check=True)
    config = GPTConfig(22, block_size=16, n_layer=1, n_head=2, n_embd=16)
    checkpoint = tmp_path / "protein.pt"
    torch.save({"model": GPT(config).state_dict(), "model_args": vars(config), "domain": "protein"}, checkpoint)
    receipt = native_v1.prepare(checkout, checkpoint, root / "data", "course_protein", steps=2)
    template = Path(receipt["template"])
    run = subprocess.run([sys.executable, "experiment.py", "--out_dir", "run_0"], cwd=template,
                         env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert "Loading the course checkpoint" in run.stdout
    result = json.loads((template / "run_0/final_info.json").read_text())
    assert "course_protein" in result
    assert "step 2:" in run.stdout and "training done" in run.stdout
    assert result["course_protein"]["means"]["total_train_time_mean"] > 0
    # Native v1 saves only when a post-initial evaluation improves. A short
    # experiment is allowed to fail to improve; do not force a fake success.
    assert (template / "run_0/ckpt.pt").exists() == ("saving checkpoint" in run.stdout)
    assert not subprocess.check_output(["git", "-C", str(checkout), "diff", "--name-only"], text=True).strip()


@pytest.mark.parametrize("domain", ["protein", "smiles"])
def test_native_task_template_keeps_supervised_predictions_and_runs_outside_course_repo(tmp_path, domain):
    import json
    import os
    import shutil
    import subprocess
    from nanoscigpt import native_v1
    from nanoscigpt.core.tokenizer import CharTokenizer
    from nanoscigpt.tasks.downstream_demo import run_downstream

    root = Path(__file__).resolve().parents[1]
    cached = root / "out/upstream/v1"
    if not cached.exists():
        pytest.skip("prepare pinned native v1 for this integration test")
    checkout = tmp_path / "original"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(cached), str(checkout)], check=True)
    torch.set_num_threads(1)
    config = GPTConfig(CharTokenizer.load(root / f"data/{domain}/tokenizer.json").vocab_size,
                       block_size=16, n_layer=1, n_head=2, n_embd=16)
    checkpoint = tmp_path / "initial.pt"
    torch.save({"domain": domain, "model_args": vars(config), "model": GPT(config).state_dict()}, checkpoint)
    task = run_downstream(domain, checkpoint, root / "data", tmp_path / "fine", epochs=1,
                          max_samples=16, adaptation="finetune")
    receipt = native_v1.prepare(checkout, task["task_checkpoint"], root / "data", "course_task", steps=2)
    assert receipt["task_type"] == task["task_type"]
    # Native v1 copies a template into a new research folder; it cannot rely on
    # imports or data paths back into nanoSciGPT at experiment execution time.
    experiment = tmp_path / "independent-research-folder"
    shutil.copytree(receipt["template"], experiment)
    process = subprocess.run([sys.executable, "experiment.py", "--out_dir", "run_0"], cwd=experiment,
                             env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    result = json.loads((experiment / "run_0/final_info.json").read_text())
    key = "val_mae" if domain == "smiles" else "val_accuracy"
    assert result["task"]["means"]["initial_" + key] == pytest.approx(task["metric_value"], abs=5.1e-5)
    assert key in result["task"]["means"]
    assert (experiment / "run_0/checkpoint.pt").exists()
    resumed = run_downstream(domain, experiment / "run_0/checkpoint.pt", root / "data", tmp_path / "round_trip",
                             epochs=1, max_samples=16, adaptation="finetune")
    assert resumed["metric_before_finetune"] == pytest.approx(result["task"]["means"][key], abs=5.1e-5)
    plotted = subprocess.run([sys.executable, "plot.py"], cwd=experiment, capture_output=True, text=True, timeout=60)
    assert plotted.returncode == 0, plotted.stderr
    assert (experiment / "task_training.png").stat().st_size > 1000
    assert not subprocess.check_output(["git", "-C", str(checkout), "diff", "--name-only"], text=True).strip()
    changed_data = tmp_path / "changed_data"
    shutil.copytree(root / "data" / domain, changed_data / domain)
    meta_path = changed_data / domain / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["source"] = "a different dataset"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="data changed"):
        native_v1.prepare(checkout, task["task_checkpoint"], changed_data, "must_not_export", steps=2)
    assert not (checkout / "templates/must_not_export").exists()
