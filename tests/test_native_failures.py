"""Test teaching failure preservation against the pinned original v1 executor."""
import importlib.util
import json
from pathlib import Path

import pytest

from nanoscigpt import upstream


@pytest.fixture
def original(tmp_path, monkeypatch):
    cached = Path(__file__).resolve().parents[1] / "out/upstream/v1"
    if not (cached / ".git").exists():
        pytest.skip("prepare pinned v1 before its executor integration tests")
    target = tmp_path / "v1"
    (target / "ai_scientist").mkdir(parents=True)
    name = "ai_scientist/perform_experiments.py"
    source = upstream.git(cached, "show", f"{upstream.PROJECTS['v1'][1]}:{name}")
    (target / name).write_text(source + "\n", encoding="utf-8")
    upstream.git(target, "init")
    upstream.git(target, "add", ".")
    upstream.git(target, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    monkeypatch.setitem(upstream.PROJECTS, "v1", ("fixture", upstream.git(target, "rev-parse", "HEAD")))
    return target


def configured(target):
    adapt = getattr(upstream, "configure_v1_failures", None)
    assert callable(adapt), "v1 has no failure-preserving teaching configuration"
    adapt(target)
    source = target / "ai_scientist/perform_experiments.py"
    spec = importlib.util.spec_from_file_location("teaching_executor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment(target, ending):
    folder = target / "lesson"
    folder.mkdir(exist_ok=True)
    script = (
        "import argparse, pathlib, sys, time\n"
        "p = argparse.ArgumentParser(); p.add_argument('--out_dir'); a = p.parse_args()\n"
        "out = pathlib.Path(a.out_dir); out.mkdir(exist_ok=True)\n"
        "(out / 'partial.txt').write_text('partial measurement')\n"
        "print('deliberate diagnostic error', file=sys.stderr, flush=True)\n" + ending
    )
    (folder / "experiment.py").write_text(script, encoding="utf-8")
    return folder, script


@pytest.mark.parametrize("ending,timeout,reason", [("sys.exit(7)\n", 10, "nonzero_exit"), ("time.sleep(20)\n", 1, "timeout")])
def test_failed_attempts_keep_partial_results_and_code_on_every_retry(original, ending, timeout, reason):
    module = configured(original)
    folder, script = experiment(original, ending)
    for _ in range(2):
        code, prompt = module.run_experiment(str(folder), 1, timeout=timeout)
        assert code != 0
        assert "failed" in prompt.lower() or "timed out" in prompt.lower()
    attempts = sorted((folder / "failed_runs").iterdir())
    assert len(attempts) == 2
    for attempt in attempts:
        assert (attempt / "artifacts/partial.txt").read_text() == "partial measurement"
        assert (attempt / "experiment.py").read_text() == script
        record = json.loads((attempt / "failure.json").read_text())
        assert record["reason"] == reason
        assert "deliberate diagnostic error" in record["stderr"]
    assert not (folder / "run_1").exists()
    assert (folder / "experiment.py").read_text() == script


def test_successful_run_keeps_original_result_contract(original):
    module = configured(original)
    folder, _ = experiment(original, "(out / 'final_info.json').write_text('{\"lesson\": {\"means\": {\"loss\": 0.5}}}')\n")
    code, prompt = module.run_experiment(str(folder), 1, timeout=10)
    assert code == 0 and "0.5" in prompt
    assert (folder / "run_1/final_info.json").is_file()
    assert not (folder / "failed_runs").exists()


class FinishedAfterOne:
    def __init__(self, responses):
        self.responses = iter(responses)

    def run(self, prompt):
        # The remote coder is the only substitute; execution and file IO remain real.
        return next(self.responses)


@pytest.mark.parametrize("plot_exit,want", [(3, False), (0, True)])
def test_plotting_failure_cannot_be_reported_as_completed_research(original, plot_exit, want):
    module = configured(original)
    folder, _ = experiment(original, "(out / 'final_info.json').write_text('{\"lesson\": {\"means\": {\"loss\": 0.5}}}')\n")
    (folder / "plot.py").write_text(f"raise SystemExit({plot_exit})\n")
    # Enough responses for the original retry limit and final notes, without a paid API.
    coder = FinishedAfterOne(["run it", "ALL_COMPLETED"] + ["plot"] * 8)
    result = module.perform_experiments({"Title": "tiny test", "Experiment": "one comparison"}, str(folder), coder, {})
    assert result is want


def test_claiming_finished_without_a_new_experiment_is_incomplete(original):
    module = configured(original)
    folder, _ = experiment(original, "sys.exit(0)\n")
    (folder / "plot.py").write_text("pass\n")
    coder = FinishedAfterOne(["ALL_COMPLETED"] + ["notes"] * 8)
    assert module.perform_experiments({"Title": "test", "Experiment": "test"}, str(folder), coder, {}) is False
    assert not (folder / "run_1.py").exists()


def test_configuration_is_repeatable_but_preserves_student_source_edits(original):
    configured(original)
    receipt = (original / "teaching_failure_setup.json").read_bytes()
    configured(original)
    assert (original / "teaching_failure_setup.json").read_bytes() == receipt
    source = original / "ai_scientist/perform_experiments.py"
    source.write_text(source.read_text() + "\n# student's edit\n")
    with pytest.raises(ValueError, match="changed"):
        upstream.configure_v1_failures(original)
    assert "student's edit" in source.read_text()


def test_configuration_refuses_unrecorded_source_changes(original):
    source = original / "ai_scientist/perform_experiments.py"
    source.write_text(source.read_text() + "\n# student's edit\n")
    adapt = getattr(upstream, "configure_v1_failures", None)
    assert callable(adapt), "missing failure adapter"
    with pytest.raises(ValueError, match="local changes"):
        adapt(original)
    assert not (original / "teaching_failure_setup.json").exists()


def test_cli_configuration_writes_a_receipt_without_running_research(original, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["upstream", "configure-failures", "v1", "--root", str(original.parent)])
    upstream.main()
    receipt = json.loads((original / "teaching_failure_setup.json").read_text())
    assert receipt["status"] == "configured_not_run"
    assert (original / receipt["changes"]).is_file()
    assert not (original / "results").exists()
