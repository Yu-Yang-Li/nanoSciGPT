import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def _text_v0(tmp_path: Path) -> Path:
    from nanoscigpt.classroom import run_domain

    run_domain("text", "smoke", DATA_ROOT, tmp_path / "classroom", cwd=ROOT)
    return tmp_path / "classroom" / "text" / "run_report.json"


def _weather_v0(tmp_path: Path) -> Path:
    from nanoscigpt.classroom import run_domain

    run_domain("weather", "smoke", DATA_ROOT, tmp_path / "classroom", cwd=ROOT)
    return tmp_path / "classroom" / "weather" / "run_report.json"


def test_autoresearch_is_included_in_the_installable_packages():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = project["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "nanoscigpt*" in includes
    assert "autoresearch*" in includes


def test_plan_inherits_the_v0_command_and_changes_only_training_budget(tmp_path):
    baseline = _text_v0(tmp_path)
    out_root = tmp_path / "autoresearch"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch.experiment",
            "--domain",
            "text",
            "--baseline_run",
            str(baseline),
            "--out_root",
            str(out_root),
            "--plan_only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    spec = json.loads((out_root / "text" / "iteration_spec.json").read_text(encoding="utf-8"))
    assert spec["iteration"]["changed"] == {
        "field": "max_iters",
        "from": 2,
        "to": 4,
    }
    assert spec["iteration"]["fixed_arguments"]["n_layer"] == "1"
    assert spec["iteration"]["fixed_arguments"]["n_embd"] == "16"
    assert spec["candidate"]["output_dir"].startswith(str((out_root / "text").resolve()))


def test_execute_writes_an_isolated_candidate_and_complete_comparison(tmp_path):
    baseline = _text_v0(tmp_path)
    out_root = tmp_path / "autoresearch"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch.experiment",
            "--domain",
            "text",
            "--baseline_run",
            str(baseline),
            "--out_root",
            str(out_root),
            "--auto_approve",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    domain_out = out_root / "text"
    candidate = json.loads(
        (domain_out / "candidate_run_report.json").read_text(encoding="utf-8")
    )
    comparison = json.loads((domain_out / "comparison.json").read_text(encoding="utf-8"))
    state = json.loads((domain_out / "research_state.json").read_text(encoding="utf-8"))

    assert Path(candidate["artifacts"]["train_log"]).is_relative_to(domain_out.resolve())
    assert comparison["candidate_value"] == candidate["primary_metric_value"]
    assert comparison["metric_direction"] == "lower_is_better"
    assert comparison["threshold"] == 0.05
    assert comparison["next_action"] in {"retain_candidate", "stop_branch"}
    assert comparison["candidate_run_report"] == str(
        (domain_out / "candidate_run_report.json").resolve()
    )
    assert state["next_action"] == comparison["next_action"]

    assert comparison["baseline"] == comparison["baseline_value"]
    assert comparison["candidate"] == comparison["candidate_value"]
    assert comparison["delta"] == comparison["observed_delta"]
    assert comparison["direction"] == comparison["metric_direction"]


def test_structured_candidate_reads_the_model_log_from_its_isolated_run(tmp_path):
    baseline = _weather_v0(tmp_path)
    out_root = tmp_path / "autoresearch"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch.experiment",
            "--domain",
            "weather",
            "--baseline_run",
            str(baseline),
            "--out_root",
            str(out_root),
            "--auto_approve",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    candidate = json.loads(
        (out_root / "weather" / "candidate_run_report.json").read_text(encoding="utf-8")
    )
    assert candidate["changed"] == {
        "field": "pretrain_steps",
        "from": 2,
        "to": 4,
    }
    assert Path(candidate["artifacts"]["train_log"]) == (
        out_root / "weather" / "candidate" / "model" / "train_log.json"
    ).resolve()


def test_execute_requires_explicit_classroom_approval(tmp_path):
    baseline = _text_v0(tmp_path)
    out_root = tmp_path / "autoresearch"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch.experiment",
            "--domain",
            "text",
            "--baseline_run",
            str(baseline),
            "--out_root",
            str(out_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode != 0
    assert "review iteration_spec.json" in completed.stderr
    assert not (out_root / "text" / "candidate_run_report.json").exists()
