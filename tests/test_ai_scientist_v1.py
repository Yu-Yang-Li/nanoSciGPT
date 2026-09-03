import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
DOMAINS = {
    "text",
    "protein",
    "dna",
    "smiles",
    "weather",
    "crystal",
    "structure3d",
    "image",
    "spectrum",
    "field",
}


@pytest.fixture(scope="module")
def completed_autoresearch(tmp_path_factory):
    from autoresearch.experiment import load_baseline_run, run_baseline_iteration
    from nanoscigpt.classroom import run_domain

    root = tmp_path_factory.mktemp("v1-real")
    classroom_root = root / "classroom"
    autoresearch_root = root / "autoresearch"
    report = run_domain("text", "smoke", DATA_ROOT, classroom_root, cwd=ROOT)
    baseline = load_baseline_run("text", classroom_root / "text" / "run_report.json")
    from autoresearch.experiment import write_iteration_spec

    write_iteration_spec(baseline, autoresearch_root)
    run_baseline_iteration(baseline, autoresearch_root)
    assert report["status"] == "completed"
    return autoresearch_root / "text"


def _run_v1(autoresearch_dir, out_dir, mode, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch.v1",
            "--domain",
            "text",
            "--autoresearch-dir",
            str(autoresearch_dir),
            "--out-dir",
            str(out_dir),
            mode,
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_v1_plan_contains_one_route_and_offline_related_work(completed_autoresearch, tmp_path):
    out_dir = tmp_path / "v1"
    completed = _run_v1(completed_autoresearch, out_dir, "--plan-only")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    related = json.loads((out_dir / "related_work.json").read_text(encoding="utf-8"))
    assert plan["route_count"] == 1
    assert plan["route"]["changed"]["field"] == "max_iters"
    assert related["novelty_assessment"] == "not_performed_offline"
    assert related["sources"]
    assert not (out_dir / "draft.md").exists()


def test_v1_confirm_uses_existing_evidence_and_writes_traceable_outputs(
    completed_autoresearch, tmp_path
):
    out_dir = tmp_path / "v1"
    completed = _run_v1(completed_autoresearch, out_dir, "--confirm-plan")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = {
        "plan.json",
        "related_work.json",
        "results.json",
        "results.csv",
        "evidence_map.json",
        "draft.md",
        "review.json",
        "claim_boundary.md",
        "workflow_state.json",
    }
    assert expected <= {path.name for path in out_dir.iterdir()}
    assert (out_dir / "figures" / "v0-v1.svg").is_file()

    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    evidence = json.loads((out_dir / "evidence_map.json").read_text(encoding="utf-8"))
    review = json.loads((out_dir / "review.json").read_text(encoding="utf-8"))
    workflow = json.loads((out_dir / "workflow_state.json").read_text(encoding="utf-8"))
    draft = (out_dir / "draft.md").read_text(encoding="utf-8")

    assert results["evidence_level"] == "evaluated"
    assert all(item["source"] == str((completed_autoresearch / "comparison.json").resolve()) for item in evidence["claims"])
    assert review["verdict"] == "ready_for_human_review"
    assert review["official_v1_reviewer_reproduced"] is False
    assert workflow["route_count"] == 1
    assert workflow["implementation"]["reproduces_original_system"] is False
    if not results["criterion_passed"]:
        assert "未达到" in draft


def test_v1_blocks_writing_when_evaluated_comparison_is_missing(tmp_path):
    autoresearch_dir = tmp_path / "autoresearch"
    autoresearch_dir.mkdir()
    (autoresearch_dir / "iteration_spec.json").write_text(
        json.dumps(
            {
                "baseline": {"primary_metric": "best_val_loss"},
                "iteration": {
                    "changed": {"field": "max_iters", "from": 2, "to": 4},
                    "fixed_arguments": {"n_layer": "1"},
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "v1"

    completed = _run_v1(autoresearch_dir, out_dir, "--confirm-plan")

    assert completed.returncode == 2
    status = json.loads((out_dir / "workflow_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "blocked_no_evaluated_evidence"
    assert not (out_dir / "draft.md").exists()


def test_offline_related_work_covers_every_classroom_domain():
    catalog = json.loads(
        (DATA_ROOT / "course" / "ai_scientist_v1_literature.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(catalog["domains"]) == DOMAINS
    for domain in DOMAINS:
        assert catalog["domains"][domain]
        assert all(source["checked"] is True for source in catalog["domains"][domain])


def test_v1_cli_help_is_domain_neutral():
    completed = subprocess.run(
        [sys.executable, "-m", "autoresearch.v1", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0
    assert "AI Scientist v1 classroom workflow" in completed.stdout


def test_v1_refuses_to_overwrite_completed_workflow_without_explicit_flag(
    completed_autoresearch, tmp_path
):
    out_dir = tmp_path / "v1"
    first = _run_v1(completed_autoresearch, out_dir, "--confirm-plan")
    assert first.returncode == 0, first.stdout + first.stderr
    original = (out_dir / "workflow_state.json").read_text(encoding="utf-8")

    repeated = _run_v1(completed_autoresearch, out_dir, "--confirm-plan")

    assert repeated.returncode == 2
    assert "--overwrite" in repeated.stderr
    assert (out_dir / "workflow_state.json").read_text(encoding="utf-8") == original


def test_v1_can_confirm_an_unchanged_plan_in_the_same_directory(
    completed_autoresearch, tmp_path
):
    out_dir = tmp_path / "v1"
    planned = _run_v1(completed_autoresearch, out_dir, "--plan-only")
    assert planned.returncode == 0, planned.stdout + planned.stderr

    confirmed = _run_v1(completed_autoresearch, out_dir, "--confirm-plan")

    assert confirmed.returncode == 0, confirmed.stdout + confirmed.stderr
    assert (out_dir / "workflow_state.json").is_file()
