import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


@pytest.fixture(scope="module")
def completed_v1(tmp_path_factory):
    from autoresearch.experiment import (
        load_baseline_run,
        run_baseline_iteration,
        write_iteration_spec,
    )
    from autoresearch.v1 import build_plan, complete_workflow, load_inputs, related_work, write_json
    from nanoscigpt.classroom import run_domain

    root = tmp_path_factory.mktemp("v2-real")
    classroom_root = root / "classroom"
    autoresearch_root = root / "autoresearch"
    v1_out = root / "v1"
    run_domain("text", "smoke", DATA_ROOT, classroom_root, cwd=ROOT)
    baseline = load_baseline_run("text", classroom_root / "text" / "run_report.json")
    write_iteration_spec(baseline, autoresearch_root)
    run_baseline_iteration(baseline, autoresearch_root)
    inputs = load_inputs("text", autoresearch_root / "text")
    plan = build_plan(inputs)
    sources = related_work("text")
    v1_out.mkdir()
    write_json(v1_out / "plan.json", plan)
    write_json(v1_out / "related_work.json", sources)
    assert complete_workflow(inputs, v1_out, plan, sources) == 0
    return v1_out / "workflow_state.json"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "autoresearch.v2", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_v2_init_creates_two_sibling_routes_without_running_a_model(completed_v1, tmp_path):
    out_root = tmp_path / "v2"
    completed = _run("init", "--from-v1", completed_v1, "--out-root", out_root)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    state_path = out_root / "text" / "tree_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["implementation"]["reproduces_original_system"] is False
    assert set(state["nodes"]) == {"route-1", "route-2"}
    assert {node["parent_id"] for node in state["nodes"].values()} == {"root"}
    assert state["nodes"]["route-1"]["status"] == "completed"
    assert state["nodes"]["route-1"]["evaluation"]["passed"] is state["nodes"]["route-1"]["evaluation"]["criterion_passed"]
    assert state["nodes"]["route-2"]["status"] == "planned"
    assert state["frontier"] == ["route-2"]
    assert not (out_root / "text" / "nodes" / "route-2" / "model").exists()


def test_v2_runs_one_route_recovers_state_and_makes_a_formal_decision(
    completed_v1, tmp_path
):
    out_root = tmp_path / "v2"
    assert _run("init", "--from-v1", completed_v1, "--out-root", out_root).returncode == 0
    state_path = out_root / "text" / "tree_state.json"

    run = _run("run-next", "--state", state_path, "--approve")
    assert run.returncode == 0, run.stdout + run.stderr
    after_run = json.loads(state_path.read_text(encoding="utf-8"))
    assert after_run["nodes"]["route-2"]["status"] == "completed"
    assert after_run["nodes"]["route-2"]["attempts"] == 1
    assert Path(after_run["nodes"]["route-2"]["run_report"]).is_relative_to(
        state_path.parent.resolve()
    )

    status = _run("status", "--state", state_path)
    assert status.returncode == 0
    unchanged = json.loads(state_path.read_text(encoding="utf-8"))
    assert unchanged["nodes"]["route-2"]["attempts"] == 1

    decision_run = _run("decide", "--state", state_path)
    assert decision_run.returncode == 0, decision_run.stdout + decision_run.stderr
    decision = json.loads(
        (state_path.parent / "route_decision.json").read_text(encoding="utf-8")
    )
    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert decision["evaluator_id"] == final_state["root"]["evaluator"]["id"]
    assert decision["retained"] in {"baseline", "route-1", "route-2"}
    assert final_state["next_action"] == "complete"
    assert all(node["status"] in {"retained", "stopped"} for node in final_state["nodes"].values())


def test_v2_run_next_requires_explicit_approval(completed_v1, tmp_path):
    out_root = tmp_path / "v2"
    assert _run("init", "--from-v1", completed_v1, "--out-root", out_root).returncode == 0
    state_path = out_root / "text" / "tree_state.json"

    completed = _run("run-next", "--state", state_path)

    assert completed.returncode == 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["nodes"]["route-2"]["status"] == "planned"
    assert state["nodes"]["route-2"]["attempts"] == 0


def test_v2_run_next_resumes_the_same_route_after_an_interrupted_attempt(
    completed_v1, tmp_path
):
    out_root = tmp_path / "v2"
    assert _run("init", "--from-v1", completed_v1, "--out-root", out_root).returncode == 0
    state_path = out_root / "text" / "tree_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["nodes"]["route-2"]["status"] = "running"
    state["nodes"]["route-2"]["attempts"] = 1
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed = _run("run-next", "--state", state_path, "--approve")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    recovered = json.loads(state_path.read_text(encoding="utf-8"))
    assert recovered["nodes"]["route-2"]["status"] == "completed"
    assert recovered["nodes"]["route-2"]["attempts"] == 2
    assert recovered["nodes"]["route-2"]["resumed_after_interruption"] is True
    assert recovered["frontier"] == []


def test_v2_decision_can_retain_the_completed_v1_route(completed_v1, tmp_path):
    out_root = tmp_path / "v2"
    assert _run("init", "--from-v1", completed_v1, "--out-root", out_root).returncode == 0
    state_path = out_root / "text" / "tree_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["nodes"]["route-1"]["evaluation"].update(
        {"passed": True, "candidate": 1.0}
    )
    state["nodes"]["route-2"].update(
        {
            "status": "completed",
            "evaluation": {"passed": False, "candidate": 2.0},
        }
    )
    state["frontier"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")

    completed = _run("decide", "--state", state_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    decision = json.loads(
        (state_path.parent / "route_decision.json").read_text(encoding="utf-8")
    )
    assert decision["retained"] == "route-1"


def test_v2_cli_help_is_available():
    completed = _run("--help")
    assert completed.returncode == 0
    assert "AI Scientist v2 classroom tree" in completed.stdout
