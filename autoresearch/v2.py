"""CPU classroom reconstruction of AI Scientist v2 route search."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .evaluator import evaluate_loss_gain
from .experiment import (
    STRUCTURED_DOMAINS,
    command_options,
    load_baseline_run,
    replace_option,
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def init_tree(v1_state_path: Path, out_root: Path) -> Path:
    v1_path = v1_state_path.resolve()
    v1 = read_json(v1_path)
    if v1.get("status") != "evaluated" or v1.get("route_count") != 1:
        raise ValueError("v1 workflow_state.json must contain one evaluated route")
    backlog = v1.get("candidate_backlog") or []
    if not backlog:
        raise ValueError("v1 workflow_state.json has no held route")

    domain = v1["domain"]
    state_path = (out_root.resolve() / domain / "tree_state.json")
    if state_path.exists():
        raise FileExistsError(f"tree state already exists: {state_path}")
    route1 = v1["route"]
    route2 = backlog[0]
    route1_evaluation = dict(route1["result"])
    route1_evaluation["passed"] = bool(route1_evaluation["criterion_passed"])
    state = {
        "schema_version": "nanoscigpt.ai_scientist_v2.tree.v1",
        "implementation": {
            "name": "CPU classroom two-route search",
            "inspired_by": "The AI Scientist v2",
            "reproduces_original_system": False,
        },
        "domain": domain,
        "root": {
            "id": "root",
            "baseline_run": v1["baseline_run"],
            "evaluator": v1["evaluator"],
            "source_v1_state": str(v1_path),
        },
        "nodes": {
            "route-1": {
                "id": "route-1",
                "parent_id": "root",
                "status": "completed",
                "attempts": 1,
                "change": route1["change"],
                "run_report": route1["run_report"],
                "evaluation": route1_evaluation,
            },
            "route-2": {
                "id": "route-2",
                "parent_id": "root",
                "status": "planned",
                "attempts": 0,
                "change": route2["change"],
            },
        },
        "frontier": ["route-2"],
        "next_action": "review_route-2_then_run-next",
    }
    atomic_write_json(state_path, state)
    return state_path


def build_route_command(baseline: dict, change: dict, node_dir: Path) -> dict:
    commands = baseline.get("commands") or []
    if not commands:
        raise ValueError("classroom V0 report does not record its training command")
    original = [str(part) for part in commands[0]]
    original[0] = sys.executable
    field = str(change["field"])
    original_options = command_options(original)
    if field not in original_options:
        raise ValueError(f"classroom V0 command has no --{field}")

    structured = baseline["domain"] in STRUCTURED_DOMAINS
    command_out = node_dir if structured else node_dir / "model"
    model_dir = node_dir / "model"
    command = replace_option(original, field, change["to"])
    command = replace_option(command, "out_dir", command_out)
    updated_options = command_options(command)
    changed_options = {
        name
        for name in set(original_options) | set(updated_options)
        if original_options.get(name) != updated_options.get(name)
    }
    if changed_options != {field, "out_dir"}:
        raise ValueError(
            f"route command changed unexpected options: {sorted(changed_options)}"
        )
    return {
        "command": command,
        "model_dir": model_dir,
        "train_log": model_dir / "train_log.json",
    }


def run_next(state_path: Path, approved: bool) -> int:
    state_path = state_path.resolve()
    state = read_json(state_path)
    if not approved:
        print("review tree_state.json, then rerun with --approve", file=sys.stderr)
        return 2
    frontier = state.get("frontier") or []
    if not frontier:
        raise ValueError("tree has no planned route")
    node_id = frontier[0]
    node = state["nodes"][node_id]
    if node.get("status") != "planned":
        raise ValueError(f"{node_id} is not planned")

    baseline = load_baseline_run(state["domain"], state["root"]["baseline_run"])
    node_dir = state_path.parent / "nodes" / node_id
    run_spec = build_route_command(baseline, node["change"], node_dir)
    node["status"] = "running"
    node["attempts"] += 1
    atomic_write_json(state_path, state)

    node_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        run_spec["command"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (node_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (node_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    report_path = (node_dir / "run_report.json").resolve()
    report = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "domain": state["domain"],
        "route_id": node_id,
        "baseline_run": baseline["report_path"],
        "command": run_spec["command"],
        "change": node["change"],
        "returncode": completed.returncode,
        "artifacts": {
            "model_dir": str(run_spec["model_dir"].resolve()),
            "train_log": str(run_spec["train_log"].resolve()),
            "stdout": str((node_dir / "stdout.txt").resolve()),
            "stderr": str((node_dir / "stderr.txt").resolve()),
        },
    }
    if completed.returncode == 0 and run_spec["train_log"].is_file():
        log = read_json(run_spec["train_log"])
        metric = baseline["primary_metric"]
        candidate = float(log[metric])
        evaluation = evaluate_loss_gain(
            baseline["baseline_value"],
            candidate,
            state["root"]["evaluator"]["minimum_delta"],
        )
        report["primary_metric"] = metric
        report["primary_metric_value"] = candidate
        report["evaluation"] = evaluation
        node["status"] = "completed"
        node["evaluation"] = evaluation
    else:
        node["status"] = "failed"
        node["failure"] = "training failed or train_log.json is missing"
    atomic_write_json(report_path, report)
    node["run_report"] = str(report_path)
    state["frontier"] = frontier[1:]
    state["next_action"] = "decide" if not state["frontier"] else "run-next"
    atomic_write_json(state_path, state)
    print(f"{node_id} -> {report_path}")
    return 0 if node["status"] == "completed" else 1


def status_text(state: dict) -> str:
    rows = [
        f"{node_id}: {node['status']} (attempts={node['attempts']})"
        for node_id, node in state["nodes"].items()
    ]
    rows.append(f"next: {state['next_action']}")
    return "\n".join(rows)


def decide(state_path: Path) -> Path:
    state_path = state_path.resolve()
    state = read_json(state_path)
    unfinished = [
        node_id
        for node_id, node in state["nodes"].items()
        if node["status"] not in {"completed", "failed"}
    ]
    if unfinished:
        raise ValueError(f"routes are not finished: {', '.join(unfinished)}")

    baseline_value = load_baseline_run(
        state["domain"], state["root"]["baseline_run"]
    )["baseline_value"]
    passing = []
    for node_id, node in state["nodes"].items():
        evaluation = node.get("evaluation") or {}
        candidate = evaluation.get("candidate")
        if node["status"] == "completed" and evaluation.get("passed") and candidate is not None:
            passing.append((float(candidate), node_id))
    retained = min(passing)[1] if passing else "baseline"

    for node_id, node in state["nodes"].items():
        if node_id == retained:
            node["status"] = "retained"
            node["stop_reason"] = None
        else:
            node["status"] = "stopped"
            node["stop_reason"] = (
                "dominated_under_same_evaluator"
                if retained != "baseline" and node.get("evaluation", {}).get("passed")
                else "criterion_not_met"
            )
    decision = {
        "schema_version": "nanoscigpt.ai_scientist_v2.decision.v1",
        "evaluator_id": state["root"]["evaluator"]["id"],
        "metric": state["root"]["evaluator"]["metric"],
        "baseline": baseline_value,
        "retained": retained,
        "rule": "retain the lowest-loss route that clears the recorded threshold; otherwise retain baseline",
        "merge_performed": False,
        "reproduces_original_system": False,
    }
    decision_path = state_path.parent / "route_decision.json"
    atomic_write_json(decision_path, decision)
    state["frontier"] = []
    state["next_action"] = "complete"
    atomic_write_json(state_path, state)
    print(f"decision -> {decision_path}")
    return decision_path


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Scientist v2 classroom tree")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create two comparable routes from v1")
    init.add_argument("--from-v1", required=True, type=Path)
    init.add_argument("--out-root", required=True, type=Path)

    run = commands.add_parser("run-next", help="run one approved frontier route")
    run.add_argument("--state", required=True, type=Path)
    run.add_argument("--approve", action="store_true")

    status = commands.add_parser("status", help="show persisted route state")
    status.add_argument("--state", required=True, type=Path)

    choose = commands.add_parser("decide", help="apply the recorded numeric evaluator")
    choose.add_argument("--state", required=True, type=Path)

    args = parser.parse_args()
    try:
        if args.command == "init":
            path = init_tree(args.from_v1, args.out_root)
            print(f"tree -> {path}")
            print("plan only: no new training was started")
            return 0
        if args.command == "run-next":
            return run_next(args.state, args.approve)
        if args.command == "status":
            print(status_text(read_json(args.state.resolve())))
            return 0
        decide(args.state)
        return 0
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
