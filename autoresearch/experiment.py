"""S2: Experiment/observation closed loop, modeled on StarWhisper Telescope
(Communications Engineering 2025).

StarWhisper Telescope loop stages (from the paper):
  1. observation planning   - generate a site/time-specific observation list
  2. function-call execution - telescope control + image acquisition via tools
  3. real-time analysis     - pipeline processes images, extracts transients
  4. dynamic follow-up       - detection triggers the next observation proposal

This module reproduces that ARCHITECTURE on the nanoSciGPT repo: plan which
experiment to run (which domain, which budget), execute it through tool
contracts, analyze the result with the formal evaluator, and let the result
decide the next action (train longer, sample, or stop).

Usage:
    python -m autoresearch.experiment --domain text --auto_approve
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nanoscigpt.classroom import validate_domain_data
from nanoscigpt.domains.registry import RUNNABLE_DOMAINS, STRUCTURED_DOMAINS

from .evaluator import CRITERIA, evaluate_loss_gain, evaluate_train, evaluate_train_gain
from .state import ResearchState
from .tools import CONTRACTS, run_tool


REPO_ROOT = Path(__file__).resolve().parent.parent
def command_options(command):
    """Return simple --name value options from a recorded classroom command."""
    options = {}
    index = 0
    while index < len(command):
        token = str(command[index])
        if token.startswith("--") and index + 1 < len(command):
            value = str(command[index + 1])
            if not value.startswith("--"):
                options[token[2:]] = value
                index += 2
                continue
        index += 1
    return options


def replace_option(command, name, value):
    updated = [str(part) for part in command]
    option = f"--{name}"
    try:
        index = updated.index(option)
    except ValueError as error:
        raise ValueError(f"recorded V0 command has no {option}") from error
    if index + 1 >= len(updated):
        raise ValueError(f"recorded V0 command has no value for {option}")
    updated[index + 1] = str(value)
    return updated


def build_candidate_command(baseline, out_root):
    """Clone the recorded V0 training command and change only its budget."""
    commands = baseline.get("commands") or []
    if not commands:
        raise ValueError("classroom V0 report does not record its training command")
    original = [str(part) for part in commands[0]]
    original[0] = sys.executable
    structured = baseline["domain"] in STRUCTURED_DOMAINS
    budget_field = "pretrain_steps" if structured else "max_iters"
    options = command_options(original)
    if budget_field not in options:
        raise ValueError(f"classroom V0 command does not record --{budget_field}")
    baseline_budget = int(options[budget_field])
    candidate_budget = max(baseline_budget + 1, baseline_budget * 2)

    domain_out = (Path(out_root) / baseline["domain"]).resolve()
    candidate_root = domain_out / "candidate"
    command_out_dir = candidate_root if structured else candidate_root / "model"
    candidate_model_dir = candidate_root / "model"
    command = replace_option(original, budget_field, candidate_budget)
    command = replace_option(command, "out_dir", command_out_dir)
    fixed = command_options(original)
    fixed.pop(budget_field, None)
    fixed.pop("out_dir", None)
    candidate_options = command_options(command)
    changed_options = {
        name
        for name in set(options) | set(candidate_options)
        if options.get(name) != candidate_options.get(name)
    }
    if changed_options != {budget_field, "out_dir"}:
        raise ValueError(
            f"candidate command changed unexpected options: {sorted(changed_options)}"
        )
    return {
        "command": command,
        "output_dir": str(candidate_root),
        "command_out_dir": str(command_out_dir),
        "model_dir": str(candidate_model_dir),
        "changed": {
            "field": budget_field,
            "from": baseline_budget,
            "to": candidate_budget,
        },
        "fixed_arguments": fixed,
    }


def load_baseline_run(domain, report_path):
    """Load the V0 artifacts produced by the classroom runner."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"classroom V0 report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError(f"classroom V0 is not completed: {report_path}")
    if report.get("domain") != domain:
        raise ValueError(
            f"classroom V0 domain={report.get('domain')} does not match requested {domain}"
        )

    artifacts = report.get("artifacts", {})
    checkpoint_path = Path(artifacts.get("checkpoint", ""))
    train_log_path = Path(artifacts.get("train_log", ""))
    if not checkpoint_path.is_file() or not train_log_path.is_file():
        raise FileNotFoundError("classroom V0 checkpoint or train log is missing")

    train_log = json.loads(train_log_path.read_text(encoding="utf-8"))
    structured = domain in STRUCTURED_DOMAINS
    primary_metric = "pretrain_val_loss" if structured else "best_val_loss"
    if primary_metric not in train_log:
        raise ValueError(f"classroom V0 train log has no {primary_metric}")

    downstream = None
    downstream_value = artifacts.get("downstream")
    if downstream_value:
        downstream_path = Path(downstream_value)
        if downstream_path.is_file():
            downstream = json.loads(downstream_path.read_text(encoding="utf-8"))

    return {
        "version": "V0",
        "domain": domain,
        "report_path": str(report_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "train_log": str(train_log_path.resolve()),
        "primary_metric": primary_metric,
        "baseline_value": float(train_log[primary_metric]),
        "downstream": downstream,
        "profile": report.get("profile"),
        "preflight": report.get("preflight"),
        "commands": report.get("commands", []),
        "fixed": ["data", "split", "evaluation"],
        "allowed_change": "training_budget",
        "classroom_rounds": 1,
    }


def write_iteration_spec(baseline, out_root):
    """Persist the classroom optimization contract before any new run."""
    output_dir = Path(out_root) / baseline["domain"]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = None
    if baseline.get("commands"):
        candidate = build_candidate_command(baseline, out_root)
    spec = {
        "mode": "classroom_deep_iteration",
        "baseline": {
            "version": baseline["version"],
            "report_path": baseline["report_path"],
            "checkpoint": baseline["checkpoint"],
            "primary_metric": baseline["primary_metric"],
            "baseline_value": baseline["baseline_value"],
            "downstream": baseline.get("downstream"),
            "profile": baseline.get("profile"),
            "training_command": (baseline.get("commands") or [None])[0],
        },
        "iteration": {
            "fixed": baseline["fixed"],
            "change": baseline["allowed_change"],
            "rounds": baseline["classroom_rounds"],
            "outputs": [
                "candidate_run_report.json",
                "comparison.json",
                "research_state.json",
            ],
        },
    }
    if candidate:
        spec["candidate"] = {
            "command": candidate["command"],
            "output_dir": candidate["output_dir"],
        }
        spec["iteration"]["changed"] = candidate["changed"]
        spec["iteration"]["fixed_arguments"] = candidate["fixed_arguments"]
    spec_path = output_dir / "iteration_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec_path


def write_comparison(baseline, evidence, out_root):
    """Persist the observed V0-to-V1 comparison from the formal evaluator."""
    output_dir = Path(out_root) / baseline["domain"]
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ("candidate_value", "value", "threshold")
    missing = [key for key in required if evidence.get(key) is None]
    if missing:
        raise ValueError(f"comparison evidence missing: {', '.join(missing)}")
    observed_delta = evidence["value"]
    candidate_value = evidence["candidate_value"]
    next_action = evidence.get(
        "next_action", "retain_candidate" if evidence["passed"] else "stop_branch"
    )
    comparison = {
        "domain": baseline["domain"],
        "from_version": baseline["version"],
        "to_version": "V1",
        "primary_metric": baseline["primary_metric"],
        "baseline_value": baseline["baseline_value"],
        "candidate_value": candidate_value,
        "observed_delta": observed_delta,
        "metric_direction": "lower_is_better",
        "threshold": evidence.get("threshold", CRITERIA["train_improve_min"]),
        "baseline": baseline["baseline_value"],
        "candidate": candidate_value,
        "delta": observed_delta,
        "direction": "lower_is_better",
        "evidence_level": evidence["level"],
        "criterion_passed": evidence["passed"],
        "next_action": next_action,
        "candidate_run_report": evidence.get("candidate_run_report"),
        "reason": evidence["reason"],
    }
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison_path


def run_baseline_iteration(baseline, out_root):
    """Run one isolated candidate from a recorded classroom V0."""
    domain_out = (Path(out_root) / baseline["domain"]).resolve()
    candidate_report_path = domain_out / "candidate_run_report.json"
    if candidate_report_path.exists():
        raise FileExistsError(
            f"candidate already exists: {candidate_report_path}; choose another --out_root"
        )
    candidate = build_candidate_command(baseline, out_root)
    candidate_root = Path(candidate["output_dir"])
    candidate_root.mkdir(parents=True, exist_ok=True)
    stdout_path = candidate_root / "stdout.txt"
    stderr_path = candidate_root / "stderr.txt"
    completed = subprocess.run(
        candidate["command"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    model_dir = Path(candidate["model_dir"])
    train_log_path = model_dir / "train_log.json"
    candidate_report = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "version": "V1",
        "domain": baseline["domain"],
        "baseline_run": baseline["report_path"],
        "command": candidate["command"],
        "changed": candidate["changed"],
        "fixed_arguments": candidate["fixed_arguments"],
        "returncode": completed.returncode,
        "artifacts": {
            "model_dir": str(model_dir),
            "train_log": str(train_log_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    }
    if completed.returncode != 0 or not train_log_path.is_file():
        candidate_report_path.write_text(
            json.dumps(candidate_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"candidate training failed; inspect {candidate_report_path} and {stderr_path}"
        )

    train_log = json.loads(train_log_path.read_text(encoding="utf-8"))
    metric = baseline["primary_metric"]
    if metric not in train_log:
        candidate_report_path.write_text(
            json.dumps(candidate_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"candidate train log has no {metric}: {train_log_path}")
    candidate_value = float(train_log[metric])
    candidate_report["primary_metric"] = metric
    candidate_report["primary_metric_value"] = candidate_value
    candidate_report_path.write_text(
        json.dumps(candidate_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    evaluation = evaluate_loss_gain(baseline["baseline_value"], candidate_value)
    delta = round(evaluation["delta"], 4)
    threshold = evaluation["threshold"]
    passed = evaluation["passed"]
    next_action = "retain_candidate" if passed else "stop_branch"
    evidence = {
        "level": "evaluated",
        "passed": passed,
        "metric": "loss_gain_vs_v0",
        "value": delta,
        "candidate_value": candidate_value,
        "threshold": threshold,
        "next_action": next_action,
        "candidate_run_report": str(candidate_report_path),
        "reason": f"gain {delta:+.4f} vs required {threshold}",
    }
    comparison_path = write_comparison(baseline, evidence, out_root)

    state_path = iteration_state_path(baseline["domain"], out_root)
    state = ResearchState(state_path)
    state.data["baseline"] = {
        "run_report": baseline["report_path"],
        "value": baseline["baseline_value"],
    }
    state.data["candidate"] = {
        "run_report": str(candidate_report_path),
        "value": candidate_value,
    }
    state.add_evidence(0, "train_candidate", "evaluated", passed, evidence["reason"])
    state.set_next_action(next_action)
    state.add_round(0, "compare_v0_v1", f"evidence -> next: {next_action}")
    state.save()
    return candidate_report_path, comparison_path, state_path


def iteration_state_path(domain, out_root):
    """Return the research-state path for a classroom V0 iteration run."""
    return Path(out_root) / domain / "research_state.json"


class ExperimentLoop:
    """Rule-based closed loop over the repo's contracted tools."""

    def __init__(self, domain, state, auto_approve=False, baseline=None):
        self.domain = domain
        self.state = state
        self.auto_approve = auto_approve
        self.baseline = baseline
        self.v0_loss = baseline["baseline_value"] if baseline else None

    def plan(self):
        """Stage 1: build the experiment plan from current hypotheses.

        StarWhisper plans observations from target visibility; we plan
        experiments from open hypotheses. The plan is a data structure, not
        free text, so students can see that "planning" = selecting actions.
        """
        supported = [h for h in self.state.data["hypotheses"] if h["status"] == "proposed"]
        steps = [
            {"tool": "prepare", "reason": "data must exist before any claim"},
            {"tool": "train_v0", "reason": "H1: smallest viable baseline"},
            {"tool": "train_extended", "reason": "H2: budget sensitivity",
             "gate": "budget_increase"},
            {"tool": "sample", "reason": "generation sanity check"},
        ]
        if self.baseline:
            steps = [step for step in steps if step["tool"] != "train_v0"]
        plan = {
            "domain": self.domain,
            "steps": steps,
        }
        return plan

    def execute_step(self, step):
        """Stage 2: execute one planned step through the tool contract.

        Mirrors StarWhisper's function-call execution: the loop never
        improvises commands; it can only invoke declared contracts.
        """
        tool = step["tool"]
        gate = step.get("gate")
        approved = True
        if gate and CONTRACTS[tool].get("requires_approval"):
            if self.auto_approve:
                print("  [human gate] SIMULATED approval (classroom mode)")
                return run_tool(tool, self.domain)
            print(f"  [human gate] '{gate}' requires approval")
            try:
                ans = input("  approve? [y/N] ").strip().lower()
                approved = ans in ("y", "yes")
            except EOFError:
                approved = False
        if not approved:
            return False, "", "human gate denied"
        return run_tool(tool, self.domain)

    def analyze(self, tool, rnd):
        """Stage 3: formal analysis of what the tool produced.

        StarWhisper runs a pipeline over images to extract transients; we run
        the formal evaluator over train logs / probe results to extract
        evidence. Both convert raw output into structured, checkable findings.
        """
        result = None
        if tool == "prepare":
            try:
                validate_domain_data(self.domain, Path(__file__).resolve().parent.parent / "data")
            except (FileNotFoundError, ValueError) as error:
                result = {"level": "ran", "passed": False, "reason": str(error)}
            else:
                result = {"level": "ran", "passed": True, "reason": "data files present"}
        elif tool == "train_v0":
            result = evaluate_train(self.domain)
            if result.get("level") == "evaluated":
                self.v0_loss = result["value"]
        elif tool == "train_extended":
            if self.v0_loss is None:
                result = {"level": "ran", "passed": False,
                          "reason": "V0 baseline was not captured before extended training"}
            else:
                result = evaluate_train_gain(self.domain, self.v0_loss)
        elif tool == "sample":
            if self.domain in STRUCTURED_DOMAINS:
                reason = "representation preview and task artifact inspected"
            else:
                reason = "checkpoint generated samples"
            result = {"level": "ran", "passed": True, "reason": reason}
        else:
            result = {"level": "ran", "passed": False, "reason": "no analyzer for tool"}
        result["tool"] = tool
        return result

    def decide(self, evidence):
        """Stage 4: evidence decides the next action.

        This is the closed-loop test: if analysis does not change the next
        action, it is not a loop, just a pipeline.
        """
        tool = evidence.get("tool", "")
        # A failed criterion is a finding, not a crash: only tool failures stop
        if not evidence["passed"] and evidence["level"] == "ran":
            return "stop"
        if tool == "prepare":
            return "train_extended" if self.baseline else "train_v0"
        elif evidence.get("metric") in ("best_val_loss", "pretrain_loss"):
            return "train_extended"
        elif evidence.get("metric") == "loss_gain_vs_v0":
            return "sample" if evidence["passed"] else "stop"
        elif tool == "sample":
            return "conclude"
        return "conclude"


def main():
    p = argparse.ArgumentParser(description="S2: model experiment iteration")
    p.add_argument("--domain", default="text", choices=RUNNABLE_DOMAINS)
    p.add_argument("--auto_approve", action="store_true")
    p.add_argument("--baseline_run", help="classroom run_report.json to use as model V0")
    p.add_argument("--out_root", default="out/autoresearch")
    p.add_argument("--plan_only", action="store_true")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    baseline = None
    if args.baseline_run:
        baseline = load_baseline_run(args.domain, args.baseline_run)
        spec_path = write_iteration_spec(baseline, args.out_root)
        print(
            f"V0 loaded: {baseline['primary_metric']}={baseline['baseline_value']:.4f}"
        )
        print(f"iteration spec -> {spec_path}")
    if args.plan_only:
        if baseline is None:
            p.error("--plan_only requires --baseline_run")
        print("plan only: no new training was started")
        return

    if baseline is not None:
        if not args.auto_approve:
            p.error("review iteration_spec.json, then rerun with --auto_approve")
        try:
            candidate_path, comparison_path, state_path = run_baseline_iteration(
                baseline, args.out_root
            )
        except (FileExistsError, RuntimeError, ValueError) as error:
            p.error(str(error))
        print(f"candidate run -> {candidate_path}")
        print(f"comparison -> {comparison_path}")
        print(f"state -> {state_path}")
        return

    state_path = (
        iteration_state_path(args.domain, args.out_root)
        if baseline
        else Path(__file__).resolve().parent / f"research_state_{args.domain}.json"
    )
    if args.fresh and state_path.exists():
        state_path.unlink()
    st = ResearchState(state_path)
    loop = ExperimentLoop(
        args.domain,
        st,
        auto_approve=args.auto_approve,
        baseline=baseline,
    )

    print(f"[S2 experiment] domain={args.domain}")
    plan = loop.plan()
    print(f"  plan: {[s['tool'] for s in plan['steps']]}")

    rnd = len(st.data["rounds"])
    for step in plan["steps"]:
        tool = step["tool"]
        print(f"\n  --- {tool} ---")
        ok, out, err = loop.execute_step(step)
        if not ok:
            st.add_evidence(rnd, tool, "ran", False, f"failed: {err[-120:]}")
            st.set_next_action("stop")
            st.add_round(rnd, tool, "tool failed; loop stops at contract boundary")
            st.save()
            rnd += 1
            break
        ev = loop.analyze(tool, rnd)
        print(f"    [evaluator] {ev['level']} passed={ev['passed']} - {ev['reason']}")
        if baseline and tool == "train_extended" and ev["level"] == "evaluated":
            comparison_path = write_comparison(baseline, ev, args.out_root)
            print(f"    V0 -> V1 comparison -> {comparison_path}")
        st.add_evidence(rnd, tool, ev["level"], ev["passed"], ev["reason"])
        next_action = loop.decide(ev)
        st.set_next_action(next_action)
        st.add_round(rnd, tool, f"evidence -> next: {next_action}")
        st.save()
        rnd += 1
        if next_action == "stop":
            print("    loop stopping: evidence did not pass")
            break
        if next_action == "conclude":
            break

    print("\n=== research state after S2 ===")
    for ev in st.data["evidence"]:
        print(f"  R{ev['round']} {ev['tool']:<16} [{ev['level']:<9}] "
              f"{'PASS' if ev['passed'] else 'fail'}: {ev['summary']}")
    print(f"state -> {state_path}")


if __name__ == "__main__":
    main()
