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
from pathlib import Path

from nanoscigpt.classroom import RUNNABLE_DOMAINS, validate_domain_data

from .evaluator import evaluate_train, evaluate_train_gain
from .state import ResearchState
from .tools import CONTRACTS, run_tool


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
    structured = domain in {"weather", "crystal", "structure3d", "image", "spectrum", "field"}
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
        "fixed": ["data", "split", "evaluation"],
        "allowed_change": "training_budget",
        "classroom_rounds": 1,
    }


def write_iteration_spec(baseline, out_root):
    """Persist the classroom optimization contract before any new run."""
    output_dir = Path(out_root) / baseline["domain"]
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "mode": "classroom_deep_iteration",
        "baseline": {
            "version": baseline["version"],
            "report_path": baseline["report_path"],
            "checkpoint": baseline["checkpoint"],
            "primary_metric": baseline["primary_metric"],
            "baseline_value": baseline["baseline_value"],
            "downstream": baseline.get("downstream"),
        },
        "iteration": {
            "fixed": baseline["fixed"],
            "change": baseline["allowed_change"],
            "rounds": baseline["classroom_rounds"],
            "outputs": ["comparison.json", "research_state"],
        },
    }
    spec_path = output_dir / "iteration_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec_path


def write_comparison(baseline, evidence, out_root):
    """Persist the observed V0-to-V1 comparison from the formal evaluator."""
    output_dir = Path(out_root) / baseline["domain"]
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "domain": baseline["domain"],
        "from_version": baseline["version"],
        "to_version": "V1",
        "primary_metric": baseline["primary_metric"],
        "baseline_value": baseline["baseline_value"],
        "observed_delta": evidence.get("value"),
        "evidence_level": evidence["level"],
        "criterion_passed": evidence["passed"],
        "reason": evidence["reason"],
    }
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison_path


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
            if self.domain in {"weather", "crystal", "structure3d", "image", "spectrum", "field"}:
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
