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
decide the next action (train longer, probe transfer, or stop).

Usage:
    python -m autoresearch.experiment --domain text --auto_approve
"""

import argparse
from pathlib import Path

from .evaluator import evaluate_train, evaluate_train_gain, evaluate_transfer_probe
from .state import ResearchState
from .tools import CONTRACTS, run_tool


class ExperimentLoop:
    """Rule-based closed loop over the repo's contracted tools."""

    def __init__(self, domain, state, auto_approve=False):
        self.domain = domain
        self.state = state
        self.auto_approve = auto_approve

    def plan(self):
        """Stage 1: build the experiment plan from current hypotheses.

        StarWhisper plans observations from target visibility; we plan
        experiments from open hypotheses. The plan is a data structure, not
        free text, so students can see that "planning" = selecting actions.
        """
        supported = [h for h in self.state.data["hypotheses"] if h["status"] == "proposed"]
        plan = {
            "domain": self.domain,
            "steps": [
                {"tool": "prepare", "reason": "data must exist before any claim"},
                {"tool": "train_v0", "reason": "H1: smallest viable baseline"},
                {"tool": "train_extended", "reason": "H2: budget sensitivity",
                 "gate": "budget_increase"},
                {"tool": "sample", "reason": "generation sanity check"},
            ],
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
            root = Path(__file__).resolve().parent.parent / "data" / self.domain
            has_data = (root / "meta.json").exists() and (
                (root / "train.bin").exists() or (root / "train_seqs.npy").exists())
            result = {"level": "ran", "passed": has_data,
                      "reason": "data files present" if has_data else "data files missing"}
        elif tool == "train_v0":
            result = evaluate_train(self.domain)
        elif tool == "train_extended":
            baseline = self._baseline()
            result = evaluate_train_gain(self.domain, baseline)
        elif tool == "sample":
            result = {"level": "ran", "passed": True, "reason": "checkpoint generated samples"}
        else:
            result = {"level": "ran", "passed": False, "reason": "no analyzer for tool"}
        result["tool"] = tool
        return result

    def _baseline(self):
        import json
        log = json.loads((Path("out") / self.domain / "train_log.json").read_text(encoding="utf-8"))
        return log["best_val_loss"]

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
            return "train_v0"
        elif evidence.get("metric") == "best_val_loss":
            return "train_extended"
        elif evidence.get("metric") == "loss_gain_vs_v0":
            return "sample"  # even a failed gain is a finding; keep going
        elif tool == "sample":
            return "conclude"
        return "conclude"


def main():
    p = argparse.ArgumentParser(description="S2: experiment closed loop (StarWhisper-style)")
    p.add_argument("--domain", default="text", choices=["text", "dna", "protein", "smiles"])
    p.add_argument("--auto_approve", action="store_true")
    args = p.parse_args()

    state_path = Path(__file__).resolve().parent / f"research_state_{args.domain}.json"
    st = ResearchState(state_path)
    loop = ExperimentLoop(args.domain, st, auto_approve=args.auto_approve)

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
