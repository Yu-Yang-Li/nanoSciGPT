"""The autoresearch loop: a rule-based virtual scientist for the classroom.

Deliberately NO LLM inside: the policy is explicit rules so students can read
exactly how feedback changes the next step. The loop demonstrates the five
B-line concepts against the real repo:

1. executable step      -> one contracted tool call per round
2. tool contract        -> autoresearch/tools.py
3. formal evaluator     -> autoresearch/evaluator.py
4. feedback changes next step -> round policy below
5. cross-round state    -> autoresearch/state.py + research_state.json

Usage:
    python -m autoresearch.run --domain text
    python -m autoresearch.run --domain protein          # includes transfer probe
    python -m autoresearch.run --domain text --fresh     # restart state
"""

import argparse
from pathlib import Path

from .evaluator import evaluate_train, evaluate_train_gain, evaluate_transfer_probe
from .state import ResearchState
from .tools import CONTRACTS, run_tool

def _state_path(domain):
    """One research-state file per domain, so rounds survive across classrooms."""
    return Path(__file__).resolve().parent / f"research_state_{domain}.json"


def header(rnd, title):
    print(f"\n=== round {rnd}: {title} ===", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="text", choices=["text", "dna", "protein", "smiles"])
    p.add_argument("--fresh", action="store_true", help="reset research state first")
    p.add_argument("--auto_approve", action="store_true",
                   help="simulate the human gate in classroom mode (explicitly logged)")
    args = p.parse_args()

    domain = args.domain
    state_path = _state_path(domain)
    if args.fresh and state_path.exists():
        state_path.unlink()
    st = ResearchState(state_path)
    rnd = len(st.data["rounds"])
    print(f"[autoresearch] domain={domain} resuming at round {rnd}")

    # ---- Round 0: formulate hypothesis (design level) ----
    if rnd == 0:
        header(0, "formulate hypotheses")
        st.add_hypothesis("H1", f"domain '{domain}' can be pretrained: val loss drops below threshold")
        st.add_hypothesis("H2", f"more training budget improves val loss by a meaningful margin")
        st.add_hypothesis("H3", "pretrained representations transfer to a downstream probe better than one-hot")
        st.add_question("does the teaching data size support any foundation-model claim?")
        st.set_next_action("prepare")
        st.add_round(0, "design", "hypotheses H1-H3 proposed; no execution yet (level: design)")
        st.add_evidence(0, "none", "design", False, "only a plan exists")
        st.save()
        rnd += 1

    baseline_loss = None

    # ---- Round 1: prepare data ----
    if st.data.get("next_action") == "prepare":
        header(1, f"prepare data for '{domain}' (contract: prepare)")
        ok, out, err = run_tool("prepare", domain)
        if ok:
            st.add_evidence(rnd, "prepare", "ran", True, f"teaching data for '{domain}' prepared")
            st.set_next_action("train_v0")
            st.add_round(rnd, "prepare", "data prepared; moving to V0 training")
        else:
            st.add_evidence(rnd, "prepare", "ran", False, f"prepare failed: {err[-200:]}")
            st.add_question(f"why does prepare fail for '{domain}'?")
            st.set_next_action("stop")
            st.add_round(rnd, "prepare", f"FAILED - stopping (see question list): {err[-200:]}")
        st.save()
        rnd += 1

    # ---- Round 2: V0 training ----
    if st.data.get("next_action") == "train_v0":
        header(2, "train V0: smallest GPT, 100 iterations (contract: train_v0)")
        ok, out, err = run_tool("train_v0", domain)
        ev = evaluate_train(domain)
        print(f"  [evaluator] level={ev['level']} passed={ev['passed']} - {ev['reason']}")
        st.add_evidence(rnd, "train_v0", ev["level"], ev["passed"], ev["reason"])
        if ev["passed"]:
            baseline_loss = ev["value"]
            st.set_hypothesis("H1", "supported", ev["reason"])
            st.set_next_action("train_extended")
            st.add_round(rnd, "train_v0", f"H1 supported (val loss {ev['value']}); next: extend budget")
        else:
            st.set_hypothesis("H1", "refuted", ev["reason"])
            st.set_next_action("stop")
            st.add_round(rnd, "train_v0", f"H1 refuted ({ev['reason']}); stopping")
        st.save()
        rnd += 1
    else:
        # recover baseline from prior state if resuming past round 2
        for ev in st.data["evidence"]:
            if ev["tool"] == "train_v0" and ev.get("passed"):
                pass

    # ---- Round 3: extended training behind a human gate ----
    if st.data.get("next_action") == "train_extended":
        header(3, "extend budget to 200 iters - REQUIRES HUMAN APPROVAL (gate: budget_increase)")
        contract_gate = CONTRACTS["train_extended"]["requires_approval"]
        approved = False
        if args.auto_approve:
            approved = True
            print("  [human gate] SIMULATED approval in classroom mode (--auto_approve)")
            st.add_boundary_note("budget_increase approval was simulated with --auto_approve; "
                                 "in real use a human must sign off")
        else:
            try:
                ans = input(f"  approve '{contract_gate}' for domain '{domain}'? [y/N] ").strip().lower()
                approved = ans in ("y", "yes")
            except EOFError:
                approved = False
            print(f"  [human gate] human decision: {'approved' if approved else 'denied'}")
        if approved:
            import json as _json
            log = _json.loads((Path("out") / domain / "train_log.json").read_text(encoding="utf-8"))
            baseline_loss = log["best_val_loss"]
            ok, out, err = run_tool("train_extended", domain)
            ev = evaluate_train_gain(domain, baseline_loss)
            print(f"  [evaluator] level={ev['level']} passed={ev['passed']} - {ev['reason']}")
            st.add_evidence(rnd, "train_extended", ev["level"], ev["passed"], ev["reason"])
            if ev["passed"]:
                st.set_hypothesis("H2", "supported", ev["reason"])
            else:
                st.set_hypothesis("H2", "refuted", ev["reason"] + " - diminishing returns on tiny data is itself a finding")
            st.set_next_action("probe_or_sample")
            st.add_round(rnd, "train_extended", f"executed under approval; H2 resolved")
        else:
            st.set_hypothesis("H2", "inconclusive", "human gate denied; experiment not run")
            st.set_next_action("probe_or_sample")
            st.add_round(rnd, "train_extended", "gate denied - respecting stop condition")
        st.save()
        rnd += 1

    # ---- Round 4: transfer probe (protein) or sampling check ----
    if st.data.get("next_action") == "probe_or_sample":
        if domain == "protein":
            header(4, "transfer probe: does OUR pretraining beat one-hot? (contract: transfer_probe)")
            ok, out, err = run_tool("transfer_probe", domain)
            ev = evaluate_transfer_probe()
            print(f"  [evaluator] level={ev['level']} passed={ev['passed']} - {ev['reason']}")
            st.add_evidence(rnd, "transfer_probe", ev["level"], ev["passed"], ev["reason"])
            if ev["passed"]:
                st.set_hypothesis("H3", "supported", ev["reason"])
            else:
                st.set_hypothesis("H3", "refuted", ev["reason"])
                st.add_boundary_note("transfer gain is small/negative at this data scale: "
                                     "no foundation-model claim is justified (mechanism identical, scale decides)")
            st.set_next_action("conclude")
            st.add_round(rnd, "transfer_probe", "H3 resolved")
        else:
            header(4, f"sampling check for '{domain}' (contract: sample)")
            ok, out, err = run_tool("sample", domain)
            if ok:
                snippet = " ".join(out.split())[:160]
                st.add_evidence(rnd, "sample", "ran", True, f"checkpoint generates: {snippet}")
                st.add_round(rnd, "sample", "generation works; H3 not applicable to this domain")
                st.add_question("transfer probe only exists for protein; add a domain-specific probe to test H3 here")
            else:
                st.add_evidence(rnd, "sample", "ran", False, err[-200:])
                st.add_round(rnd, "sample", "sampling failed")
            st.set_next_action("conclude")
        st.save()
        rnd += 1

    # ---- Round 5: conclusions with explicit claim boundaries ----
    if st.data.get("next_action") == "conclude":
        header(5, "conclusions and claim boundary")
        st.close_question("does the teaching data size support any foundation-model claim?")
        st.add_conclusion(f"what we CAN claim: the '{domain}' pipeline ran end-to-end under tool "
                          f"contracts and formal evaluation; every number above is traceable to a run")
        st.add_conclusion("what we CANNOT claim: any scientific performance or foundation-model "
                          "capability - teaching data is orders of magnitude too small")
        st.set_next_action("done")
        st.add_round(rnd, "conclude", "closed with explicit claim boundary")
        st.save()

    # ---- report ----
    print("\n=== research state (cross-round) ===")
    for h in st.data["hypotheses"]:
        res = h.get("resolution", "")
        print(f"  {h['id']} [{h['status']}] {h['text']}" + (f" | {res}" if res else ""))
    print("  evidence:")
    for ev in st.data["evidence"]:
        mark = "PASS" if ev["passed"] else ("fail" if ev["level"] != "design" else "design")
        print(f"    R{ev['round']} {ev['tool']:<16} [{ev['level']:<9}] {mark}: {ev['summary']}")
    if st.data["open_questions"]:
        print("  open questions:")
        for q in st.data["open_questions"]:
            print(f"    - {q}")
    print("  conclusions:")
    for c in st.data["conclusions"]:
        print(f"    - {c}")
    for b in st.data["boundary_notes"]:
        print(f"  [boundary] {b}")
    print(f"\nstate persisted -> {state_path}")


if __name__ == "__main__":
    main()
