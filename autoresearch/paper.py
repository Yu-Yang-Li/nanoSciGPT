"""S3: Paper/review/revision workflow, modeled on the instructor's own
agentic-research practice (galactic AI Scientist baseline -> manuscript ->
audit -> revision), not on any published paper.

Workflow stages (from the instructor's real practice):
  1. draft assembly       - pull claims from the persistent research state;
                            NEVER invent numbers, only restate evidence
  2. structural review     - check each claim against its evidence level
  3. itemized revision    - each review finding becomes a numbered revision
  4. fact audit           - every number must trace to a run artifact
  5. claim boundary       - the final "what we can/cannot claim" paragraph

Usage:
    python -m autoresearch.paper --domain text
"""

import argparse
import json
import re
from pathlib import Path


class PaperWorkflow:
    """Rule-based paper pipeline over the persistent research state."""

    def __init__(self, domain, state):
        self.domain = domain
        self.state = state

    # ---- stage 1: draft assembly -----------------------------------------
    def assemble_draft(self):
        """Build the paper skeleton strictly from recorded evidence.

        The golden rule from the instructor's practice: the writing agent may
        use locked evidence but must not reanalyse data or strengthen
        conclusions. Here that rule is enforced by construction: the draft is
        generated ONLY from state.data, never from imagination.
        """
        ev = self.state.data["evidence"]
        hyp = self.state.data["hypotheses"]
        sections = {
            "abstract": f"We ran an autoresearch loop on domain '{self.domain}'.",
            "methods": [],
            "results": [],
            "discussion": [],
        }
        for h in hyp:
            sections["methods"].append(f"Hypothesis {h['id']}: {h['text']}")
        for e in ev:
            sections["results"].append(
                f"Round {e['round']} ({e['tool']}, level={e['level']}): {e['summary']}")
        return sections

    # ---- stage 2: structural review ----------------------------------------
    def review(self, draft):
        """Review each claim against its recorded evidence level.

        Every claim gets one of four verdicts; only verdicts that map to
        evidence are allowed to survive into the revised draft.
        """
        findings = []
        for e in self.state.data["evidence"]:
            if e["level"] == "design":
                findings.append({
                    "id": f"F{len(findings)+1}",
                    "claim": e["summary"],
                    "level": e["level"],
                    "verdict": "unsupported",
                    "reason": "design-level evidence cannot appear in results",
                })
            elif e["level"] == "ran" and e["passed"]:
                findings.append({
                    "id": f"F{len(findings)+1}",
                    "claim": e["summary"],
                    "level": e["level"],
                    "verdict": "overstated",
                    "reason": "'ran' means the tool completed, not that science passed",
                })
            elif e["level"] == "evaluated" and not e["passed"]:
                findings.append({
                    "id": f"F{len(findings)+1}",
                    "claim": e["summary"],
                    "level": e["level"],
                    "verdict": "boundary",
                    "reason": "failed criteria must be reported, not hidden",
                })
        return findings

    # ---- stage 3: itemized revision -----------------------------------------
    def revise(self, draft, findings):
        """Apply each finding as a numbered revision item.

        Mirrors the instructor's LAMOST review practice: itemized original ->
        issue -> fix, so every change in the final draft is traceable to a
        review finding.
        """
        revised = dict(draft)
        revised["results"] = list(draft["results"])
        for f in findings:
            if f["verdict"] == "unsupported":
                revised["results"] = [r for r in revised["results"] if f["claim"] not in r]
            elif f["verdict"] == "overstated":
                for i, r in enumerate(revised["results"]):
                    if f["claim"] in r:
                        revised["results"][i] = r + " [revision: downgraded to 'ran']"
            elif f["verdict"] == "boundary":
                for i, r in enumerate(revised["results"]):
                    ref = f.get("claim", "")
                    if ref and ref in r:
                        revised["results"][i] = r + " [revision: negative result kept visible]"
        revised["revision_items"] = [
            {"finding": f["id"], "verdict": f["verdict"], "reason": f["reason"]}
            for f in findings
        ]
        return revised

    # ---- stage 4: fact audit -------------------------------------------------
    def audit(self, revised):
        """Every number in the draft must trace to a run artifact.

        From the instructor's audit_manuscript_numbers.py practice: numbers
        without provenance fail the audit.
        """
        numbers_in_draft = set()
        for r in revised["results"]:
            numbers_in_draft.update(re.findall(r"\d+\.\d+", r))
        numbers_in_evidence = set()
        for e in self.state.data["evidence"]:
            numbers_in_evidence.update(re.findall(r"\d+\.\d+", e["summary"]))
        orphaned = numbers_in_draft - numbers_in_evidence
        return {"passed": len(orphaned) == 0, "orphaned_numbers": sorted(orphaned)}

    # ---- stage 5: claim boundary ----------------------------------------------
    def boundary(self):
        return (f"For domain '{self.domain}', we CAN claim: the loop ran end-to-end "
                f"under tool contracts with formal evaluation. We CANNOT claim: any "
                f"foundation-model capability; teaching data is {self._data_scale()}.")

    def _data_scale(self):
        meta = Path(__file__).resolve().parent.parent / "data" / self.domain / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            n = m.get("n_sequences")
            if n is None:
                seqs = Path(meta.parent) / "train_seqs.npy"
                if seqs.exists():
                    import numpy as np
                    n = int(np.load(seqs, allow_pickle=True).shape[0])
                else:
                    n = "streaming"
            return f"{n} sequences"
        return "teaching fixtures only"


def main():
    p = argparse.ArgumentParser(description="S3: paper/review/revision workflow")
    p.add_argument("--domain", default="text", choices=["text", "dna", "protein", "smiles"])
    args = p.parse_args()

    from .state import ResearchState
    state_path = Path(__file__).resolve().parent / f"research_state_{args.domain}.json"
    st = ResearchState(state_path)
    wf = PaperWorkflow(args.domain, st)

    print(f"[S3 paper] domain={args.domain}")
    draft = wf.assemble_draft()
    print(f"  draft assembled: {len(draft['results'])} result claims, "
          f"{len(draft['methods'])} hypotheses")

    findings = wf.review(draft)
    print(f"  review findings: {len(findings)}")
    for f in findings:
        print(f"    {f['id']} [{f['verdict']}] {f['reason']}")

    revised = wf.revise(draft, findings)
    print(f"  revision items: {len(revised['revision_items'])}")

    audit = wf.audit(revised)
    print(f"  fact audit: passed={audit['passed']}"
          + (f" orphaned={audit['orphaned_numbers']}" if not audit["passed"] else ""))

    boundary = wf.boundary()
    print(f"  claim boundary: {boundary}")

    out = {
        "domain": args.domain,
        "draft": draft,
        "review_findings": findings,
        "revised": revised,
        "audit": audit,
        "boundary": boundary,
    }
    out_path = Path(__file__).resolve().parent.parent / "out" / "paper" / f"paper_{args.domain}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npaper artifact -> {out_path}")


if __name__ == "__main__":
    main()
