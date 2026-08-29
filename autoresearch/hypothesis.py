"""S1: Hypothesis generation, modeled on AstroInsight (EPJ Data Science 2026).

AstroInsight pipeline stages (from the paper):
  1. conception          - draft research ideas from data/literature signals
  2. iterative refinement - score, critique, and rewrite ideas over rounds
  3. expert validation   - human judges rate novelty/feasibility; low-score ideas die
  4. knowledge integration - accepted ideas join a persistent idea bank

This module reproduces that ARCHITECTURE with rule-based generation (no LLM):
each stage is a real, readable function so students can trace exactly how an
idea moves from raw data statistics to a validated hypothesis with a score.

Usage:
    python -m autoresearch.hypothesis --domain text
"""

import argparse
import json
from pathlib import Path


class HypothesisEngine:
    """Rule-based stand-in for AstroInsight's idea-generation pipeline."""

    def __init__(self, domain, state):
        self.domain = domain
        self.state = state

    # ---- stage 1: conception -------------------------------------------
    def conceive(self, data_stats):
        """Turn raw data statistics into candidate research ideas.

        AstroInsight starts from data-driven signals; here the signals are
        the actual token statistics of the domain corpus, so every idea is
        grounded in real numbers students can recompute.
        """
        vocab = data_stats["vocab_size"]
        n_seq = data_stats["n_sequences"]
        mode = data_stats["mode"]
        ideas = []

        # idea template A: compression / entropy angle
        ideas.append({
            "id": "IDEA-A",
            "text": (f"For domain '{self.domain}' (vocab={vocab}, {n_seq} sequences, "
                     f"mode={mode}), a causal LM can compress the corpus below the "
                     f"unigram entropy baseline, indicating exploitable sequence structure."),
            "grounding": {"metric": "val_loss", "baseline": "unigram entropy"},
            "novelty": 2, "feasibility": 5,
        })
        # idea template B: transfer angle
        ideas.append({
            "id": "IDEA-B",
            "text": (f"Representations from a '{self.domain}' causal LM transfer to a "
                     f"downstream probe and beat one-hot encoding, justifying a "
                     f"foundation-model claim at teaching scale."),
            "grounding": {"metric": "probe_acc", "baseline": "one-hot"},
            "novelty": 3, "feasibility": 4,
        })
        # idea template C: the honest negative (the most valuable idea in class)
        ideas.append({
            "id": "IDEA-C",
            "text": (f"At {n_seq} sequences the pretraining gain is too small to support "
                     f"any foundation-model claim; the correct route is a specialized "
                     f"model, and the negative result is the finding."),
            "grounding": {"metric": "transfer_delta", "baseline": "0"},
            "novelty": 4, "feasibility": 5,
        })
        return ideas

    # ---- stage 2: iterative refinement ----------------------------------
    def refine(self, ideas, round_idx):
        """One refinement round: critique each idea, sharpen or kill it.

        AstroInsight iterates draft -> critique -> rewrite; here the critique
        is an explicit rubric (novelty + feasibility - vagueness), and ideas
        below the bar get rewritten to be more specific, mirroring the paper's
        refinement loop.
        """
        for idea in ideas:
            vagueness = 2 if "foundation-model claim" in idea["text"] and idea["novelty"] < 4 else 0
            score = idea["novelty"] + idea["feasibility"] - vagueness
            idea["critique_score"] = score
            idea["history"] = idea.get("history", [])
            idea["history"].append({"round": round_idx, "score": score,
                                    "action": "scored"})
            if score < 5:
                # rewrite: make the idea more concrete
                idea["text"] += " Refined: specify the acceptance threshold before claiming support."
                idea["novelty"] += 1
                idea["history"].append({"round": round_idx, "score": idea["novelty"] + idea["feasibility"],
                                        "action": "rewritten for specificity"})
        return ideas

    # ---- stage 3: expert validation --------------------------------------
    def validate(self, ideas, auto_approve=False):
        """Human gate: a human expert rates novelty/feasibility on a 1-6 scale.

        In AstroInsight human experts score novelty at 3+/6; here the score is
        either typed by the human or (classroom mode) derived from the rule
        rubric with an explicit simulation note.
        """
        for idea in ideas:
            if auto_approve:
                idea["expert_score"] = min(6, idea["critique_score"])
                idea["validation_mode"] = "simulated (classroom)"
            else:
                try:
                    s = input(f"  rate idea {idea['id']} novelty+feasibility (1-6): ").strip()
                    idea["expert_score"] = max(1, min(6, int(s)))
                    idea["validation_mode"] = "human"
                except (ValueError, EOFError):
                    idea["expert_score"] = idea["critique_score"]
                    idea["validation_mode"] = "rubric fallback"
            idea["accepted"] = idea["expert_score"] >= 3  # AstroInsight: 3+/6 = novel
        return ideas

    # ---- stage 4: knowledge integration -----------------------------------
    def integrate(self, ideas):
        """Accepted ideas join the research state as hypotheses for S2."""
        for idea in ideas:
            if idea["accepted"]:
                self.state.add_hypothesis(idea["id"], idea["text"])
            else:
                score = idea.get("expert_score", "unrated")
                self.state.add_boundary_note(f"idea {idea['id']} rejected at expert "
                                             f"validation (score {score}/6)")
        return [i for i in ideas if i["accepted"]]


def load_data_stats(domain):
    """Recompute the actual corpus statistics that ground each idea.

    meta.json only carries vocab_size and mode; the sequence count is
    recovered from the actual data files so the grounding stays real.
    """
    root = Path(__file__).resolve().parent.parent / "data" / domain
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    if "n_sequences" not in meta:
        n_seq = 0
        if (root / "train_seqs.npy").exists():
            import numpy as np
            n_seq = int(np.load(root / "train_seqs.npy", allow_pickle=True).shape[0])
        elif (root / "train.bin").exists():
            # streaming domain: report total tokens instead
            n_seq = (root / "train.bin").stat().st_size // 2  # uint16 tokens
            meta["data_unit"] = "tokens"
        meta["n_sequences"] = n_seq
    return meta


def main():
    p = argparse.ArgumentParser(description="S1: hypothesis generation (AstroInsight-style)")
    p.add_argument("--domain", default="text", choices=["text", "dna", "protein", "smiles"])
    p.add_argument("--refine_rounds", type=int, default=2)
    p.add_argument("--auto_approve", action="store_true",
                   help="simulate the human expert gate in classroom mode")
    args = p.parse_args()

    from .state import ResearchState
    state_path = Path(__file__).resolve().parent / f"research_state_{args.domain}.json"
    st = ResearchState(state_path)

    print(f"[S1 hypothesis] domain={args.domain}")
    stats = load_data_stats(args.domain)
    print(f"  data grounding: vocab={stats.get('vocab_size')}, "
          f"sequences={stats.get('n_sequences', 'stream')}, mode={stats.get('mode')}")

    engine = HypothesisEngine(args.domain, st)
    ideas = engine.conceive(stats)
    for r in range(1, args.refine_rounds + 1):
        ideas = engine.refine(ideas, r)
        print(f"  refine round {r}: " + ", ".join(
            f"{i['id']}={i['critique_score']}" for i in ideas))
    ideas = engine.validate(ideas, auto_approve=args.auto_approve)
    accepted = engine.integrate(ideas)

    print("\n=== idea bank ===")
    for i in ideas:
        mark = "ACCEPT" if i["accepted"] else "reject"
        print(f"  {i['id']} [{mark}] expert={i['expert_score']}/6 "
              f"({i['validation_mode']}) - {i['text'][:90]}...")
    st.save()
    print(f"\naccepted hypotheses -> {state_path}")


if __name__ == "__main__":
    main()
