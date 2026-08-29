"""Tests for the three-segment autoresearch pipeline."""

import json
from pathlib import Path

import pytest

from autoresearch.evaluator import evaluate_train, evaluate_transfer_probe
from autoresearch.experiment import ExperimentLoop
from autoresearch.hypothesis import HypothesisEngine, load_data_stats
from autoresearch.paper import PaperWorkflow
from autoresearch.state import ResearchState


@pytest.fixture
def state(tmp_path):
    return ResearchState(tmp_path / "state.json")


class TestHypothesis:
    def test_conceive_produces_grounded_ideas(self, state):
        engine = HypothesisEngine("text", state)
        stats = {"vocab_size": 65, "n_sequences": 1000, "mode": "stream"}
        ideas = engine.conceive(stats)
        assert len(ideas) == 3
        for idea in ideas:
            assert "grounding" in idea
            assert idea["grounding"]["metric"]

    def test_refine_scores_and_rewrites(self, state):
        engine = HypothesisEngine("text", state)
        ideas = engine.conceive({"vocab_size": 20, "n_sequences": 100, "mode": "independent"})
        ideas = engine.refine(ideas, round_idx=1)
        for idea in ideas:
            assert "critique_score" in idea
            assert len(idea["history"]) >= 1

    def test_validate_below_threshold_rejects(self, state):
        engine = HypothesisEngine("text", state)
        ideas = engine.conceive({"vocab_size": 20, "n_sequences": 100, "mode": "independent"})
        for idea in ideas:
            idea["expert_score"] = 1
            idea["accepted"] = False
        accepted = [i for i in ideas if i.get("accepted")]
        assert len(accepted) == 0

    def test_integrate_only_accepted(self, state):
        engine = HypothesisEngine("text", state)
        ideas = engine.conceive({"vocab_size": 20, "n_sequences": 100, "mode": "independent"})
        ideas[0]["accepted"] = True
        ideas[1]["accepted"] = False
        ideas[2]["accepted"] = True
        accepted = engine.integrate(ideas)
        assert len(accepted) == 2
        assert len(state.data["hypotheses"]) == 2


class TestExperimentLoop:
    def test_plan_has_four_steps(self, state):
        loop = ExperimentLoop("text", state)
        plan = loop.plan()
        assert [s["tool"] for s in plan["steps"]] == [
            "prepare", "train_v0", "train_extended", "sample"]

    def test_decide_feedback_changes_next(self, state):
        loop = ExperimentLoop("text", state)
        # prepare passed -> next is train_v0
        ev = {"tool": "prepare", "passed": True, "level": "ran"}
        assert loop.decide(ev) == "train_v0"
        # train_v0 passed (evaluated) -> extend budget
        ev = {"tool": "train_v0", "passed": True, "level": "evaluated", "metric": "best_val_loss"}
        assert loop.decide(ev) == "train_extended"
        # gain failed -> still go to sample (negative finding preserved)
        ev = {"tool": "train_extended", "passed": False, "level": "evaluated",
              "metric": "loss_gain_vs_v0"}
        assert loop.decide(ev) == "sample"
        # tool crashed -> stop
        ev = {"tool": "prepare", "passed": False, "level": "ran"}
        assert loop.decide(ev) == "stop"


class TestPaperWorkflow:
    def test_draft_only_from_state(self, state):
        state.add_hypothesis("H1", "test hypothesis")
        state.add_evidence(0, "train_v0", "evaluated", True, "val loss 2.5 < 6.0")
        wf = PaperWorkflow("text", state)
        draft = wf.assemble_draft()
        assert any("val loss 2.5" in r for r in draft["results"])

    def test_review_flags_overstated(self, state):
        state.add_evidence(0, "sample", "ran", True, "checkpoint generated samples")
        wf = PaperWorkflow("text", state)
        findings = wf.review(wf.assemble_draft())
        assert any(f["verdict"] == "overstated" for f in findings)

    def test_audit_catches_orphan_numbers(self, state):
        state.add_evidence(0, "train_v0", "evaluated", True, "val loss 2.5 < 6.0")
        wf = PaperWorkflow("text", state)
        draft = wf.assemble_draft()
        draft["results"].append("we achieved accuracy 99.7 on the benchmark")
        audit = wf.audit(draft)
        assert not audit["passed"]
        assert "99.7" in audit["orphaned_numbers"]

    def test_revise_downgrades_overstated(self, state):
        state.add_evidence(0, "sample", "ran", True, "checkpoint generated samples")
        wf = PaperWorkflow("text", state)
        draft = wf.assemble_draft()
        findings = wf.review(draft)
        revised = wf.revise(draft, findings)
        assert any("[revision: downgraded to 'ran']" in r for r in revised["results"])


class TestStatePersistence:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "s.json"
        st = ResearchState(p)
        st.add_hypothesis("H1", "h")
        st.add_evidence(0, "t", "ran", True, "ok")
        st.save()
        st2 = ResearchState(p)
        assert st2.data["hypotheses"][0]["id"] == "H1"
        assert st2.data["evidence"][0]["passed"] is True
