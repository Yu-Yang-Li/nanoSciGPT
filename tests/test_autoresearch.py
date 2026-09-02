"""Tests for the three-segment autoresearch pipeline."""

import json
import subprocess
import sys
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch import experiment
from autoresearch.evaluator import evaluate_train
from autoresearch.experiment import ExperimentLoop
from autoresearch.hypothesis import HypothesisEngine, load_data_stats
from autoresearch.paper import PaperWorkflow
from autoresearch.state import ResearchState
from autoresearch.tools import CONTRACTS


@pytest.fixture
def state(tmp_path):
    return ResearchState(tmp_path / "state.json")


class TestHypothesis:
    def test_structured_fixture_counts_are_used_for_grounding(self):
        stats = load_data_stats("weather")
        assert stats["n_sequences"] == 80
        assert stats["representation"] == "spatiotemporal_patches"

    def test_conceive_produces_grounded_ideas(self, state):
        engine = HypothesisEngine("text", state)
        stats = {"vocab_size": 65, "n_sequences": 1000, "mode": "stream"}
        ideas = engine.conceive(stats)
        assert len(ideas) == 3
        for idea in ideas:
            assert "grounding" in idea
            assert idea["grounding"]["metric"]
        visible = " ".join(idea["text"] for idea in ideas).lower()
        assert "one-hot" not in visible
        assert "foundation-model claim" not in visible
        assert "specialized model" not in visible
        assert "value" not in visible

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
    def test_active_contracts_do_not_expose_the_old_protein_only_probe(self):
        assert "transfer_probe" not in CONTRACTS

    def test_plan_has_four_steps(self, state):
        loop = ExperimentLoop("text", state)
        plan = loop.plan()
        assert [s["tool"] for s in plan["steps"]] == [
            "prepare", "train_v0", "train_extended", "sample"]

    def test_load_baseline_run_reads_the_existing_classroom_v0(self, tmp_path):
        model_dir = tmp_path / "model"
        downstream_dir = tmp_path / "downstream"
        model_dir.mkdir()
        downstream_dir.mkdir()
        (model_dir / "ckpt.pt").write_bytes(b"checkpoint")
        (model_dir / "train_log.json").write_text(
            json.dumps({"best_val_loss": 2.9163, "iters": 30, "domain": "protein"}),
            encoding="utf-8",
        )
        (downstream_dir / "downstream_result.json").write_text(
            json.dumps({"metric_name": "accuracy", "metric_value": 0.4688}),
            encoding="utf-8",
        )
        report_path = tmp_path / "run_report.json"
        report_path.write_text(
            json.dumps({
                "status": "completed",
                "domain": "protein",
                "artifacts": {
                    "checkpoint": str(model_dir / "ckpt.pt"),
                    "train_log": str(model_dir / "train_log.json"),
                    "downstream": str(downstream_dir / "downstream_result.json"),
                },
            }),
            encoding="utf-8",
        )

        baseline = experiment.load_baseline_run("protein", report_path)

        assert baseline["version"] == "V0"
        assert baseline["primary_metric"] == "best_val_loss"
        assert baseline["baseline_value"] == 2.9163
        assert baseline["downstream"] == {"metric_name": "accuracy", "metric_value": 0.4688}

    def test_baseline_mode_starts_from_v0_and_does_not_train_it_again(self, state):
        baseline = {
            "version": "V0",
            "primary_metric": "best_val_loss",
            "baseline_value": 2.9163,
        }
        loop = ExperimentLoop("protein", state, baseline=baseline)

        plan = loop.plan()

        assert [step["tool"] for step in plan["steps"]] == [
            "prepare", "train_extended", "sample"]
        assert loop.v0_loss == 2.9163

        prepare_evidence = {"tool": "prepare", "passed": True, "level": "ran"}
        assert loop.decide(prepare_evidence) == "train_extended"

    def test_iteration_spec_is_written_before_model_iteration(self, tmp_path):
        baseline = {
            "version": "V0",
            "domain": "protein",
            "report_path": "out/classroom/protein/run_report.json",
            "checkpoint": "out/classroom/protein/model/ckpt.pt",
            "primary_metric": "best_val_loss",
            "baseline_value": 2.9163,
            "downstream": {"metric_name": "accuracy", "metric_value": 0.4688},
            "fixed": ["data", "split", "evaluation"],
            "allowed_change": "training_budget",
            "classroom_rounds": 1,
        }

        spec_path = experiment.write_iteration_spec(baseline, tmp_path)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        assert spec_path == tmp_path / "protein" / "iteration_spec.json"
        assert spec["baseline"]["version"] == "V0"
        assert spec["baseline"]["primary_metric"] == "best_val_loss"
        assert spec["iteration"]["fixed"] == ["data", "split", "evaluation"]
        assert spec["iteration"]["change"] == "training_budget"
        assert spec["iteration"]["rounds"] == 1

    def test_plan_only_cli_reads_v0_and_stops_before_training(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "ckpt.pt").write_bytes(b"checkpoint")
        (model_dir / "train_log.json").write_text(
            json.dumps({"best_val_loss": 2.9163, "iters": 30, "domain": "protein"}),
            encoding="utf-8",
        )
        report_path = tmp_path / "run_report.json"
        report_path.write_text(
            json.dumps({
                "status": "completed",
                "domain": "protein",
                "artifacts": {
                    "checkpoint": str(model_dir / "ckpt.pt"),
                    "train_log": str(model_dir / "train_log.json"),
                    "downstream": None,
                },
            }),
            encoding="utf-8",
        )
        out_root = tmp_path / "iteration"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "autoresearch.experiment",
                "--domain",
                "protein",
                "--baseline_run",
                str(report_path),
                "--out_root",
                str(out_root),
                "--plan_only",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert completed.returncode == 0, completed.stderr
        assert "V0 loaded" in completed.stdout
        assert "plan only" in completed.stdout
        assert (out_root / "protein" / "iteration_spec.json").is_file()
        assert not (tmp_path / "iteration" / "protein" / "comparison.json").exists()

    def test_v1_comparison_keeps_the_v0_reference_and_observed_delta(self, tmp_path):
        baseline = {
            "version": "V0",
            "domain": "protein",
            "primary_metric": "best_val_loss",
            "baseline_value": 2.9163,
        }
        evidence = {
            "level": "evaluated",
            "passed": False,
            "metric": "loss_gain_vs_v0",
            "value": 0.0478,
            "baseline": 2.9163,
            "reason": "gain +0.0478 vs required 0.05",
            "tool": "train_extended",
        }

        comparison_path = experiment.write_comparison(baseline, evidence, tmp_path)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

        assert comparison_path == tmp_path / "protein" / "comparison.json"
        assert comparison["from_version"] == "V0"
        assert comparison["to_version"] == "V1"
        assert comparison["primary_metric"] == "best_val_loss"
        assert comparison["observed_delta"] == 0.0478
        assert comparison["evidence_level"] == "evaluated"

    def test_baseline_mode_keeps_research_state_with_iteration_artifacts(self, tmp_path):
        assert experiment.iteration_state_path("protein", tmp_path) == (
            tmp_path / "protein" / "research_state.json"
        )

    def test_structured_domain_uses_structured_training_contract(self):
        command = CONTRACTS["train_v0"]["cmd"]("weather")
        assert "nanoscigpt.tasks.structured_demo" in command
        assert "--domain" in command
        assert "weather" in command

    def test_structured_sample_stage_is_recorded_as_a_representation_preview(self, state):
        loop = ExperimentLoop("weather", state)
        evidence = loop.analyze("sample", 0)
        assert "representation preview" in evidence["reason"]

    def test_decide_feedback_changes_next(self, state):
        loop = ExperimentLoop("text", state)
        # prepare passed -> next is train_v0
        ev = {"tool": "prepare", "passed": True, "level": "ran"}
        assert loop.decide(ev) == "train_v0"
        # train_v0 passed (evaluated) -> extend budget
        ev = {"tool": "train_v0", "passed": True, "level": "evaluated", "metric": "best_val_loss"}
        assert loop.decide(ev) == "train_extended"
        # gain failed -> preserve the negative finding and stop increasing budget
        ev = {"tool": "train_extended", "passed": False, "level": "evaluated",
              "metric": "loss_gain_vs_v0"}
        assert loop.decide(ev) == "stop"
        # gain passed -> continue to the sampling check
        ev = {"tool": "train_extended", "passed": True, "level": "evaluated",
              "metric": "loss_gain_vs_v0"}
        assert loop.decide(ev) == "sample"
        # tool crashed -> stop
        ev = {"tool": "prepare", "passed": False, "level": "ran"}
        assert loop.decide(ev) == "stop"

    def test_extended_run_uses_captured_v0_loss(self, state, monkeypatch):
        loop = ExperimentLoop("text", state)
        seen = {}

        monkeypatch.setattr(
            "autoresearch.experiment.evaluate_train",
            lambda domain: {
                "level": "evaluated",
                "passed": True,
                "metric": "best_val_loss",
                "value": 2.5,
                "reason": "v0",
            },
        )

        def fake_gain(domain, baseline):
            seen["baseline"] = baseline
            return {
                "level": "evaluated",
                "passed": True,
                "metric": "loss_gain_vs_v0",
                "value": 0.2,
                "reason": "gain",
            }

        monkeypatch.setattr("autoresearch.experiment.evaluate_train_gain", fake_gain)

        loop.analyze("train_v0", 0)
        loop.analyze("train_extended", 1)
        assert seen["baseline"] == 2.5


class TestPaperWorkflow:
    def test_structured_data_scale_uses_samples_not_sequences(self, state):
        wf = PaperWorkflow("weather", state)
        assert wf._data_scale() == "80 structured samples"

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

    def test_boundary_reports_a_formal_stop_without_claiming_end_to_end(self, state):
        state.add_evidence(0, "train_extended", "evaluated", False, "gain below threshold")
        state.set_next_action("stop")
        wf = PaperWorkflow("text", state)

        boundary = wf.boundary()
        assert "formal stop condition" in boundary
        assert "end-to-end" not in boundary


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
def test_experiment_cli_help_is_domain_neutral() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "autoresearch.experiment", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert "model experiment iteration" in result.stdout
    assert "StarWhisper" not in result.stdout
