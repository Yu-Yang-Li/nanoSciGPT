from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_skill_is_namespaced_and_states_its_runnable_boundary():
    skill_path = (
        ROOT / "skills" / "nanoscigpt-research-baseline-builder" / "SKILL.md"
    )

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: nanoscigpt-research-baseline-builder" in skill
    assert "python -m nanoscigpt.baseline --case lamost" in skill
    assert "FITS" in skill
    assert "不能直接运行" in skill


def test_nanogpt_pretraining_skill_is_packaged_as_its_own_lesson():
    skill_path = ROOT / "skills" / "nanogpt-pretraining" / "SKILL.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: nanogpt-pretraining" in skill
    assert "python -m nanoscigpt.classroom --domain text" in skill
    assert "run_report.json" in skill
    assert "downstream_result.json" in skill


def test_nanogpt_and_nanoscigpt_skills_have_distinct_triggers():
    nanogpt = (ROOT / "skills" / "nanogpt-pretraining" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    nanoscigpt = (
        ROOT / "skills" / "nanoscigpt-scientific-language" / "SKILL.md"
    ).read_text(encoding="utf-8")

    nanogpt_description = nanogpt.split("---", 2)[1]
    nanoscigpt_description = nanoscigpt.split("---", 2)[1]
    assert "text" in nanogpt_description.lower()
    assert "text" not in nanoscigpt_description.lower()
    assert "fine-tuning" not in nanoscigpt_description.lower()


def test_autoresearch_model_iteration_is_packaged_as_its_own_lesson():
    skill_path = ROOT / "skills" / "autoresearch-model-iteration" / "SKILL.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: autoresearch-model-iteration" in skill
    assert "--plan_only" in skill
    assert "--baseline_run" in skill
    assert "candidate_run_report.json" in skill
    assert "comparison.json" in skill


def test_ai_scientist_v1_is_packaged_as_its_own_lesson():
    skill_path = ROOT / "skills" / "ai-scientist-v1-workflow" / "SKILL.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: ai-scientist-v1-workflow" in skill
    assert "--plan-only" in skill
    assert "--confirm-plan" in skill
    assert "evidence_map.json" in skill


def test_ai_scientist_v2_is_packaged_as_its_own_lesson():
    skill_path = ROOT / "skills" / "ai-scientist-v2-tree-search" / "SKILL.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: ai-scientist-v2-tree-search" in skill
    assert "run-next" in skill
    assert "tree_state.json" in skill
    assert "reproduces_original_system" in skill
