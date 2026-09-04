from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nanogpt_pretraining_lesson_is_retained_inside_scientific_language():
    skill_path = ROOT / "skills" / "nanoscigpt-scientific-language" / "references" / "nanogpt-pretraining.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: nanogpt-pretraining" in skill
    assert "python -m nanoscigpt.classroom --domain text" in skill
    assert "run_report.json" in skill
    assert "downstream_result.json" in skill


def test_nanogpt_warmup_is_part_of_the_scientific_language_entrypoint():
    nanogpt = (ROOT / "skills" / "nanoscigpt-scientific-language" / "references" / "nanogpt-pretraining.md").read_text(
        encoding="utf-8"
    )
    nanoscigpt = (
        ROOT / "skills" / "nanoscigpt-scientific-language" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "--domain text" in nanogpt
    assert "--domain text" in nanoscigpt


def test_autoresearch_model_iteration_is_retained_inside_ai_scientist():
    skill_path = ROOT / "skills" / "ai-scientist-research-loop" / "references" / "autoresearch-model-iteration.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: autoresearch-model-iteration" in skill
    assert "--plan_only" in skill
    assert "--baseline_run" in skill
    assert "candidate_run_report.json" in skill
    assert "comparison.json" in skill


def test_ai_scientist_v1_is_retained_inside_ai_scientist():
    skill_path = ROOT / "skills" / "ai-scientist-research-loop" / "references" / "ai-scientist-v1-workflow.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: ai-scientist-v1-workflow" in skill
    assert "--plan-only" in skill
    assert "--confirm-plan" in skill
    assert "evidence_map.json" in skill


def test_ai_scientist_v2_is_retained_inside_ai_scientist():
    skill_path = ROOT / "skills" / "ai-scientist-research-loop" / "references" / "ai-scientist-v2-tree-search.md"

    assert skill_path.is_file()
    skill = skill_path.read_text(encoding="utf-8")
    assert "name: ai-scientist-v2-tree-search" in skill
    assert "run-next" in skill
    assert "tree_state.json" in skill
    assert "reproduces_original_system" in skill
