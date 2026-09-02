from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
