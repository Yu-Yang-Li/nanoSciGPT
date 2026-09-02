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
