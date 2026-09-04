from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_students_see_exactly_three_skill_entrypoints() -> None:
    public = sorted(path.parent.name for path in SKILLS.glob("*/SKILL.md"))
    assert public == [
        "ai-scientist-research-loop",
        "nanoscigpt-scientific-language",
        "research-baseline-builder",
    ]


def test_scientific_language_entrypoint_contains_text_warmup_and_all_domains() -> None:
    skill = _read("skills/nanoscigpt-scientific-language/SKILL.md")
    assert "python -m nanoscigpt.classroom --domain text" in skill
    for domain in (
        "protein",
        "dna",
        "smiles",
        "weather",
        "crystal",
        "structure3d",
        "image",
        "spectrum",
        "field",
    ):
        assert domain in skill
    assert "nanogpt-pretraining" not in skill


def test_scientific_language_entrypoint_marks_precomputed_results_as_fallback() -> None:
    skill = _read("skills/nanoscigpt-scientific-language/SKILL.md")
    assert "data/precomputed_results" in skill
    assert "不能说成本次运行结果" in skill


def test_ai_scientist_entrypoint_contains_the_three_progressive_stages() -> None:
    skill = _read("skills/ai-scientist-research-loop/SKILL.md")
    for marker in (
        "autoresearch.experiment",
        "autoresearch.v1",
        "autoresearch.v2",
        "comparison.json",
        "research_state.json",
        "tree_state.json",
    ):
        assert marker in skill
    assert "一次只" in skill


def test_legacy_lessons_are_retained_as_references_not_public_entrypoints() -> None:
    scientific_references = SKILLS / "nanoscigpt-scientific-language" / "references"
    ai_scientist_references = SKILLS / "ai-scientist-research-loop" / "references"
    assert (scientific_references / "nanogpt-pretraining.md").is_file()
    for filename in (
        "autoresearch-model-iteration.md",
        "ai-scientist-v1-workflow.md",
        "ai-scientist-v2-tree-search.md",
    ):
        assert (ai_scientist_references / filename).is_file()
