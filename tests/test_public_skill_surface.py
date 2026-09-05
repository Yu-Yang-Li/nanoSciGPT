from pathlib import Path
import tomllib


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


def test_console_scripts_do_not_impersonate_upstream_research_projects() -> None:
    project = tomllib.loads(_read("pyproject.toml"))["project"]
    scripts = project["scripts"]
    assert "nanoscigpt-autoresearch" not in scripts
    assert "nanoscigpt-ai-scientist-v1" not in scripts
    assert "nanoscigpt-ai-scientist-v2" not in scripts


def test_current_navigation_separates_upstream_workflows_from_legacy_lessons() -> None:
    skills_index = _read("skills/README.md")
    instructor_index = _read("docs/instructor/README.md")
    legacy_guides = (
        _read("docs/instructor/teaching-guide.md"),
        _read("docs/instructor/ai-scientist-guide.md"),
    )

    assert "karpathy/autoresearch" in skills_index
    assert "SakanaAI/AI-Scientist" in skills_index
    assert "完整原版研究流程尚未验收" in skills_index
    assert "skills/ai-scientist-research-loop/SKILL.md" in instructor_index
    assert "native-projects.md" in instructor_index
    assert "training-and-native-status-2026-09-05.md" in instructor_index
    for guide in legacy_guides:
        assert "历史" in guide[:500]
        assert "skills/ai-scientist-research-loop/SKILL.md" in guide[:1000]
