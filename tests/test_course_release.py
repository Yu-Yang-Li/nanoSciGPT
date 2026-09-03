import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    ROOT / "docs" / "current" / "course-outline-16p.md",
    ROOT / "docs" / "current" / "slide-copy-16p.md",
    ROOT / "docs" / "current" / "speaker-script-16p.md",
)
EXPECTED_SKILLS = {
    "nanoscigpt-research-baseline-builder",
    "nanogpt-pretraining",
    "nanoscigpt-scientific-language",
    "autoresearch-model-iteration",
    "ai-scientist-v1-workflow",
    "ai-scientist-v2-tree-search",
}


def _pages(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"^### P(\d+)\b", text, re.MULTILINE)]


def test_current_course_release_has_one_complete_sixteen_page_source() -> None:
    for path in CURRENT_DOCS:
        assert path.is_file(), path
        assert _pages(path.read_text(encoding="utf-8")) == list(range(1, 17))


def test_outline_time_windows_are_contiguous_from_zero_to_ninety() -> None:
    text = CURRENT_DOCS[0].read_text(encoding="utf-8")
    windows = [
        (int(start), int(end))
        for start, end in re.findall(
            r"^### P\d+｜(\d+)[—-](\d+)分钟", text, re.MULTILINE
        )
    ]

    assert len(windows) == 16
    assert windows[0][0] == 0
    assert windows[-1][1] == 90
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))


def test_repository_publishes_exactly_six_atomic_skills() -> None:
    actual = {path.name for path in (ROOT / "skills").iterdir() if path.is_dir()}

    assert actual == EXPECTED_SKILLS
    assert all((ROOT / "skills" / name / "SKILL.md").is_file() for name in actual)


def test_readme_points_to_current_course_and_six_skill_index() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for path in (
        "docs/current/course-outline-16p.md",
        "docs/current/slide-copy-16p.md",
        "docs/current/speaker-script-16p.md",
        "skills/README.md",
    ):
        assert path in readme
    for stale in ("四个领域的最小可运行链路", "三段式虚拟 AI Scientist", "100 iter"):
        assert stale not in readme
    assert "nanoscigpt-doctor" in readme
    assert "scripts/install_skills.ps1" in readme
    assert "scripts/install_skills.sh" in readme


def test_evidence_pack_has_a_console_entry() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'nanoscigpt-evidence-pack = "nanoscigpt.evidence_pack:main"' in pyproject
    assert (ROOT / "nanoscigpt" / "evidence_pack.py").is_file()


def test_student_navigation_has_no_broken_local_markdown_links() -> None:
    for source in (ROOT / "README.md", ROOT / "docs" / "README.md", ROOT / "skills" / "README.md"):
        text = source.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (source.parent / target).resolve()
            assert resolved.exists(), f"{source.relative_to(ROOT)} -> {target}"


def test_course_ci_covers_both_platforms_and_real_smoke_runs() -> None:
    workflow = ROOT / ".github" / "workflows" / "test-course.yml"

    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    for expected in (
        "ubuntu-latest",
        "windows-latest",
        "python -m nanoscigpt.doctor",
        "--domain all --profile smoke",
        "tests/test_classroom.py",
    ):
        assert expected in text
