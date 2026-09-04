from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_instructor_guides_have_one_canonical_location() -> None:
    instructor = ROOT / "docs" / "instructor"
    assert (instructor / "README.md").is_file()
    assert (instructor / "teaching-guide.md").is_file()
    assert (instructor / "ai-scientist-guide.md").is_file()

    assert "docs/instructor/teaching-guide.md" in (
        ROOT / "docs" / "teaching-guide.md"
    ).read_text(encoding="utf-8")
    assert "docs/instructor/ai-scientist-guide.md" in (
        ROOT / "docs" / "ai-scientist-guide.md"
    ).read_text(encoding="utf-8")
