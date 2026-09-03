from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "docs" / "current"


def page(text: str, number: int) -> str:
    start = text.index(f"### P{number}｜")
    marker = f"### P{number + 1}｜"
    end = text.find(marker, start)
    return text[start:] if end == -1 else text[start:end]


def test_p2_has_one_practice_prompt_and_p11_to_p14_have_none():
    copy = (CURRENT / "slide-copy-16p.md").read_text(encoding="utf-8")

    assert page(copy, 2).count("**课堂实践**") == 1
    for number in (11, 12, 13, 14):
        assert "**课堂实践**" not in page(copy, number)


def test_ai_scientist_history_cases_match_across_current_materials():
    expected = {
        12: (
            "Robot Scientist",
            "Adam",
            "Eve",
            "ARES",
            "Ada",
            "Mobile Robotic Chemist",
        ),
        13: ("Coscientist", "小来", "ChemCrow", "SAMPLE"),
        14: ("StarWhisper", "Virtual Lab", "Co-Scientist", "Robin"),
    }
    paths = (
        CURRENT / "course-outline-16p.md",
        CURRENT / "slide-copy-16p.md",
        CURRENT / "speaker-script-16p.md",
        CURRENT / "ppt-skill-teaching-map-16p.md",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for number, names in expected.items():
            section = page(text, number)
            for name in names:
                assert name in section, f"{path.name} P{number} missing {name}"


def test_course_copy_does_not_misstate_the_runnable_text_or_rfdiffusion_tasks():
    paths = (
        CURRENT / "course-outline-16p.md",
        CURRENT / "slide-copy-16p.md",
        CURRENT / "speaker-script-16p.md",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "词语接龙\"变成\"问答对话" not in text
        assert "词语接龙->问答对话" not in text
        assert "标点密度" in page(text, 3)
        assert "蛋白质骨架" in page(text, 5)
