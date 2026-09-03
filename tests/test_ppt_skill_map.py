import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs" / "current" / "ppt-skill-teaching-map-16p.md"


def test_ppt_skill_map_covers_every_page_once():
    text = MAP.read_text(encoding="utf-8")
    pages = [int(value) for value in re.findall(r"^### P(\d+)\b", text, re.MULTILINE)]

    assert pages == list(range(1, 17))


def test_ppt_skill_map_names_the_real_student_commands_and_outputs():
    text = MAP.read_text(encoding="utf-8")

    for phrase in (
        "nanoscigpt-baseline --case lamost",
        "nanoscigpt.classroom --describe",
        "nanoscigpt.classroom --domain text",
        "autoresearch.experiment",
        "autoresearch.v1",
        "autoresearch.v2",
        "nanoscigpt.evidence_pack",
        "run_report.json",
        "comparison.json",
        "workflow_state.json",
    ):
        assert phrase in text


def test_history_pages_do_not_interrupt_the_timeline_with_new_commands():
    text = MAP.read_text(encoding="utf-8")
    for page in (5, 6, 7, 8, 12, 13, 14):
        section = re.search(
            rf"^### P{page}\b(?P<body>.*?)(?=^### P\d+\b|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        ).group("body")
        assert "不新增命令" in section


def test_student_utterances_are_natural_not_placeholder_prompts():
    text = MAP.read_text(encoding="utf-8")

    assert "基于[课程仓库中的Skill]" not in text
    assert "提供的skill地址" not in text
    assert "你可以这样说" in text
