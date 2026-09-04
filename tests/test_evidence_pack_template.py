from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_student_evidence_pack_template_has_the_required_sections() -> None:
    template = (ROOT / "docs" / "evidence-pack-template.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. 科学问题",
        "## 2. 输入与数据",
        "## 3. 模型或工作流设置",
        "## 4. Agent Prompt 与完整轨迹",
        "## 5. 运行与评价结果",
        "## 6. 失败、反证或停止",
        "## 7. 根据反馈修改的下一步",
    ):
        assert heading in template
    for marker in ("设计", "已运行", "已比较", "已评测", "已外部验证"):
        assert marker in template
    assert "不把教学数据结果外推成科学结论" in template
