from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (
    "text",
    "protein",
    "dna",
    "smiles",
    "weather",
    "crystal",
    "structure3d",
    "image",
    "spectrum",
    "field",
)


def test_registry_exports_the_ten_classroom_domains_once():
    from nanoscigpt.domains.registry import DOMAIN_SPECS, RUNNABLE_DOMAINS

    assert RUNNABLE_DOMAINS == EXPECTED
    assert tuple(spec.name for spec in DOMAIN_SPECS) == EXPECTED
    assert len(set(RUNNABLE_DOMAINS)) == len(RUNNABLE_DOMAINS)
    assert all(spec.representation for spec in DOMAIN_SPECS)
    assert all(spec.task_name for spec in DOMAIN_SPECS)


def test_registry_rejects_duplicate_names():
    from nanoscigpt.domains.registry import DOMAIN_SPECS, build_registry

    duplicate = replace(DOMAIN_SPECS[0], task_name="a second text task")
    with pytest.raises(ValueError, match="duplicate domain"):
        build_registry((*DOMAIN_SPECS, duplicate))


def test_runtime_modules_share_the_registry_domain_groups():
    from autoresearch import evaluator, experiment, tools
    from nanoscigpt import classroom
    from nanoscigpt.domains.registry import RUNNABLE_DOMAINS, STRUCTURED_DOMAINS
    from nanoscigpt.tasks import structured_demo

    assert classroom.RUNNABLE_DOMAINS is RUNNABLE_DOMAINS
    assert experiment.RUNNABLE_DOMAINS is RUNNABLE_DOMAINS
    assert experiment.STRUCTURED_DOMAINS is STRUCTURED_DOMAINS
    assert tools.STRUCTURED_DOMAINS is STRUCTURED_DOMAINS
    assert structured_demo.STRUCTURED_DOMAINS is STRUCTURED_DOMAINS
    assert evaluator.is_structured_domain("weather") is True
    assert evaluator.is_structured_domain("protein") is False


def test_custom_domain_guide_names_every_required_contract():
    guide = (ROOT / "docs" / "current" / "custom-domain-guide.md").read_text(
        encoding="utf-8"
    )
    example = (ROOT / "examples" / "custom_domain" / "README.md").read_text(
        encoding="utf-8"
    )
    for phrase in ("数据读取", "科学表示", "预训练目标", "下游任务", "评价器"):
        assert phrase in guide
        assert phrase in example
    assert "不承诺" in guide
