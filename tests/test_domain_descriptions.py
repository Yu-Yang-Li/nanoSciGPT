import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_every_domain_declares_the_teaching_semantics_used_by_the_code():
    from nanoscigpt.domains.registry import DOMAIN_SPECS

    assert len(DOMAIN_SPECS) == 10
    for spec in DOMAIN_SPECS:
        assert spec.model_unit
        assert spec.preserved_relations
        assert spec.pretraining_objective


def test_structured_descriptions_match_masked_reconstruction_implementation():
    from nanoscigpt.domains.registry import get_domain_spec

    weather = get_domain_spec("weather")
    spectrum = get_domain_spec("spectrum")
    crystal = get_domain_spec("crystal")

    assert weather.pretraining_objective == "重建被遮住的时空网格块"
    assert spectrum.pretraining_objective == "重建被遮住的连续波长区间"
    assert crystal.pretraining_objective == "判断被遮住位置的原子种类"


def test_domain_cards_distinguish_fine_tuning_from_frozen_probes():
    from nanoscigpt.classroom import describe_domain

    text = describe_domain("text", DATA)
    protein = describe_domain("protein", DATA)
    spectrum = describe_domain("spectrum", DATA)

    assert text["downstream_training"] == "full_fine_tune"
    assert protein["downstream_training"] == "frozen_probe"
    assert spectrum["downstream_training"] == "frozen_probe"


def test_describe_domain_distinguishes_bundled_example_from_student_data():
    from nanoscigpt.classroom import describe_domain

    card = describe_domain("spectrum", DATA)

    assert card["status"] == "ready"
    assert card["domain"] == "spectrum"
    assert card["source_kind"] == "synthetic_fixture"
    assert "deterministic" in card["source_name"]
    assert card["student_data_loaded"] is False
    assert card["support_level"] == "bundled_example_only"
    assert card["pretraining_objective"] == "重建被遮住的连续波长区间"


def test_classroom_describe_cli_returns_one_machine_readable_card():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.classroom",
            "--describe",
            "weather",
            "--data_root",
            str(DATA),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    card = json.loads(completed.stdout)
    assert card["domain"] == "weather"
    assert card["pretraining_objective"] == "重建被遮住的时空网格块"
    assert card["student_data_loaded"] is False
    assert "classroom run" not in completed.stdout.lower()
