import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_doctor_reports_environment_data_and_six_skills() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nanoscigpt.doctor", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "ready"
    assert report["python"]["supported"] is True
    assert set(report["dependencies"]) == {"torch", "numpy", "pandas", "sklearn"}
    assert all(item["available"] for item in report["dependencies"].values())
    assert report["data"]["ready"] == 10
    assert report["data"]["total"] == 10
    assert report["skills"]["ready"] == 6
    assert report["skills"]["total"] == 6


def test_doctor_returns_nonzero_when_a_required_dependency_is_missing() -> None:
    from nanoscigpt.doctor import build_report

    report = build_report(ROOT, required_modules=("module_that_does_not_exist_123",))

    assert report["status"] == "not_ready"
    assert report["dependencies"]["module_that_does_not_exist_123"]["available"] is False


def test_doctor_has_an_installed_console_entry() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'nanoscigpt-doctor = "nanoscigpt.doctor:main"' in pyproject
