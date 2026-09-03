import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "research-baseline-builder",
    "nanogpt-pretraining",
    "nanoscigpt-scientific-language",
    "autoresearch-model-iteration",
    "ai-scientist-v1-workflow",
    "ai-scientist-v2-tree-search",
}


def _installed(destination: Path) -> set[str]:
    return {path.name for path in destination.iterdir() if path.is_dir()}


def test_powershell_installer_copies_exactly_six_skills_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "codex-skills"
    command = [
        "pwsh",
        "-NoProfile",
        "-File",
        str(ROOT / "scripts" / "install_skills.ps1"),
        "-Destination",
        str(destination),
    ]

    first = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
    )
    second = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
    )

    assert first.returncode == 0, first.stdout + first.stderr
    assert _installed(destination) == EXPECTED
    assert all((destination / name / "SKILL.md").is_file() for name in EXPECTED)
    assert second.returncode != 0
    assert "already exists" in (second.stdout + second.stderr)


@pytest.mark.skipif(sys.platform == "win32", reason="PowerShell is the supported Windows installer")
def test_bash_installer_copies_exactly_six_skills_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "codex-skills"
    command = [
        "bash",
        str(ROOT / "scripts" / "install_skills.sh"),
        str(destination),
    ]

    first = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
    )
    second = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
    )

    assert first.returncode == 0, first.stdout + first.stderr
    assert _installed(destination) == EXPECTED
    assert second.returncode != 0
    assert "already exists" in (second.stdout + second.stderr)
