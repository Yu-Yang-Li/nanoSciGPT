import json
import subprocess
import sys

import pytest

from scripts import capture_classroom_dialogue as capture


def test_timeout_preserves_partial_dialogue_and_does_not_resume(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(["discuss only", "continue"]), encoding="utf-8")
    output = tmp_path / "capture"
    monkeypatch.setattr(sys, "argv", ["capture", "--inputs", str(inputs), "--output", str(output)])
    monkeypatch.setattr(capture.shutil, "which", lambda _: "codex")
    calls = []

    def timeout_fixture(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(json.dumps({"type": "thread.started", "thread_id": "timeout-fixture"}) + "\n")
        kwargs["stderr"].write("partial diagnostic")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(capture.subprocess, "run", timeout_fixture)
    assert capture.main() == 1
    state = json.loads((output / "session.json").read_text())
    assert state["completed_turns"] == 0
    assert state["attempted_turns"] == 1
    assert state["stop_reason"] == "cli_timeout"
    assert state["classroom_execution_passed"] is False
    assert "partial diagnostic" in (output / "turn-1-stderr.txt").read_text()
    assert "timeout-fixture" in (output / "turn-1.jsonl").read_text()
    assert (output / "dialogue.md").exists()
    assert len(calls) == 1


@pytest.mark.parametrize("sandbox_failure", [True, False])
def test_capture_records_attempted_vs_completed_and_returns_failure(tmp_path, monkeypatch, sandbox_failure):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(["I choose protein", "continue"]), encoding="utf-8")
    output = tmp_path / "capture"
    monkeypatch.setattr(sys, "argv", ["capture", "--inputs", str(inputs), "--output", str(output)])
    monkeypatch.setattr(capture.shutil, "which", lambda _: "codex")
    calls = []

    def fake_cli(command, **kwargs):
        # Unit-test event fixture, not an actual model reply or teaching run.
        calls.append(command)
        kwargs["stdout"].write(json.dumps({"type": "thread.started", "thread_id": "unit-test-thread"}) + "\n")
        kwargs["stderr"].write("apply deny-read ACLs" if sandbox_failure else "")
        from pathlib import Path
        Path(command[command.index("-o") + 1]).write_text("unit-test reply", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(capture.subprocess, "run", fake_cli)
    result = capture.main()
    state = json.loads((output / "session.json").read_text())
    assert result == (1 if sandbox_failure else 0)
    assert state["attempted_turns"] == (1 if sandbox_failure else 2)
    assert state["completed_turns"] == (0 if sandbox_failure else 2)
    assert state["classroom_execution_passed"] is False
    assert len(calls) == (1 if sandbox_failure else 2)


@pytest.mark.parametrize("model", [None, "scnet/GLM-5.3"])
def test_capture_preserves_default_model_and_explicit_sandbox_on_resume(tmp_path, monkeypatch, model):
    from pathlib import Path

    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(["choose protein", "continue"]), encoding="utf-8")
    output = tmp_path / "capture"
    arguments = ["capture", "--inputs", str(inputs), "--output", str(output)]
    if model:
        arguments.extend(["--model", model])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(capture.shutil, "which", lambda _: "codex")
    invocations = []

    def cli_event_fixture(command, **kwargs):
        # The process boundary is mocked; model behavior is not under test.
        invocations.append(command)
        kwargs["stdout"].write(json.dumps({"type": "thread.started", "thread_id": "test-thread"}) + "\n")
        Path(command[command.index("-o") + 1]).write_text("fixture reply", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(capture.subprocess, "run", cli_event_fixture)
    assert capture.main() == 0
    assert "resume" in invocations[1]
    for command in invocations:
        if model is None:
            assert "--model" not in command
        else:
            assert command[command.index("--model") + 1] == model
        overrides = [command[i + 1] for i, token in enumerate(command) if token == "-c"]
        assert 'sandbox_mode="workspace-write"' in overrides
        assert 'approval_policy="never"' in overrides
        assert 'features.memories=false' in overrides
