import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIALOGUE_DIR = ROOT / "docs" / "acceptance" / "cli-dialogues"


def _events(log_path: Path) -> list[dict]:
    events = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # The CLI preserves human-readable startup and transport diagnostics.
            continue
    return events


def test_cli_dialogue_index_preserves_exact_inputs_and_outputs():
    index_path = DIALOGUE_DIR / "scenarios.json"
    scenarios = json.loads(index_path.read_text(encoding="utf-8"))

    expected_ids = {
        "baseline-course",
        "baseline-own",
        "baseline-private",
        "science-course",
        "science-own",
        "science-private",
        "ai-course",
        "ai-own",
        "ai-private",
        "glm53-science-course",
    }
    assert {scenario["id"] for scenario in scenarios} == expected_ids

    default_scenarios = [scenario for scenario in scenarios if scenario["model"] == "gpt-5.6-sol"]
    assert len(default_scenarios) == 9
    assert {scenario["skill"] for scenario in default_scenarios} == {
        "research-baseline-builder",
        "nanoscigpt-scientific-language",
        "ai-scientist-research-loop",
    }

    for scenario in scenarios:
        assert scenario["student_input"].strip()
        assert scenario["actual_output"].strip()
        log_path = DIALOGUE_DIR / scenario["log"]
        assert log_path.is_file()

        events = _events(log_path)
        assert any(event.get("type") == "thread.started" for event in events)
        assert any(event.get("type") == "turn.completed" for event in events)

        final_messages = [
            event["item"]["text"]
            for event in events
            if event.get("type") == "item.completed"
            and event.get("item", {}).get("type") == "agent_message"
        ]
        assert final_messages
        assert scenario["actual_output"] == final_messages[-1]
