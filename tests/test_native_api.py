"""Exercise the adapter against pinned original function bodies, without cloud calls."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanoscigpt import upstream


@pytest.fixture
def original(tmp_path, monkeypatch):
    cached = Path(__file__).resolve().parents[1] / "out/upstream/v1"
    if not (cached / ".git").exists():
        pytest.skip("prepare pinned v1 to test its real routing code")
    target = tmp_path / "v1"
    (target / "ai_scientist").mkdir(parents=True)
    for name in ("ai_scientist/llm.py", "launch_scientist.py"):
        source = upstream.git(cached, "show", f"{upstream.PROJECTS['v1'][1]}:{name}")
        (target / name).write_text(source + "\n", encoding="utf-8")
    (target / "experiment.py").write_text("# student's experiment stays untouched\n")
    upstream.git(target, "init")
    upstream.git(target, "add", ".")
    upstream.git(target, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    pin = upstream.git(target, "rev-parse", "HEAD")
    monkeypatch.setitem(upstream.PROJECTS, "v1", ("fixture", pin))
    return target


def configure(path, model="scnet/GLM-5.3", reviewer="gpt-6-astra"):
    adapter = getattr(upstream, "configure_v1_api", None)
    assert callable(adapter), "v1 has no explicit compatible-API teaching adapter"
    return adapter(path, model, reviewer)


def llm_namespace(path):
    # Keep original function bodies; omit optional provider imports and retry decorators.
    tree = ast.parse((path / "ai_scientist/llm.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))
                                        for _ in range(kwargs.get("n", 1))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    namespace = {"openai": SimpleNamespace(OpenAI=lambda: client)}
    exec(compile(tree, "original_llm.py", "exec"), namespace)
    return namespace, calls


@pytest.mark.parametrize("model", ["scnet/GLM-5.3", "gpt-6-astra", "claude-compatible-local", "llama-3-1-405b-instruct"])
def test_explicit_route_preserves_actual_model_name_in_single_and_batch_requests(original, model):
    configure(original, model)
    namespace, calls = llm_namespace(original)
    client, selected = namespace["create_client"](model)
    assert selected == model
    assert model in namespace["AVAILABLE_LLMS"]
    answer, history = namespace["get_response_from_llm"]("task", client, selected, "system")
    answers, histories = namespace["get_batch_responses_from_llm"]("review", client, selected, "system", n_responses=2)
    assert answer == "answer" and answers == ["answer", "answer"]
    expected_counts = [1, 2] if "gpt" in model else [1, 1, 1]
    assert [call["model"] for call in calls] == [model] * len(expected_counts)
    assert calls[0]["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    # Preserve upstream's sequential fallback for non-GPT providers, which may reject n>1.
    assert [call["n"] for call in calls] == expected_counts
    assert history[-1]["content"] == "answer" and len(histories) == 2


def test_coder_and_both_reviews_follow_the_explicit_models(original):
    configure(original)
    namespace, _ = llm_namespace(original)
    tree = ast.parse((original / "launch_scientist.py").read_text(encoding="utf-8"))
    # Execute the real selection blocks and review calls, replacing only external clients.
    namespace.update(model="scnet/GLM-5.3", Model=lambda *a, **k: (a, k), paper_text="test paper",
                     perform_review=lambda *a, **k: k)
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and any(isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "main_model" for t in stmt.targets)
                      for stmt in node.body)]
    # Only outermost model-selection blocks, not their elif children.
    blocks = [node for node in blocks if not any(node in other.orelse for other in blocks)]
    assert len(blocks) == 2
    for block in blocks:
        exec(compile(ast.Module(body=[block], type_ignores=[]), "select.py", "exec"), namespace)
        args, kwargs = namespace["main_model"]
        assert args == ("openai/scnet/GLM-5.3",)
        assert kwargs["weak_model"] == "openai/scnet/GLM-5.3"
    reviews = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == "perform_review"]
    assert len(reviews) == 2
    for call in reviews:
        result = eval(compile(ast.Expression(call), "review.py", "eval"), namespace)
        assert result["model"] == "gpt-6-astra"


def test_dirty_source_is_rejected_before_any_other_file_is_changed(original):
    launcher = original / "launch_scientist.py"
    launcher.write_text(launcher.read_text(encoding="utf-8") + "\n# student edit\n", encoding="utf-8")
    before = {name: (original / name).read_bytes() for name in ("ai_scientist/llm.py", "launch_scientist.py", "experiment.py")}
    with pytest.raises(ValueError, match="local changes"):
        configure(original)
    assert before == {name: (original / name).read_bytes() for name in before}


def test_repeat_keeps_identical_setup_but_rejects_changed_selection_or_files(original):
    first = configure(original)
    assert configure(original) == first
    assert first["status"] == "configured_not_run"
    assert (original / "experiment.py").read_text() == "# student's experiment stays untouched\n"
    assert (original / "teaching_api_changes.diff").stat().st_size > 100
    with pytest.raises(ValueError, match="another"):
        configure(original, "different-model")
    (original / "ai_scientist/llm.py").write_text("# changed after configuration\n")
    with pytest.raises(ValueError, match="changed"):
        configure(original)


def test_cli_configures_the_selected_checkout_without_starting_research(original, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["upstream", "configure-api", "v1", "--root", str(original.parent),
                                      "--model", "scnet/GLM-5.3", "--review-model", "gpt-6-astra"])
    upstream.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "configured_not_run"
    assert receipt["model"] == "scnet/GLM-5.3"
    assert not (original / "results").exists()


def test_preexisting_patch_is_not_overwritten(original):
    patch = original / "teaching_api_changes.diff"
    patch.write_text("previous work\n")
    before = (original / "ai_scientist/llm.py").read_bytes()
    with pytest.raises(ValueError, match="existing"):
        configure(original)
    assert patch.read_text() == "previous work\n"
    assert (original / "ai_scientist/llm.py").read_bytes() == before
