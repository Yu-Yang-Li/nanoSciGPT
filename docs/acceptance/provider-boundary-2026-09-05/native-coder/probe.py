"""Single native Aider turn, stopped for human/code review before execution."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
source = root / "out/upstream/v1/templates/student_protein_task_roundtrip_v2"
output = root / "out/native-coder-bounded-20260905/run"
output.mkdir(exist_ok=False)
names = ("experiment.py", "plot.py", "native_gpt.py", "task_data.npz", "task_setup.json", "initial_model.pt")
hashes = {}
for name in names:
    shutil.copyfile(source / name, output / name)
    hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
with (output / "baseline-stdout.txt").open("w", encoding="utf-8") as stdout, (output / "baseline-stderr.txt").open("w", encoding="utf-8") as stderr:
    subprocess.run([sys.executable, "-X", "utf8", "experiment.py", "--out_dir=run_0"], cwd=output,
                   stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, timeout=30, check=True)
idea = json.loads((root / "out/native-api-20260905/ideation/ideas.json").read_text())[-1]
baseline = json.loads((output / "run_0/final_info.json").read_text())
baseline = {key: value["means"] for key, value in baseline.items()}
(output / "notes.txt").write_text(f"# {idea['Title']}\n\nBaseline run_0 (already executed): {baseline}\n", encoding="utf-8")
sys.path.insert(0, str(root / "out/upstream/v1"))
from ai_scientist.perform_experiments import coder_prompt
prompt = coder_prompt.format(title=idea["Title"], idea=idea["Experiment"], max_runs=3, baseline_results=baseline)
(output / "input.txt").write_text(prompt, encoding="utf-8")
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:10100/v1"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:10100/v1"
os.environ["OPENAI_API_KEY"] = "local-no-auth-probe"
os.environ["AIDER_ANALYTICS"] = "false"
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from aider.coders import base_coder
base_coder.RETRY_TIMEOUT = 0  # Teaching probe: no outer transport retries.

record = {"status": "started", "model": "openai/scnet/GLM-5.3", "data_identity": "handwritten teaching protein labels",
          "input_sha256": hashes, "source_commit": "1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb",
          "experiments_run": False, "scope": "original coder prompt and Aider diff coder, one turn only",
          "baseline_run": True, "max_tokens": 20000, "api_timeout_seconds": 240,
          "requested_thinking": "omitted_provider_does_not_accept_disabled",
          "outer_transport_retries": 0, "max_edit_reflections": 1}
(output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
start = time.monotonic()
try:
    os.chdir(output)
    model = Model("openai/scnet/GLM-5.3", weak_model="openai/scnet/GLM-5.3")
    model.extra_params = {"max_tokens": 20000, "timeout": 240, "num_retries": 0}
    original_send = model.send_completion
    requests = []
    def capture_completion(messages, functions, stream, temperature=None):
        if requests:
            raise RuntimeError("One API request budget reached; preserve the first result for review")
        requests.append({"model": model.name, "messages": messages, "functions": functions, "stream": stream, "temperature": temperature, "extra_params": model.extra_params})
        (output / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
        response_hash, response = original_send(messages, functions, stream, temperature)
        (output / "response.json").write_text(json.dumps(response.model_dump(), indent=2), encoding="utf-8")
        return response_hash, response
    model.send_completion = capture_completion
    io = InputOutput(yes=True, pretty=False, chat_history_file=str(output / "chat.txt"))
    coder = Coder.create(main_model=model, fnames=[str(output / n) for n in ("experiment.py", "plot.py", "notes.txt")],
                         read_only_fnames=[str(output / "task_setup.json")], io=io, stream=False, use_git=False,
                         edit_format="diff", auto_commits=False, auto_lint=False, auto_test=False,
                         suggest_shell_commands=False, detect_urls=False, map_tokens=0)
    coder.max_reflections = 1
    answer = coder.run(prompt)
    (output / "answer.txt").write_text(answer or "", encoding="utf-8")
    (output / "messages.json").write_text(json.dumps({"done": coder.done_messages, "current": coder.cur_messages}, indent=2), encoding="utf-8")
    record.update(status="returned_pending_code_review", tokens_sent=coder.total_tokens_sent, tokens_received=coder.total_tokens_received)
except Exception as exc:
    record.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500])
finally:
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    record["output_sha256"] = {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in names}
    record["fixed_inputs_unchanged"] = all(record["output_sha256"][n] == hashes[n] for n in names[2:])
    patch = ""
    for n in ("experiment.py", "plot.py"):
        patch += "".join(difflib.unified_diff((source / n).read_text().splitlines(True), (output / n).read_text().splitlines(True), fromfile=f"before/{n}", tofile=f"after/{n}"))
    (output / "changes.diff").write_text(patch, encoding="utf-8")
    if record["status"] == "returned_pending_code_review" and (not patch or not answer):
        record["status"] = "failed_no_implemented_change"
    (output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
sys.exit(0 if record["status"] == "returned_pending_code_review" and record["fixed_inputs_unchanged"] else 1)
