"""Run original v1 ideation/reflection on the existing teaching task only."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import httpx
import openai

root = Path(__file__).resolve().parents[2]
source = root / "out/upstream/v1/templates/student_protein_task_roundtrip_v2"
output = root / "out/native-api-20260905/ideation"
output.mkdir(exist_ok=False)
hashes = {}
for name in ("experiment.py", "native_gpt.py", "plot.py", "prompt.json", "seed_ideas.json", "task_setup.json"):
    shutil.copyfile(source / name, output / name)
    hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
sys.path.insert(0, str(root / "out/upstream/v1"))
from ai_scientist.generate_ideas import generate_ideas

requests = []
responses = []
def request_hook(request):
    requests.append(json.loads(request.content))
    (output / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")

def response_hook(response):
    response.read()
    responses.append({"status": response.status_code, "body": response.json()})
    (output / "responses.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")

record = {"evidence_type": "original_v1_ideation_only_not_full_research", "model": "gpt-6-astra",
          "max_num_generations": 1, "num_reflections": 2, "source_files_sha256": hashes,
          "data_identity": "handwritten teaching protein CSV, not experimental activity measurements",
          "status": "started", "experiments_run": False, "novelty_search_run": False}
(output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
start = time.monotonic()
try:
    transport = httpx.Client(timeout=60, event_hooks={"request": [request_hook], "response": [response_hook]})
    with openai.OpenAI(base_url="http://127.0.0.1:10100/v1", api_key="local-no-auth-probe",
                       max_retries=0, timeout=60, http_client=transport) as client:
        ideas = generate_ideas(str(output), client, "gpt-6-astra", max_num_generations=1, num_reflections=2)
    seed_count = len(json.loads((output / "seed_ideas.json").read_text(encoding="utf-8")))
    record.update(seed_ideas=seed_count, new_ideas=len(ideas) - seed_count, api_requests=len(requests))
    record["status"] = "passed" if record["new_ideas"] == 1 else "no_new_idea"
except Exception as exc:
    record.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000])
finally:
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    (output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
sys.exit(0 if record["status"] == "passed" else 1)
