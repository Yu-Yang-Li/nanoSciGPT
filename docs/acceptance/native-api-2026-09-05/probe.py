"""Bounded calls through original upstream functions, not a research run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--mode", choices=("v1-text", "v2-text", "v2-tool", "v2-image"), required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--image", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
sys.path.insert(0, str(args.upstream.resolve()))
# This loopback endpoint was independently tested without Authorization.
# Placeholder is only for the SDK constructor; not a cloud API credential.
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:10100/v1"
os.environ["OPENAI_API_KEY"] = "local-no-auth-probe"
record = {"mode": args.mode, "model_requested": args.model,
          "evidence_type": "upstream_function_api_probe_not_research",
          "endpoint": os.environ["OPENAI_BASE_URL"], "max_tokens": 256,
          "source_commit": None, "status": "started"}
start = time.monotonic()
try:
    import subprocess
    record["source_commit"] = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if args.mode == "v1-text":
        from ai_scientist import llm
        source = Path(llm.__file__)
        record["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        # Token budget only, original source file remains unchanged.
        llm.MAX_NUM_TOKENS = 256
        client, name = llm.create_client(args.model)
        client = client.with_options(timeout=20.0, max_retries=0)
        record["prompt"] = "Reply with exactly OK."
        answer, history = llm.get_response_from_llm(
            record["prompt"], client, name, "This is an API transport check.", temperature=0.1)
        record.update(answer=answer, history=history, matched_expected=answer.strip() == "OK")
    else:
        from ai_scientist.treesearch.backend import backend_openai
        from ai_scientist.treesearch.backend.utils import FunctionSpec
        source = Path(backend_openai.__file__)
        record["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = None
        prompt = "Reply with exactly OK."
        if args.mode == "v2-tool":
            spec = FunctionSpec("record_probe", {"type": "object", "properties": {
                "value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False},
                "Record the requested integer. This tool does not execute anything.")
            prompt = "Call record_probe with value 7."
        record["prompt"] = prompt
        if args.mode == "v2-image":
            import base64
            contents = args.image.read_bytes()
            question = "Read the chart. Return only the x-axis training step at which the blue curve reaches its maximum."
            record.update(prompt=question, image=str(args.image), image_sha256=hashlib.sha256(contents).hexdigest())
            prompt = [{"type": "text", "text": question}, {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(contents).decode("ascii")}}]
        answer, seconds, inputs, outputs, info = backend_openai.query(
            "This is an API transport check.", prompt, func_spec=spec,
            model=args.model, temperature=0.1, max_tokens=256)
        record.update(answer=answer, api_seconds=seconds, prompt_tokens=inputs,
                      completion_tokens=outputs, response_info=info,
                      matched_expected=answer == ({"value": 7} if spec else "3" if args.mode == "v2-image" else "OK"))
    record["status"] = "passed" if record["matched_expected"] else "unexpected_output"
except Exception as exc:
    record.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000])
finally:
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
sys.exit(0 if record["status"] == "passed" else 1)
