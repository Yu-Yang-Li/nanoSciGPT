"""Capture a fresh original v1 teaching-template baseline, not an Agent/API run."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", default="nanoSciGPT_teaching")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    setup = json.loads((checkout / "teaching_setup.json").read_text(encoding="utf-8"))
    if setup["project"] != "v1" or setup["device"] != "cpu":
        raise ValueError("this acceptance run requires the v1 CPU teaching template")
    actual_commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != setup["commit"]:
        raise ValueError("upstream revision has changed")
    source = (checkout / "templates" / args.template).resolve()
    if source.parent != checkout / "templates":
        raise ValueError("template must be a direct child of the native templates directory")
    bridge_path = source / "course_bridge.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8")) if bridge_path.exists() else None
    if bridge and hashlib.sha256((source / "initial_model.pt").read_bytes()).hexdigest() != bridge["initial_checkpoint_sha256"]:
        raise ValueError("course checkpoint no longer matches bridge receipt")
    template = checkout / "templates" / ("nanoSciGPT_verify_" + uuid.uuid4().hex[:8])
    shutil.copytree(source, template,
                    ignore=shutil.ignore_patterns("run_*", "__pycache__"))
    command = [sys.executable, "experiment.py", "--out_dir", "run_0"]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    output.mkdir(parents=True)
    start = time.monotonic()
    with (output / "stdout.txt").open("w", encoding="utf-8") as stdout, (output / "stderr.txt").open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=template, env=environment, stdout=stdout, stderr=stderr, timeout=120)
    supervised = bool(bridge and bridge.get("task_type"))
    record = {"scope": "v1_supervised_teaching_template_baseline_only" if supervised else "original_v1_template_baseline_only", "agent_api_run": False,
              "commit": actual_commit, "command": command, "cwd": str(template),
              "returncode": process.returncode, "elapsed_seconds": round(time.monotonic() - start, 3),
              "experiment_sha256": hashlib.sha256((template / "experiment.py").read_bytes()).hexdigest()}
    if bridge:
        record["course_bridge"] = bridge
    if process.returncode == 0:
        record["result"] = json.loads((template / "run_0/final_info.json").read_text())
        shutil.copyfile(template / "run_0/final_info.json", output / "final_info.json")
    (output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
