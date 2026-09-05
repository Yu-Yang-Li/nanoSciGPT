"""Capture supervised course -> native v1 task template -> course continuation.

This runs real local experiments, not the API-dependent v1 research agents.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--domain", choices=("text", "dna", "protein", "smiles"), required=True)
    parser.add_argument("--checkout", type=Path, default=ROOT / "out/upstream/v1")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    original_hash = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}

    def run(name, arguments, cwd=ROOT):
        invocation = [sys.executable, *map(str, arguments)]
        started = time.monotonic()
        with (output / f"{name}.stdout.txt").open("w", encoding="utf-8") as stdout, (output / f"{name}.stderr.txt").open("w", encoding="utf-8") as stderr:
            process = subprocess.run(invocation, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, timeout=120)
        records.append({"name": name, "command": invocation, "cwd": str(cwd), "exit_code": process.returncode,
                        "elapsed_seconds": round(time.monotonic() - started, 3)})
        (output / "commands.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect saved logs")

    def fine(name, checkpoint):
        run(name, ["-m", "nanoscigpt.tasks.downstream_demo", "--domain", args.domain,
                   "--ckpt", checkpoint, "--data_root", args.data_root.resolve(), "--out_dir", output / name,
                   "--epochs", 2, "--max_samples", 32, "--adaptation", "finetune"])
        return json.loads((output / name / "downstream_result.json").read_text(encoding="utf-8"))

    first = fine("course-fine", args.ckpt.resolve())
    run("export", ["-m", "nanoscigpt.native_v1", "--checkout", args.checkout.resolve(),
                   "--ckpt", first["task_checkpoint"], "--data_root", args.data_root.resolve(),
                   "--name", args.name, "--steps", 4])
    run("native", ["scripts/capture_native_v1_baseline.py", "--checkout", args.checkout.resolve(),
                   "--template", args.name, "--output", output / "native"])
    native = json.loads((output / "native/record.json").read_text(encoding="utf-8"))
    template = Path(native["cwd"])
    run("plot", ["plot.py"], cwd=template)
    continued = fine("course-return", template / "run_0/checkpoint.pt")
    metric = "val_mae" if first["task_type"] == "regression" else "val_accuracy"
    values = native["result"]["task"]["means"]
    assert abs(first["metric_value"] - values["initial_" + metric]) <= 5.1e-5
    assert abs(continued["metric_before_finetune"] - values[metric]) <= 5.1e-5
    assert original_hash == hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    assert (template / "task_training.png").stat().st_size > 1000
    summary = {"evidence_type": "direct_command_runs_not_cli_dialogue", "agent_api_run": False,
               "source_checkpoint_sha256": original_hash, "course": first, "native": native,
               "course_return": continued, "round_trip_predictions_match": True}
    (output / "acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
