"""Run all bundled teaching routes sequentially and preserve real command logs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("text", "protein", "dna", "smiles", "weather", "crystal", "structure3d", "image", "spectrum", "field")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    records = []

    def command(name, *arguments):
        invocation = [sys.executable, *map(str, arguments)]
        start = time.monotonic()
        with (output / f"{name}.stdout.txt").open("w", encoding="utf-8") as stdout, (output / f"{name}.stderr.txt").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(invocation, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=180)
        records.append({"name": name, "command": invocation, "exit_code": result.returncode,
                        "elapsed_seconds": round(time.monotonic() - start, 3)})
        (output / "commands.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"{name}: exit={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see recorded logs")

    command("baseline", "-m", "nanoscigpt.baseline", "--case", "lamost", "--out_root", output / "baseline")
    summary = {"evidence_type": "direct_command_runs_not_cli_dialogue", "domains": {}}
    for domain in DOMAINS:
        command(f"{domain}-pretrain", "-m", "nanoscigpt.classroom", "--domain", domain,
                "--profile", "classroom", "--out_root", output / "training")
        checkpoint = output / "training" / domain / "model/ckpt.pt"
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        task_output = output / "training" / domain / "finetune"
        command(f"{domain}-finetune", "-m", "nanoscigpt.tasks.downstream_demo", "--domain", domain,
                "--ckpt", checkpoint, "--adaptation", "finetune", "--epochs", "2", "--max_samples", "32", "--out_dir", task_output)
        task = json.loads((task_output / "downstream_result.json").read_text(encoding="utf-8"))
        unchanged = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
        if not unchanged or not task["pretrained_parameters_updated"]:
            raise RuntimeError(f"{domain} parameter update or original-file preservation check failed")
        summary["domains"][domain] = {**task, "original_checkpoint_unchanged": unchanged}
    command("text-sft", "-m", "nanoscigpt.tasks.text_sft", "--ckpt", output / "training/text/model/ckpt.pt",
            "--out_dir", output / "text-sft", "--steps", "200")
    summary["text_sft"] = json.loads((output / "text-sft/sft_result.json").read_text(encoding="utf-8"))
    (output / "acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {output / 'acceptance.json'}")


if __name__ == "__main__":
    main()
