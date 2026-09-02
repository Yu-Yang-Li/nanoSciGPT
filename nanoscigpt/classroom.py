"""Offline-first classroom runner for the bundled nanoSciGPT examples."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SEQUENCE_DOMAINS = ("text", "protein", "dna", "smiles")
STRUCTURED_DOMAINS = ("weather", "crystal", "structure3d", "image", "spectrum", "field")
RUNNABLE_DOMAINS = SEQUENCE_DOMAINS + STRUCTURED_DOMAINS
DEFAULT_PROFILE = "classroom"
CPU_PROFILES = {
    "smoke": {
        "device": "cpu",
        "max_iters": 2,
        "eval_interval": 1,
        "eval_iters": 1,
        "batch_size": 2,
        "block_size": 32,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "max_new_tokens": 4,
        "task_epochs": 2,
        "task_samples": 32,
        "structured_pretrain_steps": 2,
        "structured_task_steps": 2,
    },
    "classroom": {
        "device": "cpu",
        "max_iters": 30,
        "eval_interval": 15,
        "eval_iters": 5,
        "batch_size": 8,
        "block_size": 64,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "max_new_tokens": 20,
        "task_epochs": 20,
        "task_samples": 128,
        "structured_pretrain_steps": 20,
        "structured_task_steps": 20,
    },
}


def validate_manifest_files(domain, data_root):
    """Check every bundled file needed by the selected classroom flow."""
    data_root = Path(data_root)
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("domains", {}).get(domain)
    if entry is None:
        raise ValueError(f"{domain} is not declared in {manifest_path}")
    missing = []
    for relative in entry.get("required_files", []):
        parts = Path(relative).parts
        if parts and parts[0].lower() == "data":
            parts = parts[1:]
        candidate = data_root.joinpath(*parts)
        if not candidate.is_file():
            missing.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            f"{domain} bundled data is incomplete; missing: {', '.join(missing)}"
        )
    return entry


def validate_domain_data(domain, data_root="data"):
    if domain not in RUNNABLE_DOMAINS:
        raise ValueError(f"unknown classroom domain: {domain}")
    data_root = Path(data_root)
    manifest_entry = validate_manifest_files(domain, data_root)
    if domain in STRUCTURED_DOMAINS:
        from .tasks.structured_demo import validate_structured_fixture

        report = validate_structured_fixture(domain, data_root)
        report["source_name"] = manifest_entry.get("source_name") if manifest_entry else None
        return report
    data_dir = data_root / domain
    meta_path = data_dir / "meta.json"
    tokenizer_path = data_dir / "tokenizer.json"
    if not meta_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"{domain} data is incomplete under {data_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mode = meta.get("mode")
    if mode == "stream":
        train_path = data_dir / "train.bin"
        val_path = data_dir / "val.bin"
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(f"{domain} stream files are missing under {data_dir}")
        train = np.memmap(train_path, dtype=np.uint16, mode="r")
        val = np.memmap(val_path, dtype=np.uint16, mode="r")
        train_items, val_items = len(train), len(val)
        max_token = max(int(train.max()), int(val.max()))
    elif mode == "independent":
        train_path = data_dir / "train_seqs.npy"
        val_path = data_dir / "val_seqs.npy"
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(f"{domain} sequence files are missing under {data_dir}")
        train = np.load(train_path, allow_pickle=True)
        val = np.load(val_path, allow_pickle=True)
        train_items, val_items = len(train), len(val)
        max_token = max(
            max(int(np.max(sequence)) for sequence in train if len(sequence)),
            max(int(np.max(sequence)) for sequence in val if len(sequence)),
        )
    else:
        raise ValueError(f"{domain} has unsupported dataset mode={mode!r}")
    if train_items == 0 or val_items == 0:
        raise ValueError(f"{domain} has an empty train/validation split")
    if max_token >= int(meta["vocab_size"]):
        raise ValueError(f"{domain} contains token id {max_token} outside its vocabulary")
    return {
        "status": "ready",
        "domain": domain,
        "mode": mode,
        "train_items": train_items,
        "val_items": val_items,
        "vocab_size": int(meta["vocab_size"]),
        "data_dir": str(data_dir.resolve()),
        "source_name": manifest_entry.get("source_name") if manifest_entry else None,
    }


def run_command(command, cwd):
    print(f">>> {' '.join(str(part) for part in command)}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
    return completed


def run_domain(domain, profile, data_root, out_root, cwd=None):
    if profile not in CPU_PROFILES:
        raise ValueError(f"unknown profile={profile}; choose from {tuple(CPU_PROFILES)}")
    settings = CPU_PROFILES[profile]
    data_root = Path(data_root).resolve()
    out_dir = Path(out_root).resolve() / domain
    model_dir = out_dir / "model"
    downstream_dir = out_dir / "downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path(cwd or Path.cwd()).resolve()
    preflight = validate_domain_data(domain, data_root)
    started = time.perf_counter()

    if domain in STRUCTURED_DOMAINS:
        command = [
            sys.executable,
            "-m",
            "nanoscigpt.tasks.structured_demo",
            "--domain",
            domain,
            "--data_root",
            data_root,
            "--out_dir",
            out_dir,
            "--pretrain_steps",
            settings["structured_pretrain_steps"],
            "--task_steps",
            settings["structured_task_steps"],
        ]
        run_command(command, cwd)
        report = {
            "status": "completed",
            "domain": domain,
            "profile": profile,
            "device": settings["device"],
            "preflight": preflight,
            "downstream_task": "completed",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "artifacts": {
                "checkpoint": str(out_dir / "model" / "ckpt.pt"),
                "train_log": str(out_dir / "model" / "train_log.json"),
                "downstream": str(out_dir / "downstream" / "downstream_result.json"),
                "representation_preview": str(out_dir / "representation_preview.json"),
            },
            "commands": [[str(part) for part in command]],
        }
        report_path = out_dir / "run_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"classroom run completed: {domain} -> {report_path}", flush=True)
        return report

    trainer_command = [
        sys.executable,
        "-m",
        "nanoscigpt.core.trainer",
        "--domain",
        domain,
        "--data_root",
        data_root,
        "--out_dir",
        model_dir,
        "--device",
        settings["device"],
        "--max_iters",
        settings["max_iters"],
        "--eval_interval",
        settings["eval_interval"],
        "--eval_iters",
        settings["eval_iters"],
        "--batch_size",
        settings["batch_size"],
        "--block_size",
        settings["block_size"],
        "--n_layer",
        settings["n_layer"],
        "--n_head",
        settings["n_head"],
        "--n_embd",
        settings["n_embd"],
    ]
    run_command(trainer_command, cwd)

    sampler_command = [
        sys.executable,
        "-m",
        "nanoscigpt.core.sampler",
        "--domain",
        domain,
        "--data_root",
        data_root,
        "--out_dir",
        model_dir,
        "--device",
        settings["device"],
        "--num_samples",
        1,
        "--max_new_tokens",
        settings["max_new_tokens"],
    ]
    run_command(sampler_command, cwd)

    commands = [trainer_command, sampler_command]
    downstream_command = [
        sys.executable,
        "-m",
        "nanoscigpt.tasks.downstream_demo",
        "--domain",
        domain,
        "--ckpt",
        model_dir / "ckpt.pt",
        "--data_root",
        data_root,
        "--out_dir",
        downstream_dir,
        "--epochs",
        settings["task_epochs"],
        "--max_samples",
        settings["task_samples"],
    ]
    run_command(downstream_command, cwd)
    commands.append(downstream_command)
    downstream_task = "completed"

    report = {
        "status": "completed",
        "domain": domain,
        "profile": profile,
        "device": settings["device"],
        "preflight": preflight,
        "downstream_task": downstream_task,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "artifacts": {
            "checkpoint": str(model_dir / "ckpt.pt"),
            "train_log": str(model_dir / "train_log.json"),
            "downstream": str(downstream_dir / "downstream_result.json"),
        },
        "commands": [[str(part) for part in command] for command in commands],
    }
    report_path = out_dir / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"classroom run completed: {domain} -> {report_path}", flush=True)
    return report


def list_domains(data_root):
    ready = []
    for domain in RUNNABLE_DOMAINS:
        try:
            report = validate_domain_data(domain, data_root)
        except (FileNotFoundError, ValueError) as error:
            print(f"{domain}: unavailable ({error})")
        else:
            ready.append(domain)
            print(
                f"{domain}: ready "
                f"({report['mode']}, train={report['train_items']}, val={report['val_items']})"
            )
    return ready


def main():
    parser = argparse.ArgumentParser(
        description="Run an offline, CPU-first nanoSciGPT classroom example."
    )
    parser.add_argument("--domain", choices=RUNNABLE_DOMAINS + ("all",))
    parser.add_argument("--profile", choices=tuple(CPU_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_root", default="out/classroom")
    parser.add_argument("--list", action="store_true", help="show only choices that are ready")
    args = parser.parse_args()

    if args.list:
        list_domains(args.data_root)
        return
    if not args.domain:
        parser.error(
            "choose a domain reported by --list, choose --domain all, or use --list"
        )
    domains = RUNNABLE_DOMAINS if args.domain == "all" else (args.domain,)
    for domain in domains:
        run_domain(domain, args.profile, args.data_root, args.out_root)


if __name__ == "__main__":
    main()
