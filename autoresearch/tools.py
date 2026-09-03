"""Tool contracts: the ONLY way the virtual scientist can act on this repo.

Teaching point (B-line): an AI Scientist is not "an LLM doing whatever it
wants". Every action must be declared in advance as a contract: command
template, declared outputs, budget cap, and whether it needs human approval.
Anything not in a contract is not an available action.
"""

import subprocess
import sys
from pathlib import Path

from nanoscigpt.domains.registry import STRUCTURED_DOMAINS

REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_BASE = ["--block_size", "64", "--batch_size", "8",
               "--n_layer", "2", "--n_head", "2", "--n_embd", "64",
               "--eval_interval", "50", "--eval_iters", "10"]


def prepare_command(domain, **_):
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "nanoscigpt.tasks.prepare_domain",
        "--domain",
        domain,
    ]


def train_command(domain, iterations):
    if domain in STRUCTURED_DOMAINS:
        steps = 10 if iterations == 100 else 20
        return [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "nanoscigpt.tasks.structured_demo",
            "--domain",
            domain,
            "--out_dir",
            str(Path("out") / domain),
            "--pretrain_steps",
            str(steps),
            "--task_steps",
            "5",
        ]
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "nanoscigpt.core.trainer",
        "--domain",
        domain,
        "--max_iters",
        str(iterations),
    ] + _TRAIN_BASE


def sample_command(domain, **_):
    if domain in STRUCTURED_DOMAINS:
        return [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "nanoscigpt.tasks.inspect_structured",
            "--domain",
            domain,
            "--out_dir",
            str(Path("out") / domain),
        ]
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "nanoscigpt.core.sampler",
        "--domain",
        domain,
        "--max_new_tokens",
        "40",
        "--num_samples",
        "1",
    ]

CONTRACTS = {
    "prepare": {
        "desc": "prepare teaching data for one domain",
        "cmd": prepare_command,
        "declared_outputs": ["data/<domain>/meta.json", "data/<domain>/prepared payload"],
        "max_seconds": 600,
        "requires_approval": None,
    },
    "train_v0": {
        "desc": "run the smallest sequence or structured pretraining baseline",
        "cmd": lambda domain, **_: train_command(domain, 100),
        "declared_outputs": ["out/<domain>/checkpoint", "out/<domain>/train log"],
        "max_seconds": 900,
        "requires_approval": None,
    },
    "train_extended": {
        "desc": "double the domain-appropriate pretraining budget",
        "cmd": lambda domain, **_: train_command(domain, 200),
        "declared_outputs": ["out/<domain>/checkpoint", "out/<domain>/train log"],
        "max_seconds": 1800,
        "requires_approval": "budget_increase",
    },
    "sample": {
        "desc": "draw samples from the trained checkpoint",
        "cmd": sample_command,
        "declared_outputs": ["stdout sample or representation preview"],
        "max_seconds": 300,
        "requires_approval": None,
    },
}


def run_tool(name, domain, verbose=True):
    """Execute one contracted tool against this repo. Returns (ok, stdout, stderr).

    The scientist never builds its own shell commands; it picks a contract.
    """
    c = CONTRACTS.get(name)
    if c is None:
        raise SystemExit(f"action '{name}' has no tool contract - refused by design")
    if "domains" in c and domain not in c["domains"]:
        return False, "", f"contract '{name}' not applicable to domain '{domain}'"
    cmd = c["cmd"](domain)
    if verbose:
        print(f"  [tool] {' '.join(cmd)}", flush=True)
    try:
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=c["max_seconds"], encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "", f"timeout after {c['max_seconds']}s"
    return r.returncode == 0, r.stdout, r.stderr
