"""Tool contracts: the ONLY way the virtual scientist can act on this repo.

Teaching point (B-line): an AI Scientist is not "an LLM doing whatever it
wants". Every action must be declared in advance as a contract: command
template, declared outputs, budget cap, and whether it needs human approval.
Anything not in a contract is not an available action.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_TRAIN_BASE = ["--block_size", "64", "--batch_size", "8",
               "--n_layer", "2", "--n_head", "2", "--n_embd", "64",
               "--eval_interval", "50", "--eval_iters", "10"]

CONTRACTS = {
    "prepare": {
        "desc": "prepare teaching data for one domain",
        "cmd": lambda domain, **_: [sys.executable, "-X", "utf8", "-m",
                                    f"nanoscigpt.domains.{domain}.prepare"],
        "declared_outputs": ["data/<domain>/train.*", "data/<domain>/val.*", "data/<domain>/meta.json"],
        "max_seconds": 600,
        "requires_approval": None,
    },
    "train_v0": {
        "desc": "train the smallest GPT for 100 iterations (V0 baseline)",
        "cmd": lambda domain, **_: [sys.executable, "-X", "utf8", "-m", "nanoscigpt.core.trainer",
                                    "--domain", domain, "--max_iters", "100"] + _TRAIN_BASE,
        "declared_outputs": ["out/<domain>/ckpt.pt", "out/<domain>/train_log.json"],
        "max_seconds": 900,
        "requires_approval": None,
    },
    "train_extended": {
        "desc": "double the training budget to 200 iterations (budget increase)",
        "cmd": lambda domain, **_: [sys.executable, "-X", "utf8", "-m", "nanoscigpt.core.trainer",
                                    "--domain", domain, "--max_iters", "200"] + _TRAIN_BASE,
        "declared_outputs": ["out/<domain>/ckpt.pt", "out/<domain>/train_log.json"],
        "max_seconds": 1800,
        "requires_approval": "budget_increase",
    },
    "sample": {
        "desc": "draw samples from the trained checkpoint",
        "cmd": lambda domain, **_: [sys.executable, "-X", "utf8", "-m", "nanoscigpt.core.sampler",
                                    "--domain", domain, "--max_new_tokens", "40", "--num_samples", "1"],
        "declared_outputs": ["stdout samples"],
        "max_seconds": 300,
        "requires_approval": None,
    },
    "transfer_probe": {
        "desc": "frozen-encoder probe: does our pretraining transfer? (protein only)",
        "cmd": lambda domain, **_: [sys.executable, "-X", "utf8", "-m", "nanoscigpt.tasks.transfer_probe"],
        "declared_outputs": ["out/transfer_probe/probe_results.json"],
        "max_seconds": 600,
        "requires_approval": None,
        "domains": ["protein"],
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
