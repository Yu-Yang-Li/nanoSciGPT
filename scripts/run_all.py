"""Run all four domains end-to-end: prepare -> train -> sample."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOMAINS = [
    ("text", ["python", "-m", "nanoscigpt.domains.text.prepare"]),
    ("protein", ["python", "-m", "nanoscigpt.domains.protein.prepare", "--size", "500"]),
    ("dna", ["python", "-m", "nanoscigpt.domains.dna.prepare", "--fasta", "../nanoGPT-DNA/data/chr21.fa", "--num_bases", "500000"]),
    ("smiles", ["python", "-m", "nanoscigpt.domains.smiles.prepare"]),
]


def run(cmd):
    print(f">>> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    for domain, prepare_cmd in DOMAINS:
        run(prepare_cmd)
        run(["python", "-m", "nanoscigpt.core.trainer", "--domain", domain,
             "--max_iters", "100", "--eval_interval", "50", "--eval_iters", "10",
             "--block_size", "64", "--batch_size", "8",
             "--n_layer", "2", "--n_head", "2", "--n_embd", "64"])
        run(["python", "-m", "nanoscigpt.core.sampler", "--domain", domain,
             "--max_new_tokens", "40", "--num_samples", "1"])
        print(f"=== {domain} done ===\n", flush=True)


if __name__ == "__main__":
    main()
