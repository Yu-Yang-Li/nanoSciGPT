"""Full autoresearch pipeline: S1 hypothesis -> S2 experiment -> S3 paper.

The three segments map onto the AI Scientist lecture timeline:
  S1  hypothesis generation    (AstroInsight, EPJ Data Science 2026)
  S2  experiment closed loop   (StarWhisper Telescope, Comms Eng 2025)
  S3  paper/review/revision    (instructor's agentic-research practice)

Usage:
    python -m autoresearch.pipeline --domain text --auto_approve
"""

import argparse
from pathlib import Path

from nanoscigpt.domains.registry import RUNNABLE_DOMAINS


def main():
    p = argparse.ArgumentParser(description="Run the full S1->S2->S3 pipeline")
    p.add_argument("--domain", default="text", choices=RUNNABLE_DOMAINS)
    p.add_argument("--fresh", action="store_true", help="reset research state first")
    p.add_argument("--auto_approve", action="store_true")
    args = p.parse_args()

    state_path = Path(__file__).resolve().parent / f"research_state_{args.domain}.json"
    if args.fresh and state_path.exists():
        state_path.unlink()

    import subprocess
    import sys
    steps = [
        ("S1 hypothesis", ["-m", "autoresearch.hypothesis"], True),
        ("S2 experiment", ["-m", "autoresearch.experiment"], True),
        ("S3 paper", ["-m", "autoresearch.paper"], False),
    ]
    for name, mod, needs_approval in steps:
        cmd = [sys.executable, "-X", "utf8"] + mod + ["--domain", args.domain]
        if args.auto_approve and needs_approval:
            cmd.append("--auto_approve")
        print(f"\n{'='*60}\n[{name}]\n{'='*60}", flush=True)
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
        if r.returncode != 0:
            raise SystemExit(f"{name} failed with code {r.returncode}")

    print(f"\n{'='*60}\npipeline complete for domain '{args.domain}'\n{'='*60}", flush=True)


if __name__ == "__main__":
    main()
