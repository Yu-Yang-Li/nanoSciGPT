"""Repository-level entry point for the first supervised classroom baseline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / "skills"
    / "research-baseline-builder"
    / "scripts"
    / "run_research_baseline_workflow.py"
)
LAMOST = ROOT / "data" / "course" / "lamost_atlas_a_teff_2000.csv"


def build_command(args: argparse.Namespace) -> list[str]:
    if args.case == "lamost":
        topic = "LAMOST光谱有效温度回归"
        csv_path = LAMOST
        target = "teff"
        task = "regression"
    else:
        topic = args.topic
        csv_path = args.csv.resolve()
        target = args.target
        task = args.task

    command = [
        sys.executable,
        str(WORKFLOW),
        topic,
        "--root",
        str(args.out_root.resolve()),
        "--template",
        "rf",
        "--run-baseline",
        "--task",
        task,
        "--csv",
        str(csv_path),
        "--target",
        target,
    ]
    if args.force:
        command.append("--force")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bundled LAMOST baseline or a student CSV baseline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", choices=("lamost",))
    source.add_argument("--csv", type=Path)
    parser.add_argument("--target", help="Target column for --csv.")
    parser.add_argument(
        "--task", choices=("classification", "regression"), default="classification"
    )
    parser.add_argument("--topic", default="student table")
    parser.add_argument("--out_root", type=Path, default=Path("out/baseline"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.csv is not None and not args.target:
        parser.error("--csv requires --target")
    if not WORKFLOW.is_file():
        parser.error(f"baseline workflow not found: {WORKFLOW}")
    if args.case == "lamost" and not LAMOST.is_file():
        parser.error(f"bundled LAMOST data not found: {LAMOST}")

    completed = subprocess.run(build_command(args), cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
