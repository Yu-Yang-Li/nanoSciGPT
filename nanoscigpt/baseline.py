"""Repository-level entry point for the first supervised classroom baseline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / "skills"
    / "nanoscigpt-research-baseline-builder"
    / "scripts"
    / "run_research_baseline_workflow.py"
)
LAMOST = ROOT / "data" / "course" / "lamost_atlas_a_teff_2000.csv"


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def build_command(args: argparse.Namespace) -> list[str]:
    if args.case == "lamost":
        topic = "LAMOST光谱有效温度回归"
        csv_path = LAMOST
        target = "teff"
        task = "regression"
        template = "rf"
    elif args.series_csv is not None:
        topic = args.topic
        csv_path = args.series_csv.resolve()
        target = None
        task = None
        template = "gru"
    else:
        topic = args.topic
        csv_path = args.csv.resolve()
        target = args.target
        task = args.task
        template = "rf"

    command = [
        sys.executable,
        str(WORKFLOW),
        topic,
        "--root",
        str(args.out_root.resolve()),
        "--template",
        template,
        "--run-baseline",
        "--csv",
        str(csv_path),
    ]
    if template == "gru":
        command += ["--value-column", args.value_column]
        if args.time_column:
            command += ["--time-column", args.time_column]
        command += ["--epochs", str(args.epochs), "--seq_len", str(args.seq_len)]
    else:
        command += ["--task", task, "--target", target]
    if args.force:
        command.append("--force")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bundled LAMOST baseline, a labeled table, or a time-series CSV."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", choices=("lamost",))
    source.add_argument("--csv", type=Path)
    source.add_argument("--series-csv", type=Path)
    parser.add_argument("--target", help="Target column for --csv.")
    parser.add_argument("--value-column", help="Numeric signal column for --series-csv.")
    parser.add_argument(
        "--time-column",
        help="Optional ordering column for --series-csv; otherwise existing row order is used.",
    )
    parser.add_argument(
        "--task", choices=("classification", "regression"), default="classification"
    )
    parser.add_argument("--topic", default="student table")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--out_root", type=Path, default=Path("out/baseline"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.csv is not None and not args.target:
        parser.error("--csv requires --target")
    if args.series_csv is not None and not args.value_column:
        parser.error("--series-csv requires --value-column")
    if not WORKFLOW.is_file():
        parser.error(f"baseline workflow not found: {WORKFLOW}")
    if args.case == "lamost" and not LAMOST.is_file():
        parser.error(f"bundled LAMOST data not found: {LAMOST}")

    completed = subprocess.run(
        build_command(args), cwd=ROOT, env=subprocess_environment()
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
