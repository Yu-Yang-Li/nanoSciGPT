"""Print the structured representation preview produced by a classroom run."""

import argparse
import json
from pathlib import Path

from .structured_demo import STRUCTURED_DOMAINS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=STRUCTURED_DOMAINS)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path("out") / args.domain
    preview_path = out_dir / "representation_preview.json"
    result_path = out_dir / "downstream" / "downstream_result.json"
    if not preview_path.is_file() or not result_path.is_file():
        raise SystemExit(f"structured artifacts missing under {out_dir}")
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"representation preview: {preview}")
    print(f"task artifact: {result['task_name']} ({result['metric_name']})")


if __name__ == "__main__":
    main()
