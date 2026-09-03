"""Check whether the local nanoSciGPT classroom is ready to run."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


REQUIRED_MODULES = ("torch", "numpy", "pandas", "sklearn")
DISTRIBUTIONS = {"sklearn": "scikit-learn"}
SKILL_NAMES = (
    "research-baseline-builder",
    "nanogpt-pretraining",
    "nanoscigpt-scientific-language",
    "autoresearch-model-iteration",
    "ai-scientist-v1-workflow",
    "ai-scientist-v2-tree-search",
)


def _dependency_report(name: str) -> dict:
    available = importlib.util.find_spec(name) is not None
    distribution = DISTRIBUTIONS.get(name, name)
    version = None
    if available:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"available": available, "version": version}


def _data_report(root: Path) -> dict:
    manifest_path = root / "data" / "manifest.json"
    if not manifest_path.is_file():
        return {"ready": 0, "total": 0, "domains": {}, "error": "data/manifest.json missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    domains = {}
    for name, entry in manifest.get("domains", {}).items():
        missing = [
            relative
            for relative in entry.get("required_files", [])
            if not (root / relative).is_file()
        ]
        domains[name] = {"ready": not missing, "missing": missing}
    return {
        "ready": sum(item["ready"] for item in domains.values()),
        "total": len(domains),
        "domains": domains,
    }


def _skill_report(root: Path) -> dict:
    skills = {
        name: {"ready": (root / "skills" / name / "SKILL.md").is_file()}
        for name in SKILL_NAMES
    }
    return {
        "ready": sum(item["ready"] for item in skills.values()),
        "total": len(skills),
        "items": skills,
    }


def build_report(
    root: Path,
    required_modules: tuple[str, ...] = REQUIRED_MODULES,
) -> dict:
    root = Path(root).resolve()
    dependencies = {name: _dependency_report(name) for name in required_modules}
    data = _data_report(root)
    skills = _skill_report(root)
    python_supported = sys.version_info >= (3, 10)
    ready = (
        python_supported
        and all(item["available"] for item in dependencies.values())
        and data["total"] == 10
        and data["ready"] == data["total"]
        and skills["ready"] == skills["total"]
    )
    return {
        "status": "ready" if ready else "not_ready",
        "root": str(root),
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "executable": sys.executable,
            "supported": python_supported,
        },
        "dependencies": dependencies,
        "data": data,
        "skills": skills,
    }


def _print_human(report: dict) -> None:
    print(f"nanoSciGPT classroom: {report['status']}")
    print(
        f"python {report['python']['version']}: "
        f"{'ready' if report['python']['supported'] else 'requires 3.10+'}"
    )
    for name, item in report["dependencies"].items():
        suffix = f" {item['version']}" if item["version"] else ""
        print(f"dependency {name}: {'ready' if item['available'] else 'missing'}{suffix}")
    print(f"bundled data: {report['data']['ready']}/{report['data']['total']} ready")
    print(f"course skills: {report['skills']['ready']}/{report['skills']['total']} ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the complete report as JSON")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root; defaults to the installed source checkout",
    )
    args = parser.parse_args()
    try:
        report = build_report(args.root)
    except (OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
