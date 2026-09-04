"""Export portable classroom fallback summaries from completed smoke runs."""

import argparse
import json
from pathlib import Path


DOMAINS = (
    "text",
    "dna",
    "protein",
    "smiles",
    "weather",
    "crystal",
    "structure3d",
    "image",
    "spectrum",
    "field",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_result(domain: str, smoke_root: Path, data_manifest: dict, recorded_on: str) -> dict:
    domain_root = smoke_root / domain
    run_report = read_json(domain_root / "run_report.json")
    train_log = read_json(domain_root / "model" / "train_log.json")
    downstream = read_json(domain_root / "downstream" / "downstream_result.json")

    if run_report.get("status") != "completed" or downstream.get("status") != "completed":
        raise ValueError(f"{domain} smoke run is not complete")

    pretraining_name = train_log.pop("pretraining")
    train_log.pop("domain", None)
    return {
        "schema_version": 1,
        "artifact_type": "precomputed_classroom_fallback",
        "domain": domain,
        "status": "completed",
        "profile": run_report["profile"],
        "recorded_device": run_report["device"],
        "recorded_on": recorded_on,
        "teaching_only": True,
        "evidence_status": (
            "A portable summary of a completed repository smoke run; "
            "it is not evidence that the current student session ran."
        ),
        "source_data": {
            "name": data_manifest["domains"][domain]["source_name"],
            "manifest_entry": f"data/manifest.json#domains.{domain}",
        },
        "pretraining": {
            "name": pretraining_name,
            "metrics": train_log,
        },
        "downstream": {
            key: downstream[key]
            for key in (
                "task_name",
                "task_type",
                "label_source",
                "metric_name",
                "metric_value",
                "target_unit",
                "train_samples",
                "val_samples",
                "encoder_frozen",
                "pretrained_parameters_updated",
            )
            if key in downstream
        },
        "limitations": "See data/README.md for the domain-specific teaching boundary.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", default="out/acceptance-smoke")
    parser.add_argument("--data-manifest", default="data/manifest.json")
    parser.add_argument("--output-root", default="data/precomputed_results")
    parser.add_argument("--recorded-on", required=True)
    args = parser.parse_args()

    smoke_root = Path(args.smoke_root)
    output_root = Path(args.output_root)
    data_manifest = read_json(Path(args.data_manifest))
    output_root.mkdir(parents=True, exist_ok=True)

    result_manifest = {
        "schema_version": 1,
        "purpose": (
            "Portable fallback summaries for teaching when a live smoke run cannot finish. "
            "These files must never be presented as a result from the current student session."
        ),
        "recorded_on": args.recorded_on,
        "domains": {},
    }
    for domain in DOMAINS:
        result = build_result(domain, smoke_root, data_manifest, args.recorded_on)
        result_path = output_root / f"{domain}.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result_manifest["domains"][domain] = {
            "result_file": result_path.as_posix(),
            "source_run": f"out/acceptance-smoke/{domain}/run_report.json",
        }

    (output_root / "manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
