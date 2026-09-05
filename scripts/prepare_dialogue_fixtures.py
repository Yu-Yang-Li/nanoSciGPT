"""Recreate the three student-input fixtures used in the original CLI retests."""

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT / "data/smiles/delaney-processed.csv", destination / "student-solubility.csv")
    shutil.copyfile(ROOT / "data/weather/fixture.npz", destination / "student-weather.npz")
    (destination / "student-model").mkdir()
    (destination / "student-model/metrics.json").write_text('{"rmse": 0.42}\n', encoding="utf-8")
    manifest = {
        "measurement_status": "synthetic_test_input_not_a_run",
        "files": {
            "student-solubility.csv": "Copy of bundled Delaney ESOL data, not new student data.",
            "student-weather.npz": "Copy of generated teaching weather fixture; targets are scalar, intentionally not full next-time fields.",
            "student-model/metrics.json": "Invented student-reported input for a missing-run-evidence test, not a measured model result.",
        },
    }
    (destination / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tmp/dialogue-fixtures")
    print(prepare(parser.parse_args().output))
