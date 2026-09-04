import csv
import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_dependencies_cover_the_bundled_lamost_baseline():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {item.split(">=")[0].lower() for item in project["dependencies"]}

    assert {"numpy", "pandas", "scikit-learn", "torch"} <= names


def test_lamost_baseline_cli_preserves_course_data_identity_and_artifacts(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.baseline",
            "--case",
            "lamost",
            "--out_root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = list(tmp_path.rglob("workflow_status.json"))
    assert len(reports) == 1
    workspace = reports[0].parent
    status = json.loads(reports[0].read_text(encoding="utf-8"))
    summary = json.loads((workspace / "baseline_summary.json").read_text(encoding="utf-8"))

    assert status["status"] == "baseline_completed"
    assert status["data_mode"] == "bundled_course"
    assert summary["data_mode"] == "bundled_course"
    assert summary["data_source_name"] == "ATLAS-A LAMOST teaching subset"
    assert summary["task"] == "regression"
    assert (workspace / "rf_model.joblib").is_file()
    assert (workspace / "metrics.json").is_file()
    assert (workspace / "train_log.txt").is_file()


def test_baseline_cli_accepts_a_student_csv_without_calling_it_course_data(tmp_path):
    csv_path = tmp_path / "student.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_a", "feature_b", "target"])
        for index in range(40):
            writer.writerow([index, index % 5, index % 2])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.baseline",
            "--csv",
            str(csv_path),
            "--target",
            "target",
            "--task",
            "classification",
            "--topic",
            "student table",
            "--out_root",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report_path = next((tmp_path / "out").rglob("workflow_status.json"))
    status = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads((report_path.parent / "baseline_summary.json").read_text(encoding="utf-8"))
    assert status["data_mode"] == "user_csv"
    assert summary["data_mode"] == "user_csv"
    assert summary["data_source_name"] is None


def test_baseline_cli_does_not_train_a_student_csv_without_a_target(tmp_path):
    csv_path = tmp_path / "unlabeled.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_a", "feature_b"])
        writer.writerow([1, 2])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.baseline",
            "--csv",
            str(csv_path),
            "--out_root",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "--csv requires --target" in completed.stderr
    assert not (tmp_path / "out").exists()
