import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_student_protein_fasta_can_prepare_and_run_without_course_data(tmp_path):
    fasta = tmp_path / "student.fasta"
    sequences = [
        "M" + "ACDEFGHIKLMNPQRSTVWY" * 2 + amino
        for amino in "ACDEFGHIKLMNPQRSTVWY"
    ]
    fasta.write_text(
        "".join(f">student-{index}\n{sequence}\n" for index, sequence in enumerate(sequences)),
        encoding="utf-8",
    )
    data_root = tmp_path / "prepared"
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.domains.protein.prepare",
            "--fasta",
            str(fasta),
            "--out_dir",
            str(data_root / "protein"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    meta = json.loads((data_root / "protein" / "meta.json").read_text(encoding="utf-8"))
    assert meta["source_kind"] == "user_file"
    assert meta["source"] == str(fasta.resolve())
    assert meta["accepted_sequences"] == len(sequences)

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.classroom",
            "--domain",
            "protein",
            "--data_root",
            str(data_root),
            "--profile",
            "smoke",
            "--out_root",
            str(tmp_path / "runs"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(
        (tmp_path / "runs" / "protein" / "run_report.json").read_text(encoding="utf-8")
    )
    downstream = json.loads(
        (tmp_path / "runs" / "protein" / "downstream" / "downstream_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["preflight"]["source_kind"] == "user_file"
    assert report["preflight"]["source_name"] == str(fasta.resolve())
    assert downstream["label_source"] == "sequence-derived teaching label"
    assert downstream["teaching_only"] is True
