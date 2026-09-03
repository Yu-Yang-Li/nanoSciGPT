import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("domain", "sample_shape", "patch_size"),
    [
        ("weather", (4, 8, 8), 4),
        ("image", (1, 8, 8), 4),
        ("spectrum", (1, 16), 4),
        ("field", (4, 16), 4),
        ("structure3d", (10, 3), 1),
    ],
)
def test_structured_user_contract_accepts_each_supported_shape(
    domain, sample_shape, patch_size
):
    from nanoscigpt.prepare_structured import validate_arrays

    arrays = {
        "train_x": np.ones((5, *sample_shape), dtype=np.float32),
        "val_x": np.ones((2, *sample_shape), dtype=np.float32),
        "train_y": np.ones(5, dtype=np.float32),
        "val_y": np.ones(2, dtype=np.float32),
    }

    validate_arrays(domain, arrays, patch_size)


@pytest.mark.parametrize(
    ("domain", "sample_shape", "patch_size"),
    [
        ("weather", (4, 8, 8), 4),
        ("image", (1, 8, 8), 4),
        ("spectrum", (1, 16), 4),
        ("field", (4, 16), 4),
        ("structure3d", (10, 3), 1),
    ],
)
def test_structured_user_contract_accepts_unlabeled_pretraining_arrays(
    domain, sample_shape, patch_size
):
    from nanoscigpt.prepare_structured import validate_arrays

    arrays = {
        "train_x": np.ones((5, *sample_shape), dtype=np.float32),
        "val_x": np.ones((2, *sample_shape), dtype=np.float32),
    }

    validate_arrays(domain, arrays, patch_size, require_labels=False)


def test_structured_user_contract_rejects_non_numeric_arrays():
    from nanoscigpt.prepare_structured import validate_arrays

    arrays = {
        "train_x": np.array([[['not-a-number']]], dtype=object),
        "val_x": np.array([[['not-a-number']]], dtype=object),
        "train_y": np.ones(1, dtype=np.float32),
        "val_y": np.ones(1, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="numeric"):
        validate_arrays("spectrum", arrays, 1)


def crystal_arrays(train_samples=8, val_samples=3, nodes=4):
    rng = np.random.default_rng(17)

    def split(count):
        atomic_numbers = rng.integers(1, 15, size=(count, nodes), dtype=np.int64)
        fractional = rng.random((count, nodes, 3), dtype=np.float32)
        mask = np.ones((count, nodes), dtype=bool)
        lattice = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], count, axis=0)
        target = atomic_numbers.mean(axis=1).astype(np.float32)
        return atomic_numbers, fractional, mask, lattice, target

    train = split(train_samples)
    val = split(val_samples)
    return {
        "train_atomic_numbers": train[0],
        "val_atomic_numbers": val[0],
        "train_fractional": train[1],
        "val_fractional": val[1],
        "train_mask": train[2],
        "val_mask": val[2],
        "train_lattice": train[3],
        "val_lattice": val[3],
        "train_y": train[4],
        "val_y": val[4],
    }


def test_crystal_user_contract_accepts_periodic_graph_arrays():
    from nanoscigpt.prepare_structured import validate_arrays

    validate_arrays("crystal", crystal_arrays(), patch_size=1)


def test_crystal_user_contract_accepts_unlabeled_pretraining_arrays():
    from nanoscigpt.prepare_structured import validate_arrays

    arrays = crystal_arrays()
    arrays.pop("train_y")
    arrays.pop("val_y")
    validate_arrays("crystal", arrays, patch_size=1, require_labels=False)


def test_crystal_user_contract_rejects_invalid_atomic_numbers():
    from nanoscigpt.prepare_structured import validate_arrays

    arrays = crystal_arrays()
    arrays["train_atomic_numbers"][0, 0] = 119
    with pytest.raises(ValueError, match="1 through 118"):
        validate_arrays("crystal", arrays, patch_size=1)


def test_student_spectrum_npz_runs_with_user_labels_and_provenance(tmp_path):
    source = tmp_path / "spectra.npz"
    rng = np.random.default_rng(42)
    np.savez_compressed(
        source,
        train_x=rng.normal(size=(20, 1, 32)).astype(np.float32),
        val_x=rng.normal(size=(6, 1, 32)).astype(np.float32),
        train_y=np.linspace(4000, 7000, 20, dtype=np.float32),
        val_y=np.linspace(4500, 6500, 6, dtype=np.float32),
    )
    data_root = tmp_path / "prepared"
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.prepare_structured",
            "--domain",
            "spectrum",
            "--npz",
            str(source),
            "--out-dir",
            str(data_root / "spectrum"),
            "--patch-size",
            "8",
            "--task-name",
            "stellar temperature regression",
            "--sample-unit",
            "one normalized spectrum",
            "--target-unit",
            "kelvin",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    meta = json.loads((data_root / "spectrum" / "meta.json").read_text(encoding="utf-8"))
    assert meta["source_kind"] == "user_file"
    assert meta["source"] == str(source.resolve())
    assert meta["teaching_only"] is False
    assert meta["label_source"] == "user_provided"

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.classroom",
            "--domain",
            "spectrum",
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
        (tmp_path / "runs" / "spectrum" / "run_report.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (
            tmp_path
            / "runs"
            / "spectrum"
            / "downstream"
            / "downstream_result.json"
        ).read_text(encoding="utf-8")
    )
    assert report["preflight"]["source_kind"] == "user_file"
    assert report["preflight"]["source_name"] == str(source.resolve())
    assert result["label_source"] == "user_provided"
    assert result["teaching_only"] is False
    assert result["task_name"] == "stellar temperature regression"


def test_student_unlabeled_spectrum_npz_can_run_pretraining_only(tmp_path):
    source = tmp_path / "unlabeled-spectra.npz"
    rng = np.random.default_rng(43)
    np.savez_compressed(
        source,
        train_x=rng.normal(size=(20, 1, 32)).astype(np.float32),
        val_x=rng.normal(size=(6, 1, 32)).astype(np.float32),
    )
    data_root = tmp_path / "prepared"
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.prepare_structured",
            "--domain",
            "spectrum",
            "--npz",
            str(source),
            "--out-dir",
            str(data_root / "spectrum"),
            "--patch-size",
            "8",
            "--sample-unit",
            "one normalized spectrum",
            "--skip-downstream",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    meta = json.loads((data_root / "spectrum" / "meta.json").read_text(encoding="utf-8"))
    assert meta["has_labels"] is False
    assert "task_name" not in meta
    assert "label_source" not in meta

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.classroom",
            "--domain",
            "spectrum",
            "--data_root",
            str(data_root),
            "--profile",
            "smoke",
            "--out_root",
            str(tmp_path / "runs"),
            "--skip-downstream",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(
        (tmp_path / "runs" / "spectrum" / "run_report.json").read_text(encoding="utf-8")
    )
    assert report["downstream_task"] == "not_requested"
    assert "downstream" not in report["artifacts"]
    assert not (tmp_path / "runs" / "spectrum" / "downstream").exists()


def test_student_crystal_npz_runs_with_user_labels_and_provenance(tmp_path):
    source = tmp_path / "crystals.npz"
    np.savez_compressed(source, **crystal_arrays())
    data_root = tmp_path / "prepared"
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.prepare_structured",
            "--domain",
            "crystal",
            "--npz",
            str(source),
            "--out-dir",
            str(data_root / "crystal"),
            "--task-name",
            "formation energy regression",
            "--sample-unit",
            "one periodic crystal cell",
            "--target-unit",
            "eV per atom",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.classroom",
            "--domain",
            "crystal",
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
        (tmp_path / "runs" / "crystal" / "run_report.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (tmp_path / "runs" / "crystal" / "downstream" / "downstream_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["preflight"]["source_name"] == str(source.resolve())
    assert report["preflight"]["source_kind"] == "user_file"
    assert result["label_source"] == "user_provided"
    assert result["teaching_only"] is False
    assert result["task_name"] == "formation energy regression"
