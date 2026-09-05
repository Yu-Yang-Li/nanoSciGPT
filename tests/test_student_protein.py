import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoscigpt.classroom import run_domain
from nanoscigpt.tasks.downstream_demo import run_downstream


ROOT = Path(__file__).resolve().parents[1]


def source_csv(path, labeled=True):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "split", "activity"])
        for index, residue in enumerate("ACDEFGHI"):
            writer.writerow(["M" + residue * 8, "train" if index < 6 else "val", 10 + index if labeled else ""])


@pytest.mark.parametrize("labeled", [True, False])
def test_student_protein_runs_own_sequences_and_never_invents_labels(tmp_path, labeled):
    from nanoscigpt.student_protein import prepare

    source = tmp_path / "student.csv"
    source_csv(source, labeled)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = tmp_path / "data"
    prepare(source, data, "sequence", "activity" if labeled else None, "split")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert np.load(data / "protein/train_seqs.npy", allow_pickle=True).shape == (6,)
    torch.set_num_threads(1)
    report = run_domain("protein", "smoke", data, tmp_path / "run", cwd=ROOT)
    result = json.loads(Path(report["artifacts"]["downstream"]).read_text())
    if labeled:
        assert result["task_type"] == "regression"
        assert result["label_source"] == "student CSV column: activity"
        assert np.load(data / "protein/train_y.npy").tolist() == [10, 11, 12, 13, 14, 15]
        assert result["source_sha256"] == digest
        tuned = run_downstream("protein", report["artifacts"]["checkpoint"], data, tmp_path / "fine",
                               epochs=1, max_samples=8, adaptation="finetune")
        assert tuned["pretrained_parameters_updated"]
    else:
        assert report["downstream_task"] == "skipped_no_labels"
        assert "metric_value" not in result
        assert not (data / "protein/train_y.npy").exists()


def test_student_import_rejects_overlap_without_writing_prepared_data(tmp_path):
    from nanoscigpt.student_protein import prepare

    source = tmp_path / "overlap.csv"
    source.write_text("sequence,split,activity\nMACDE,train,1\nMACDE,val,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both train and val"):
        prepare(source, tmp_path / "data", "sequence", "activity", "split")
    assert not (tmp_path / "data/protein").exists()


def test_student_import_does_not_overwrite_an_existing_dataset(tmp_path):
    from nanoscigpt.student_protein import prepare

    source = tmp_path / "student.csv"
    source_csv(source)
    data = tmp_path / "data/protein"
    data.mkdir(parents=True)
    (data / "keep.txt").write_text("student work")
    with pytest.raises(FileExistsError):
        prepare(source, data.parent, "sequence", "activity", "split")
    assert (data / "keep.txt").read_text() == "student work"


def test_small_student_csv_can_use_automatic_holdout(tmp_path):
    from nanoscigpt.student_protein import prepare

    source = tmp_path / "student.csv"
    source_csv(source)
    prepare(source, tmp_path / "data", "sequence", "activity")
    assert len(np.load(tmp_path / "data/protein/val_y.npy")) == 2
    assert len(np.load(tmp_path / "data/protein/train_y.npy")) == 6
