import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RUNNABLE = (
    "text",
    "protein",
    "dna",
    "smiles",
    "weather",
    "crystal",
    "structure3d",
    "image",
    "spectrum",
    "field",
)


def test_run_command_prints_child_error_before_raising(monkeypatch, capsys, tmp_path):
    from nanoscigpt.classroom import run_command

    def failed(command, **kwargs):
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command, 1, stdout="child stdout\n", stderr="child stderr\n"
        )

    monkeypatch.setattr(subprocess, "run", failed)

    with pytest.raises(subprocess.CalledProcessError):
        run_command(["python", "broken.py"], tmp_path)

    captured = capsys.readouterr()
    assert "child stdout" in captured.out
    assert "child stderr" in captured.err


def test_bundled_data_manifest_covers_every_runnable_domain():
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert set(manifest["domains"]) == set(RUNNABLE)
    for domain, entry in manifest["domains"].items():
        assert entry["bundled"] is True
        assert entry["source_name"]
        if "source_url" in entry:
            assert entry["source_url"].startswith("https://")
        else:
            assert (ROOT / entry["generator"]).is_file()
        assert entry["license_note"]
        for relative_path in entry["required_files"]:
            assert (ROOT / relative_path).is_file(), (domain, relative_path)


def test_classroom_profiles_default_to_small_cpu_runs():
    from nanoscigpt.classroom import CPU_PROFILES, DEFAULT_PROFILE

    assert DEFAULT_PROFILE == "classroom"
    assert CPU_PROFILES[DEFAULT_PROFILE]["device"] == "cpu"
    assert CPU_PROFILES[DEFAULT_PROFILE]["max_iters"] <= 30
    assert CPU_PROFILES[DEFAULT_PROFILE]["n_layer"] <= 2
    assert CPU_PROFILES[DEFAULT_PROFILE]["n_embd"] <= 64


@pytest.mark.parametrize("domain", RUNNABLE)
def test_preflight_accepts_all_bundled_domains(domain):
    from nanoscigpt.classroom import validate_domain_data

    report = validate_domain_data(domain, DATA_ROOT)

    assert report["status"] == "ready"
    assert report["domain"] == domain
    assert report["train_items"] > 0
    assert report["val_items"] > 0


def test_preflight_rejects_a_choice_when_a_manifest_file_is_missing(tmp_path):
    from nanoscigpt.classroom import validate_domain_data

    shutil.copytree(DATA_ROOT / "smiles", tmp_path / "smiles")
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "smiles" / "delaney-processed.csv").unlink()

    with pytest.raises(FileNotFoundError, match="delaney-processed.csv"):
        validate_domain_data("smiles", tmp_path)


@pytest.mark.parametrize("domain", RUNNABLE)
def test_cpu_smoke_run_finishes_for_every_student_choice(tmp_path, domain):
    command = [
        sys.executable,
        "-m",
        "nanoscigpt.classroom",
        "--domain",
        domain,
        "--profile",
        "smoke",
        "--data_root",
        str(DATA_ROOT),
        "--out_root",
        str(tmp_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    report_path = tmp_path / domain / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["device"] == "cpu"
    assert (tmp_path / domain / "model" / "ckpt.pt").is_file()
    assert "classroom run completed" in completed.stdout
    train_log = json.loads(
        (tmp_path / domain / "model" / "train_log.json").read_text(encoding="utf-8")
    )
    if domain in {"weather", "crystal", "structure3d", "image", "spectrum", "field"}:
        assert "pretrain_val_loss" in train_log
    task = json.loads(
        (tmp_path / domain / "downstream" / "downstream_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["downstream_task"] == "completed"
    assert task["status"] == "completed"
    assert task["teaching_only"] is True
    if domain == "text":
        assert task["task_name"] == "text punctuation-density teaching classification"
        assert task["label_source"] == "text-derived teaching label"


def test_classroom_profile_shows_the_expected_lesson_for_every_student_choice(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nanoscigpt.classroom",
            "--domain",
            "all",
            "--profile",
            "classroom",
            "--data_root",
            str(DATA_ROOT),
            "--out_root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    structured = {"weather", "crystal", "structure3d", "image", "spectrum", "field"}
    for domain in RUNNABLE:
        domain_dir = tmp_path / domain
        report = json.loads((domain_dir / "run_report.json").read_text(encoding="utf-8"))
        train_log = json.loads(
            (domain_dir / "model" / "train_log.json").read_text(encoding="utf-8")
        )
        task = json.loads(
            (domain_dir / "downstream" / "downstream_result.json").read_text(
                encoding="utf-8"
            )
        )

        assert report["status"] == "completed"
        assert report["profile"] == "classroom"
        assert report["preflight"]["status"] == "ready"
        assert (domain_dir / "model" / "ckpt.pt").is_file()
        assert task["status"] == "completed"
        assert task["teaching_only"] is True
        if domain in structured:
            assert train_log["pretrain_loss_end"] < train_log["pretrain_loss_start"]
            assert (domain_dir / "representation_preview.json").is_file()
        else:
            assert train_log["best_val_loss"] < train_log["history"][0]["val_loss"]
            assert (domain_dir / "model" / "samples.txt").is_file()


def test_classroom_list_only_offers_domains_that_pass_preflight():
    completed = subprocess.run(
        [sys.executable, "-m", "nanoscigpt.classroom", "--list", "--data_root", str(DATA_ROOT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    for domain in RUNNABLE:
        assert f"{domain}: ready" in completed.stdout


def test_text_lesson_keeps_training_history_and_runs_real_fine_tuning(tmp_path):
    import torch

    from nanoscigpt.classroom import run_domain

    report = run_domain("text", "smoke", DATA_ROOT, tmp_path, cwd=ROOT)
    model_dir = tmp_path / "text" / "model"
    train_log = json.loads((model_dir / "train_log.json").read_text(encoding="utf-8"))
    task = json.loads(
        (tmp_path / "text" / "downstream" / "downstream_result.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["lesson_stage"] == "nanogpt"
    assert len(train_log["history"]) >= 2
    assert {"iter", "train_loss", "val_loss"} <= set(train_log["history"][0])
    assert (model_dir / "samples.txt").is_file()
    assert report["artifacts"]["samples"].endswith("samples.txt")
    assert task["training_mode"] == "full_fine_tune"
    assert task["encoder_frozen"] is False
    assert task["pretrained_parameters_updated"] is True
    fine_path = tmp_path / "text" / "downstream" / "finetuned_ckpt.pt"
    assert fine_path.is_file()
    assert report["artifacts"]["finetuned_checkpoint"].endswith("finetuned_ckpt.pt")
    pretrained = torch.load(model_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    finetuned = torch.load(fine_path, map_location="cpu", weights_only=False)
    assert any(
        not torch.equal(pretrained["model"][name], finetuned["model"][name])
        for name in pretrained["model"]
    )


def test_scientific_sequence_lesson_keeps_the_frozen_probe_boundary(tmp_path):
    from nanoscigpt.classroom import run_domain

    run_domain("protein", "smoke", DATA_ROOT, tmp_path, cwd=ROOT)
    task = json.loads(
        (tmp_path / "protein" / "downstream" / "downstream_result.json").read_text(
            encoding="utf-8"
        )
    )

    assert task["training_mode"] == "frozen_probe"
    assert task["encoder_frozen"] is True
    assert task["pretrained_parameters_updated"] is False


def test_scientific_lesson_names_its_stage_and_frozen_encoder(tmp_path):
    from nanoscigpt.classroom import run_domain

    report = run_domain("weather", "smoke", DATA_ROOT, tmp_path, cwd=ROOT)
    task = json.loads(
        (tmp_path / "weather" / "downstream" / "downstream_result.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["lesson_stage"] == "nanoscigpt"
    assert task["encoder_frozen"] is True
    assert task["pretrained_parameters_updated"] is False


def test_classroom_refuses_to_silently_overwrite_a_finished_run(tmp_path):
    from nanoscigpt.classroom import run_domain

    finished = tmp_path / "protein" / "run_report.json"
    finished.parent.mkdir(parents=True)
    finished.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_domain("protein", "smoke", DATA_ROOT, tmp_path, cwd=ROOT)


@pytest.mark.parametrize(
    ("domain", "representation", "task_type"),
    [
        ("weather", "spatiotemporal_patches", "regression"),
        ("crystal", "periodic_graph", "regression"),
        ("structure3d", "pairwise_distance_tokens", "regression"),
        ("image", "image_patches", "regression"),
        ("spectrum", "wavelength_patches", "regression"),
        ("field", "space_time_patches", "regression"),
    ],
)
def test_structured_fixture_declares_representation_and_task(domain, representation, task_type):
    meta = json.loads((DATA_ROOT / domain / "meta.json").read_text(encoding="utf-8"))

    assert meta["representation"] == representation
    assert meta["task_type"] == task_type
    assert meta["teaching_only"] is True


def test_structured_fixture_generator_is_byte_reproducible(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        subprocess.run(
            [
                sys.executable,
                "scripts/build_structured_fixtures.py",
                "--data_root",
                str(root),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    for domain in ("weather", "crystal", "structure3d", "image", "spectrum", "field"):
        for name in ("fixture.npz", "meta.json"):
            assert (roots[0] / domain / name).read_bytes() == (roots[1] / domain / name).read_bytes()
