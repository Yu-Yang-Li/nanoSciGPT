import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nanoscigpt.evidence_pack", *args],
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )


def test_exports_evaluated_run_without_inventing_a_scientific_claim(tmp_path: Path) -> None:
    run_report = _write_json(
        tmp_path / "run_report.json",
        {
            "status": "completed",
            "domain": "protein",
            "profile": "classroom",
            "device": "cpu",
            "preflight": {
                "source_name": "deterministic protein teaching fixture",
                "representation": "amino_acid_tokens",
                "task_name": "protein family teaching classification",
                "train_items": 450,
                "val_items": 50,
            },
            "downstream_task": "completed",
        },
    )
    comparison = _write_json(
        tmp_path / "comparison.json",
        {
            "from_version": "V0",
            "to_version": "V1",
            "primary_metric": "pretrain_val_loss",
            "baseline_value": 2.916258,
            "observed_delta": 0.0478,
            "criterion_passed": False,
            "reason": "gain +0.0478 vs required 0.05",
        },
    )
    state = _write_json(
        tmp_path / "research_state.json",
        {
            "evidence": [
                {
                    "level": "evaluated",
                    "passed": False,
                    "summary": "gain +0.0478 vs required 0.05",
                }
            ],
            "next_action": "stop",
        },
    )
    output = tmp_path / "evidence-pack.md"

    result = _run(
        "--run-report",
        str(run_report),
        "--comparison",
        str(comparison),
        "--state",
        str(state),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    for expected in (
        "证据身份：已评测",
        "领域：蛋白质序列（protein）",
        "教学数据：deterministic protein teaching fixture",
        "具体任务：protein family teaching classification",
        "V0 → V1",
        "gain +0.0478 vs required 0.05",
        "主要评价项：验证损失（pretrain_val_loss）",
        "评价依据：gain +0.0478 vs required 0.05",
        "下一步决定：停止（stop）",
        "当前能够声称",
        "当前不能声称",
        "未达到预设改进阈值",
    ):
        assert expected in text
    assert "科学发现" not in text


def test_run_without_comparison_is_marked_as_run_not_evaluated(tmp_path: Path) -> None:
    run_report = _write_json(
        tmp_path / "run_report.json",
        {
            "status": "completed",
            "domain": "spectrum",
            "preflight": {
                "source_name": "teaching spectrum fixture",
                "representation": "wavelength_patches",
                "task_name": "temperature teaching regression",
            },
            "downstream_task": "completed",
        },
    )
    output = tmp_path / "evidence-pack.md"

    result = _run("--run-report", str(run_report), "--output", str(output))

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "证据身份：已运行" in text
    assert "领域：光谱（spectrum）" in text
    assert "比较状态：尚无比较记录" in text
    assert "尚不能声称模型已经改善" in text


def test_passed_comparison_does_not_label_the_evidence_as_a_failure(tmp_path: Path) -> None:
    run_report = _write_json(
        tmp_path / "run_report.json",
        {
            "status": "completed",
            "domain": "protein",
            "profile": "classroom",
            "preflight": {"source_name": "teaching protein subset"},
            "downstream_task": "completed",
        },
    )
    comparison = _write_json(
        tmp_path / "comparison.json",
        {
            "from_version": "V0",
            "to_version": "V1",
            "primary_metric": "best_val_loss",
            "baseline_value": 3.0510,
            "observed_delta": 0.1826,
            "criterion_passed": True,
            "reason": "gain +0.1826 vs required 0.05",
        },
    )
    state = _write_json(tmp_path / "research_state.json", {"next_action": "conclude"})
    output = tmp_path / "evidence-pack.md"

    result = _run(
        "--run-report",
        str(run_report),
        "--comparison",
        str(comparison),
        "--state",
        str(state),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "评价与下一步" in text
    assert "评价依据：gain +0.1826 vs required 0.05" in text
    assert "下一步决定：结束本轮（conclude）" in text
    assert "失败/停止：gain +0.1826" not in text


def test_missing_run_report_stops_without_creating_a_pack(tmp_path: Path) -> None:
    output = tmp_path / "evidence-pack.md"

    result = _run(
        "--run-report",
        str(tmp_path / "missing.json"),
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert "找不到运行报告" in result.stderr
    assert not output.exists()


def test_reads_downstream_result_only_from_the_reported_artifact_path(tmp_path: Path) -> None:
    downstream = _write_json(
        tmp_path / "downstream_result.json",
        {
            "status": "completed",
            "task_name": "protein composition teaching classification",
            "metric_name": "accuracy",
            "metric_value": 0.4688,
            "teaching_only": True,
        },
    )
    run_report = _write_json(
        tmp_path / "run_report.json",
        {
            "status": "completed",
            "domain": "protein",
            "profile": "classroom",
            "preflight": {"source_name": "teaching protein subset"},
            "downstream_task": "completed",
            "artifacts": {"downstream": str(downstream)},
        },
    )
    output = tmp_path / "evidence-pack.md"

    result = _run("--run-report", str(run_report), "--output", str(output))

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "领域：蛋白质序列（protein）" in text
    assert "运行状态：已完成（completed）" in text
    assert "具体任务：protein composition teaching classification" in text
    assert "具体任务评价：准确率（accuracy） = 0.4688" in text
    assert str(downstream) in text


def test_unlabeled_run_is_not_promoted_by_a_skipped_downstream_file(tmp_path):
    task = _write_json(tmp_path / "task.json", {"status": "skipped_no_labels", "reason": "no target supplied"})
    report = _write_json(tmp_path / "run.json", {"status": "completed", "domain": "protein",
                          "downstream_task": "skipped_no_labels", "artifacts": {"downstream": str(task)}})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--output", str(output))
    assert result.returncode == 0, result.stderr
    content = output.read_text(encoding="utf-8")
    assert "证据身份：已运行" in content
    assert "证据身份：已评测" not in content
    assert "无标签" in content
    assert "具体任务评价：" not in content
    assert "训练流程和具体任务接口已经运行" not in content


def test_failed_training_report_does_not_claim_completed_training(tmp_path):
    report = _write_json(tmp_path / "run.json", {"status": "failed", "domain": "protein"})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--output", str(output))
    assert result.returncode == 0, result.stderr
    content = output.read_text(encoding="utf-8")
    assert "证据身份：未完成" in content
    assert "已经运行，并留下了运行报告" not in content


def test_latest_task_result_and_native_attachments_keep_original_files(tmp_path):
    import hashlib

    report = _write_json(tmp_path / "run.json", {"status": "completed", "domain": "protein"})
    task = _write_json(tmp_path / "finetune.json", {"status": "completed", "domain": "protein",
                         "task_name": "student activity regression", "metric_name": "mae", "metric_value": 3.75})
    original_log = tmp_path / "native-results.tsv"
    original_log.write_text("commit\tmetric\tstatus\naaa\t3.75\tdiscard\n", encoding="utf-8")
    digest = hashlib.sha256(original_log.read_bytes()).hexdigest()
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--downstream-result", str(task),
                  "--attachment", str(original_log), "--output", str(output))
    assert result.returncode == 0, result.stderr
    content = output.read_text(encoding="utf-8")
    assert "3.75" in content and "student activity regression" in content
    assert str(original_log) in content and digest in content
    assert "不据附件数量判断研究是否完成" in content
    assert hashlib.sha256(original_log.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_score_is_not_evaluated(tmp_path, metric):
    task = _write_json(tmp_path / "task.json", {"status": "completed", "metric_name": "mae", "metric_value": metric})
    report = _write_json(tmp_path / "run.json", {"status": "completed", "domain": "protein",
                          "artifacts": {"downstream": str(task)}})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--output", str(output))
    assert result.returncode == 0, result.stderr
    content = output.read_text(encoding="utf-8")
    assert "证据身份：已评测" not in content
    assert "具体任务评价：" not in content


def test_missing_attachment_and_existing_output_are_not_silently_ignored(tmp_path):
    report = _write_json(tmp_path / "run.json", {"status": "completed"})
    output = tmp_path / "pack.md"
    missing = _run("--run-report", str(report), "--attachment", str(tmp_path / "missing.tsv"), "--output", str(output))
    assert missing.returncode != 0 and not output.exists()
    output.write_text("student notes", encoding="utf-8")
    existing = _run("--run-report", str(report), "--output", str(output))
    assert existing.returncode != 0
    assert output.read_text(encoding="utf-8") == "student notes"


def test_attachment_alone_does_not_promote_execution_to_evaluation(tmp_path):
    report = _write_json(tmp_path / "run.json", {"status": "completed", "domain": "protein"})
    native = _write_json(tmp_path / "final_info.json", {"task": {"means": {"val_accuracy": 0.9}}})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--attachment", str(native), "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert "证据身份：已运行" in output.read_text(encoding="utf-8")


def test_task_for_another_domain_is_rejected(tmp_path):
    report = _write_json(tmp_path / "run.json", {"status": "completed", "domain": "protein"})
    task = _write_json(tmp_path / "task.json", {"status": "completed", "domain": "smiles",
                                               "metric_name": "mae", "metric_value": 1.2})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--downstream-result", str(task), "--output", str(output))
    assert result.returncode != 0 and not output.exists()


def test_invalid_task_json_is_reported_without_creating_a_pack(tmp_path):
    task = tmp_path / "task.json"
    task.write_text("[]", encoding="utf-8")
    report = _write_json(tmp_path / "run.json", {"status": "completed", "artifacts": {"downstream": str(task)}})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--output", str(output))
    assert result.returncode == 2 and not output.exists()


def test_worse_finetuning_result_is_visible_in_the_pack(tmp_path):
    report = _write_json(tmp_path / "run.json", {"status": "completed", "domain": "protein"})
    task = _write_json(tmp_path / "task.json", {"status": "completed", "metric_name": "mae",
                      "metric_before_finetune": 3.0, "metric_value": 4.0})
    output = tmp_path / "pack.md"
    result = _run("--run-report", str(report), "--downstream-result", str(task), "--output", str(output))
    assert result.returncode == 0, result.stderr
    content = output.read_text(encoding="utf-8")
    assert "3 → 4" in content
    assert "误差增大" in content
    assert "尚无比较记录" not in content
    assert "下一步决定：未记录" in content
