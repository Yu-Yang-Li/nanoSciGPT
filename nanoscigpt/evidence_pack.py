"""Export existing classroom evidence as a small, reviewable Markdown pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


DOMAIN_LABELS = {
    "text": "文本",
    "protein": "蛋白质序列",
    "dna": "DNA序列",
    "smiles": "分子字符串",
    "weather": "天气网格",
    "crystal": "晶体结构",
    "structure3d": "三维结构",
    "image": "科学图像",
    "spectrum": "光谱",
    "field": "连续物理场",
}
REPRESENTATION_LABELS = {
    "amino_acid_tokens": "氨基酸序列",
    "wavelength_patches": "按波长位置切分的光谱片段",
    "pairwise_distance_tokens": "点间距离",
    "periodic_graph": "周期原子图",
}
METRIC_LABELS = {
    "accuracy": "准确率",
    "mae": "平均绝对误差",
    "best_val_loss": "验证损失",
    "pretrain_val_loss": "验证损失",
}
STATUS_LABELS = {"completed": "已完成", "ready": "已准备", "failed": "失败", "skipped_no_labels": "无标签，未做监督任务"}
NEXT_ACTION_LABELS = {"stop": "停止", "conclude": "结束本轮"}


def _label(value: Any, labels: dict[str, str]) -> str:
    if value in (None, "未记录"):
        return "未记录"
    raw = str(value)
    translated = labels.get(raw)
    return f"{translated}（{raw}）" if translated else raw


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"记录必须是JSON对象：{path}")
    return value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _attachment_summary(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"- `{path.resolve()}`｜{path.stat().st_size}字节｜SHA256 `{digest.hexdigest()}`"


def _optional_json(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"找不到{label}：{path}")
    return _read_json(path)


def _render(
    run_report: dict[str, Any],
    comparison: dict[str, Any] | None,
    state: dict[str, Any] | None,
    downstream: dict[str, Any] | None,
    source_paths: list[Path],
    attachments: list[str] | None = None,
) -> str:
    preflight = run_report.get("preflight", {})
    domain = run_report.get("domain", "未记录")
    source_name = (downstream or {}).get("source") or preflight.get("source_name") or "未记录"
    task_name = (downstream or {}).get("task_name", preflight.get("task_name", "未记录"))
    representation = preflight.get("representation", "未记录")
    ran = run_report.get("status") == "completed"
    task_evaluated = bool(downstream and downstream.get("status") == "completed"
                          and downstream.get("metric_name") and _finite_number(downstream.get("metric_value")))
    task_compared = task_evaluated and _finite_number(downstream.get("metric_before_finetune"))
    comparison_evaluated = bool(comparison and comparison.get("primary_metric")
                                and _finite_number(comparison.get("baseline_value"))
                                and _finite_number(comparison.get("observed_delta")))
    evidence_identity = ("已评测" if task_evaluated or comparison_evaluated else "已运行") if ran else "未完成"
    source_label = (
        "教学数据"
        if "teaching" in str(source_name).lower()
        else "数据来源"
    )

    lines = [
        "# 最小证据包",
        "",
        f"- 证据身份：{evidence_identity}",
        f"- 领域：{_label(domain, DOMAIN_LABELS)}",
        f"- {source_label}：{source_name}",
        f"- 表示方式：{_label(representation, REPRESENTATION_LABELS)}",
        f"- 具体任务：{task_name}",
        f"- 运行状态：{_label(run_report.get('status'), STATUS_LABELS)}",
        f"- 具体任务状态：{_label((downstream or {}).get('status', run_report.get('downstream_task')), STATUS_LABELS)}",
    ]
    if task_evaluated:
        lines.append(
            f"- 具体任务评价：{_label(downstream.get('metric_name'), METRIC_LABELS)} = "
            f"{downstream.get('metric_value', '未记录')}"
        )
        before = downstream.get("metric_before_finetune")
        if _finite_number(before):
            after = downstream["metric_value"]
            delta = after - before
            lines.append(f"- 本轮微调记录：{before:g} → {after:g}（后减前：{delta:+.6g}）")
            if downstream["metric_name"] == "mae" and delta > 0:
                lines.append("- 本轮误差增大，未作为成功改进；记录原样保留。")
            elif downstream["metric_name"] == "accuracy" and delta < 0:
                lines.append("- 本轮准确率下降，未作为成功改进；记录原样保留。")
    lines.extend(["", "## 评价与下一步", ""])

    if not comparison_evaluated or not ran:
        lines.extend(
            [
                ("- 比较状态：已有本轮微调前后评价，未提供独立版本比较记录" if task_compared and ran else
                 "- 比较状态：尚无比较记录" if comparison is None else
                 "- 比较状态：记录不足或本次运行未完成，未认定比较已完成"),
                f"- 下一步决定：{_label((state or {}).get('next_action'), NEXT_ACTION_LABELS)}",
                "",
                "## 当前能够声称",
                "",
                ("- 训练流程已经运行；具体任务是否完成及其评价，以上述记录为准。" if ran else
                 "- 本次流程未完成；只能说明已有运行记录，不能据此声称训练完成。"),
                "",
                "## 当前不能声称",
                "",
                ("- 本轮微调前后数值不能代替独立对照，不能据此外推收益。" if task_compared else
                 "- 尚不能声称模型已经改善，因为没有提供同口径比较结果。"),
                "- 运行完成不等于结果已经经过外部验证。",
            ]
        )
    else:
        from_version = comparison.get("from_version", "V0")
        to_version = comparison.get("to_version", "V1")
        passed = bool(comparison.get("criterion_passed", False))
        reason = comparison.get("reason", "未记录")
        lines.extend(
            [
                f"- 版本：{from_version} → {to_version}",
                f"- 主要评价项：{_label(comparison.get('primary_metric'), METRIC_LABELS)}",
                f"- V0：{comparison.get('baseline_value', '未记录')}",
                f"- 变化：{comparison.get('observed_delta', '未记录')}",
                f"- 评价依据：{reason}",
                f"- 下一步决定：{_label((state or {}).get('next_action'), NEXT_ACTION_LABELS)}",
                "",
                "## 当前能够声称",
                "",
                "- 已完成一次保持数据、划分和评价方式不变的比较。",
                "- 本轮改动、评价结果和下一步决定已经留档。",
                "",
                "## 当前不能声称",
                "",
            ]
        )
        if passed:
            lines.append("- 达到课堂阈值仍不能替代独立数据、真实实验或外部复核。")
        else:
            lines.append("- 未达到预设改进阈值，不能声称本轮改动有效。")
        lines.append("- 教学数据上的结果不能直接外推到真实研究对象。")

    lines.extend(["", "## 来源文件", ""])
    lines.extend(f"- `{path.resolve()}`" for path in source_paths)
    if attachments:
        lines.extend(["", "## 原始附件", "", "只列出原文件及其摘要，不据附件数量判断研究是否完成。", ""])
        lines.extend(attachments)
        lines.extend(["", "原版Agent是否实际运行、实验是否改变下一步、稿件是否生成，需逐项核对原始日志；附件不自动获得新的证据等级。"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--downstream-result", type=Path, help="最新任务结果；不提供时沿用运行报告里的路径")
    parser.add_argument("--attachment", type=Path, action="append", default=[], help="保留原格式的日志、指标、代码差异或稿件；可重复提供")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        print(f"输出已存在，请换新路径以保留旧记录：{args.output}", file=sys.stderr)
        return 2

    if not args.run_report.is_file():
        print(f"找不到运行报告：{args.run_report}", file=sys.stderr)
        return 2

    try:
        run_report = _read_json(args.run_report)
        comparison = _optional_json(args.comparison, "比较结果")
        state = _optional_json(args.state, "研究状态")
        attachments = [_attachment_summary(path) for path in args.attachment]
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    downstream = None
    downstream_path = None
    reported_downstream = run_report.get("artifacts", {}).get("downstream")
    try:
        if args.downstream_result:
            downstream_path = args.downstream_result
            downstream = _optional_json(downstream_path, "具体任务结果")
        elif reported_downstream:
            downstream_path = Path(reported_downstream)
            if not downstream_path.is_absolute():
                downstream_path = args.run_report.parent / downstream_path
            if downstream_path.is_file():
                downstream = _read_json(downstream_path)
        if downstream and downstream.get("domain") not in (None, run_report.get("domain")):
            raise ValueError("具体任务结果与运行报告的领域不一致")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    source_paths = [args.run_report]
    source_paths.extend(path for path in (args.comparison, args.state) if path is not None)
    if downstream_path is not None and downstream is not None:
        source_paths.append(downstream_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(_render(run_report, comparison, state, downstream, source_paths, attachments))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
