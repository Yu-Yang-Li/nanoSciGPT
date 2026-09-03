"""Export existing classroom evidence as a small, reviewable Markdown pack."""

from __future__ import annotations

import argparse
import json
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
STATUS_LABELS = {"completed": "已完成", "ready": "已准备"}
NEXT_ACTION_LABELS = {"stop": "停止", "conclude": "结束本轮"}


def _label(value: Any, labels: dict[str, str]) -> str:
    if value in (None, "未记录"):
        return "未记录"
    raw = str(value)
    translated = labels.get(raw)
    return f"{translated}（{raw}）" if translated else raw


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
) -> str:
    preflight = run_report.get("preflight", {})
    domain = run_report.get("domain", "未记录")
    source_name = preflight.get("source_name", "未记录")
    task_name = (downstream or {}).get("task_name", preflight.get("task_name", "未记录"))
    representation = preflight.get("representation", "未记录")
    evidence_identity = "已评测" if comparison is not None or downstream is not None else "已运行"
    source_label = (
        "教学数据"
        if run_report.get("profile") == "classroom" or "teaching" in str(source_name).lower()
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
        f"- 具体任务状态：{_label(run_report.get('downstream_task'), STATUS_LABELS)}",
    ]
    if downstream is not None:
        lines.append(
            f"- 具体任务评价：{_label(downstream.get('metric_name'), METRIC_LABELS)} = "
            f"{downstream.get('metric_value', '未记录')}"
        )
    lines.extend(["", "## 评价与下一步", ""])

    if comparison is None:
        lines.extend(
            [
                "- 比较状态：尚无比较记录",
                f"- 下一步决定：{_label((state or {}).get('next_action'), NEXT_ACTION_LABELS) if state else '先完成同口径比较'}",
                "",
                "## 当前能够声称",
                "",
                "- 训练流程和具体任务接口已经运行，并留下了运行报告。",
                "",
                "## 当前不能声称",
                "",
                "- 尚不能声称模型已经改善，因为没有提供同口径比较结果。",
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
    lines.extend(f"- `{path}`" for path in source_paths)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.run_report.is_file():
        print(f"找不到运行报告：{args.run_report}", file=sys.stderr)
        return 2

    try:
        run_report = _read_json(args.run_report)
        comparison = _optional_json(args.comparison, "比较结果")
        state = _optional_json(args.state, "研究状态")
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2

    downstream = None
    downstream_path = None
    reported_downstream = run_report.get("artifacts", {}).get("downstream")
    if reported_downstream:
        downstream_path = Path(reported_downstream)
        if not downstream_path.is_absolute():
            downstream_path = args.run_report.parent / downstream_path
        if downstream_path.is_file():
            downstream = _read_json(downstream_path)

    source_paths = [args.run_report]
    source_paths.extend(path for path in (args.comparison, args.state) if path is not None)
    if downstream_path is not None and downstream is not None:
        source_paths.append(downstream_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _render(run_report, comparison, state, downstream, source_paths),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
