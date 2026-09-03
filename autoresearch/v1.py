"""CPU classroom reconstruction of the linear AI Scientist v1 workflow."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from nanoscigpt.domains.registry import RUNNABLE_DOMAINS, STRUCTURED_DOMAINS


ROOT = Path(__file__).resolve().parent.parent
LITERATURE = ROOT / "data" / "course" / "ai_scientist_v1_literature.json"
STRUCTURED_REPRESENTATION_ROUTES = {
    "weather": "spatiotemporal_patch_or_variable_grouping",
    "crystal": "neighbor_radius_or_graph_edges",
    "structure3d": "distance_or_angle_features",
    "image": "patch_size_or_augmentation",
    "spectrum": "wavelength_binning_or_normalization",
    "field": "spatial_resolution_or_boundary_encoding",
}
GENERATED_OUTPUTS = {
    "plan.json",
    "related_work.json",
    "results.json",
    "results.csv",
    "evidence_map.json",
    "draft.md",
    "review.json",
    "claim_boundary.md",
    "workflow_state.json",
    "workflow_status.json",
    "figures",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(
    out_dir: Path, *, plan_only: bool, overwrite: bool, plan: dict, sources: dict
) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return
    existing = {path.name for path in out_dir.iterdir()}
    if not existing:
        return

    unchanged_plan = False
    if existing <= {"plan.json", "related_work.json"} and existing == {
        "plan.json",
        "related_work.json",
    }:
        try:
            unchanged_plan = (
                read_json(out_dir / "plan.json") == plan
                and read_json(out_dir / "related_work.json") == sources
            )
        except (json.JSONDecodeError, OSError):
            unchanged_plan = False
    if not plan_only and unchanged_plan:
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory already contains workflow material: {out_dir}; "
            "use a new --out-dir or pass --overwrite"
        )

    for name in GENERATED_OUTPUTS:
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def load_inputs(domain: str, autoresearch_dir: Path) -> dict:
    paths = {
        "spec": autoresearch_dir / "iteration_spec.json",
        "comparison": autoresearch_dir / "comparison.json",
        "state": autoresearch_dir / "research_state.json",
        "candidate": autoresearch_dir / "candidate_run_report.json",
    }
    inputs = {"domain": domain, "spec_path": paths["spec"].resolve()}
    inputs["spec"] = read_json(paths["spec"])
    for name in ("comparison", "state", "candidate"):
        inputs[f"{name}_path"] = paths[name].resolve()
        inputs[name] = read_json(paths[name]) if paths[name].is_file() else None
    return inputs


def build_plan(inputs: dict) -> dict:
    spec = inputs["spec"]
    changed = spec.get("iteration", {}).get("changed")
    metric = spec.get("baseline", {}).get("primary_metric")
    if not isinstance(changed, dict):
        raise ValueError("iteration_spec.json has no single changed field")
    if not metric:
        raise ValueError("iteration_spec.json has no primary metric")
    return {
        "schema_version": "nanoscigpt.ai_scientist_v1.plan.v1",
        "implementation": {
            "name": "CPU classroom linear workflow",
            "inspired_by": "The AI Scientist v1",
            "reproduces_original_system": False,
        },
        "domain": inputs["domain"],
        "route_count": 1,
        "research_question": f"改变 {changed['field']} 是否改善 {metric}？",
        "route": {
            "id": "route-1",
            "status": "planned",
            "changed": changed,
            "fixed_arguments": spec["iteration"].get("fixed_arguments", {}),
            "baseline_run": spec.get("baseline", {}).get("report_path"),
            "candidate_run": str(inputs["candidate_path"]),
            "comparison": str(inputs["comparison_path"]),
        },
    }


def related_work(domain: str) -> dict:
    catalog = read_json(LITERATURE)
    return {
        "schema_version": "nanoscigpt.ai_scientist_v1.related_work.v1",
        "domain": domain,
        "novelty_assessment": "not_performed_offline",
        "sources": catalog["common"] + catalog["domains"][domain],
    }


def evidence_errors(inputs: dict) -> list[str]:
    comparison = inputs.get("comparison")
    candidate = inputs.get("candidate")
    state = inputs.get("state")
    missing = []
    if comparison is None:
        return ["comparison.json"]
    if comparison.get("evidence_level") != "evaluated":
        missing.append("comparison.evidence_level=evaluated")
    for key in ("baseline", "candidate", "delta", "threshold", "direction", "next_action"):
        if comparison.get(key) is None:
            missing.append(f"comparison.{key}")
    if candidate is None or candidate.get("status") != "completed":
        missing.append("candidate_run_report.status=completed")
    elif str(inputs["candidate_path"]) != comparison.get("candidate_run_report"):
        missing.append("comparison.candidate_run_report")
    if state is None or state.get("next_action") != comparison.get("next_action"):
        missing.append("research_state.next_action")
    return missing


def build_results(inputs: dict) -> dict:
    comparison = inputs["comparison"]
    return {
        "schema_version": "nanoscigpt.ai_scientist_v1.results.v1",
        "domain": inputs["domain"],
        "metric": comparison["primary_metric"],
        "baseline": comparison["baseline"],
        "candidate": comparison["candidate"],
        "delta": comparison["delta"],
        "threshold": comparison["threshold"],
        "direction": comparison["direction"],
        "criterion_passed": comparison["criterion_passed"],
        "next_action": comparison["next_action"],
        "evidence_level": comparison["evidence_level"],
    }


def write_results_csv(path: Path, results: dict) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results))
        writer.writeheader()
        writer.writerow(results)


def write_svg(path: Path, results: dict) -> None:
    values = [float(results["baseline"]), float(results["candidate"])]
    upper = max(values) or 1.0
    bars = []
    for x, label, value, color in zip(
        (180, 430), ("V0", "V1"), values, ("#35687a", "#d59a62")
    ):
        height = 220 * value / upper
        y = 300 - height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="120" height="{height:.2f}" fill="{color}"/>'
            f'<text x="{x + 60}" y="330" text-anchor="middle">{label}</text>'
            f'<text x="{x + 60}" y="{y - 10:.2f}" text-anchor="middle">{value:.4f}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="380" viewBox="0 0 720 380">'
        '<rect width="720" height="380" fill="#fbfaf6"/>'
        f'<text x="360" y="40" text-anchor="middle" font-size="22">{results["metric"]}</text>'
        '<line x1="100" y1="300" x2="620" y2="300" stroke="#47555a"/>'
        + "".join(bars)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def build_evidence_map(inputs: dict, results: dict) -> dict:
    source = str(inputs["comparison_path"])
    pointers = {"baseline": "/baseline", "candidate": "/candidate", "delta": "/delta", "threshold": "/threshold"}
    return {
        "schema_version": "nanoscigpt.ai_scientist_v1.evidence_map.v1",
        "claims": [
            {
                "claim_id": f"R{index}",
                "claim": f"{key}={results[key]}",
                "level": "evaluated",
                "source": source,
                "json_pointer": pointer,
                "value": results[key],
            }
            for index, (key, pointer) in enumerate(pointers.items(), start=1)
        ],
    }


def draft_text(plan: dict, results: dict, sources: dict) -> str:
    outcome = "达到预先设定的门槛，本路线保留，等待进一步复核。" if results["criterion_passed"] else "未达到预先设定的门槛，因此停止本路线并保留这条负结果。"
    cited = "；".join(source["title"] for source in sources["sources"])
    return f"""# AI Scientist v1 课堂研究短稿

## 研究问题

{plan['research_question']}

## 相关工作

离线课程目录提供了以下入口：{cited}。本次没有执行在线新颖性检索。

## 实验设置

只改变 `{plan['route']['changed']['field']}`：从 {plan['route']['changed']['from']} 调整为 {plan['route']['changed']['to']}；其余记录在 `plan.json`。

## 结果

V0 的 {results['metric']} 为 {results['baseline']:.4f}，V1 为 {results['candidate']:.4f}，按“V0减V1”计算的差值为 {results['delta']:.4f}，门槛为 {results['threshold']:.4f}。{outcome}

## 讨论

这次比较只回答当前数据、模型和评价设置下，增加训练预算是否带来足够变化。

## 局限与证据边界

结果来自课程规模的单次运行，尚未进行重复实验、外部数据验证或科学机制验证；稿件需要教师或研究者复核。
"""


def build_workflow_state(plan: dict, results: dict, inputs: dict) -> dict:
    changed = plan["route"]["changed"]
    domain = inputs["domain"]
    if domain in STRUCTURED_DOMAINS:
        alternate = {
            "id": "route-2",
            "status": "held",
            "execution_mode": "design_only",
            "design_reason": (
                "the current structured classroom command exposes only training budget; "
                "a second representation route must be implemented before execution"
            ),
            "change": {
                "field": "scientific_representation",
                "from": "bundled_baseline",
                "to": STRUCTURED_REPRESENTATION_ROUTES[domain],
            },
        }
    else:
        block_size = int(plan["route"]["fixed_arguments"]["block_size"])
        alternate = {
            "id": "route-2",
            "status": "held",
            "execution_mode": "executable",
            "change": {
                "field": "block_size",
                "from": block_size,
                "to": block_size * 2,
            },
        }
    return {
        "schema_version": "nanoscigpt.ai_scientist_v1.workflow.v1",
        "implementation": plan["implementation"],
        "status": "evaluated",
        "domain": domain,
        "route_count": 1,
        "research_question": plan["research_question"],
        "baseline_run": plan["route"]["baseline_run"],
        "evaluator": {
            "id": "pretrain_loss_gain.v1",
            "metric": results["metric"],
            "direction": results["direction"],
            "minimum_delta": results["threshold"],
        },
        "route": {"id": "route-1", "status": "completed", "change": changed, "run_report": str(inputs["candidate_path"]), "result": results},
        "candidate_backlog": [alternate],
    }


def complete_workflow(inputs: dict, out_dir: Path, plan: dict, sources: dict) -> int:
    missing = evidence_errors(inputs)
    if missing:
        write_json(out_dir / "workflow_status.json", {"status": "blocked_no_evaluated_evidence", "missing": missing})
        return 2
    results = build_results(inputs)
    write_json(out_dir / "results.json", results)
    write_results_csv(out_dir / "results.csv", results)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    write_svg(figures / "v0-v1.svg", results)
    evidence = build_evidence_map(inputs, results)
    write_json(out_dir / "evidence_map.json", evidence)
    draft = draft_text(plan, results, sources)
    (out_dir / "draft.md").write_text(draft, encoding="utf-8")
    (out_dir / "claim_boundary.md").write_text(
        "当前能够声称：完成了一次同口径的V0与V1比较。\n\n当前不能声称：发现了新机制、通过外部验证或复现了官方AI Scientist。\n",
        encoding="utf-8",
    )
    review = {
        "reviewer": "rule_based_classroom_evidence_audit",
        "official_v1_reviewer_reproduced": False,
        "checks": {
            "all_numbers_traced": len(evidence["claims"]) == 4,
            "single_change_preserved": bool(plan["route"]["changed"]),
            "negative_result_visible": results["criterion_passed"] or "未达到" in draft,
            "related_work_resolved": bool(sources["sources"]),
            "claim_boundary_present": True,
        },
    }
    review["verdict"] = "ready_for_human_review" if all(review["checks"].values()) else "needs_revision"
    write_json(out_dir / "review.json", review)
    write_json(out_dir / "workflow_state.json", build_workflow_state(plan, results, inputs))
    write_json(out_dir / "workflow_status.json", {"status": review["verdict"], "reproduces_original_system": False, "draft": str((out_dir / "draft.md").resolve()), "review": str((out_dir / "review.json").resolve())})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Scientist v1 classroom workflow")
    parser.add_argument("--domain", required=True, choices=RUNNABLE_DOMAINS)
    parser.add_argument("--autoresearch-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only files generated by this classroom workflow",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--confirm-plan", action="store_true")
    args = parser.parse_args()
    try:
        inputs = load_inputs(args.domain, args.autoresearch_dir.resolve())
        plan = build_plan(inputs)
        sources = related_work(args.domain)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    out_dir = args.out_dir.resolve()
    try:
        prepare_output_dir(
            out_dir,
            plan_only=args.plan_only,
            overwrite=args.overwrite,
            plan=plan,
            sources=sources,
        )
    except (FileExistsError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    write_json(out_dir / "plan.json", plan)
    write_json(out_dir / "related_work.json", sources)
    if args.plan_only:
        print(f"v1 plan -> {out_dir / 'plan.json'}")
        print("plan only: no draft was written")
        return 0
    result = complete_workflow(inputs, out_dir, plan, sources)
    print(f"v1 {'classroom workflow' if result == 0 else 'blocked'} -> {out_dir / 'workflow_status.json'}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
