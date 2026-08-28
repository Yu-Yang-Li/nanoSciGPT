"""A3 final: evidence-based model route decision.

Teaching point: the course's core claim - "没有证据支持训练新模型时，必须
降级路线". This module walks the decision chain: does the data support
pretraining? do two tasks share structure? does transfer beat scratch?
Students get a concrete artifact (decision.json) instead of vibes.
"""

import argparse
import json
from pathlib import Path


QUESTIONS = [
    {
        "id": "data_scale",
        "question": "无标签数据量是否达到万级以上（或领域公认预训练规模）？",
        "if_no": "数据不足，预训练收益不可靠；回到专用模型路线。",
    },
    {
        "id": "task_sharing",
        "question": "是否有至少两个任务共享同一对象表示（同一 tokenizer/图构造/场网格）？",
        "if_no": "单任务场景，基座复用无从谈起；继续专用模型。",
    },
    {
        "id": "transfer_evidence",
        "question": "冻结表征 + 探针是否优于 one-hot/随机初始化基线（A2 实验已跑）？",
        "if_no": "迁移收益未证实；不升级为基座路线。",
    },
    {
        "id": "multi_task_gain",
        "question": "共享编码器多任务是否优于各任务单独训练（A3 实验已跑）？",
        "if_no": "多任务无增益，统一接口不成立；保持任务专用头。",
    },
    {
        "id": "budget",
        "question": "算力/时间/许可是否允许维护一个共享基座（而非 N 个专用模型）？",
        "if_no": "维护成本不支持；调用现成基座而非自训。",
    },
]


def decide(answers):
    """answers: dict of question_id -> bool. Returns route + reasoning."""
    failed = [q for q in QUESTIONS if not answers.get(q["id"], False)]
    if not failed:
        return {
            "route": "train_new_foundation",
            "reasoning": "数据规模、任务共享、迁移证据、多任务增益、维护预算全部满足，可训练新基座。",
        }
    first = failed[0]
    routes = {
        "data_scale": ("use_specialized_model", "数据规模不足，预训练不可靠。"),
        "task_sharing": ("use_specialized_model", "任务不共享表示结构，基座无复用面。"),
        "transfer_evidence": ("adapt_existing_foundation", "迁移未证实，先适配现成基座再评估自训。"),
        "multi_task_gain": ("use_specialized_model", "多任务无增益，统一接口不成立。"),
        "budget": ("call_existing_foundation", "维护预算不足，调用现成模型 API/权重。"),
    }
    route, why = routes[first["id"]]
    return {
        "route": route,
        "reasoning": f"第一处失败：{first['question']} {first['if_no']} {why}",
        "failed_checks": [f["id"] for f in failed],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out/route_decision")
    p.add_argument("--answers", default=None, help="JSON dict of {question_id: bool}; omit for interactive")
    args = p.parse_args()

    if args.answers:
        answers = json.loads(args.answers)
    else:
        answers = {}
        print("A线最终判断：逐题回答 y/n（直接回车视为 n）\n")
        for q in QUESTIONS:
            a = input(f"{q['question']} [y/n]: ").strip().lower()
            answers[q["id"]] = a == "y"

    result = decide(answers)
    print(f"\n路线: {result['route']}")
    print(f"依据: {result['reasoning']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "decision.json", "w") as f:
        json.dump({"answers": answers, "decision": result}, f, indent=2)
    print(f"-> {out/'decision.json'}")


if __name__ == "__main__":
    main()
