"""Carry search experience across fields: write a skill from some fields, lock it, search an unseen field with it.

The eleven fields are split into four folds. The skill of a fold is written by the model from the archived
searches of the other folds only (docs/autoresearch-eleven/results.json, runs 1 and 2). It is then locked and
added to the proposal prompt when searching the fields of that fold; everything else is autoresearch.common_search.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from autoresearch import common_search as search
from autoresearch.eleven_search import FENCE, ask_model
from nanoscigpt.tasks import eleven

FOLDS = {"A": ("smiles", "image", "dna"), "B": ("spectrum", "weather", "crystal"),
         "C": ("protein", "field", "rna"), "D": ("structure3d", "text")}
ARCHIVE = Path(__file__).resolve().parents[1] / "docs" / "search-skill"
KINDS = {"symbol": "符号序列", "patch": "数值块序列", "crystal": "晶胞（晶格加原子）"}
SKILL_LIMIT = 1500
EVIDENCE_REPEATS = (1, 2)


def fold_of(field):
    return next(name for name, members in FOLDS.items() if field in members)


def evidence(field):
    """What the skill writer sees of one training field: every round of its archived runs, relative to the start."""
    spec, default = search.searched(field), search.start(field)
    runs = [r for r in search.archived(field) if r["repeat"] in EVIDENCE_REPEATS and r["method"] in ("model", "random")]
    first = runs[0]["trials"][0]
    lines = [f"## {spec['name']}：{KINDS[spec['kind']]}，每条 {spec['positions']} 个 token，一个 token = {spec['token']}。任务：{spec['question']}",
             f"起点：readout={default['readout']}，pretrain_steps={default['pretrain_steps']}；起点的开发集误差是只猜均值的 {first['dev'] / first['guess_mean_dev']:.2f} 倍"]
    for run in runs:
        chosen = run["confirmation"]
        lines.append(f"{'DSK' if run['method'] == 'model' else '随机搜索'} 第 {run['repeat']} 次（选中第 {chosen['round']} 轮，留出误差是起点的 {chosen['A'] / chosen['start_A']:.3f} 倍）：")
        for trial in run["trials"][1:]:
            changed = {key: value for key, value in trial["config"].items() if value != default[key]}
            ratios = np.array(trial["dev_by_seed"]) / np.array(first["dev_by_seed"])
            lines.append(f"  第 {trial['round']} 轮 {json.dumps(changed, ensure_ascii=False)} → 开发集误差是起点的 {trial['dev'] / first['dev']:.3f} 倍"
                         f"（各种子 {ratios.min():.2f}–{ratios.max():.2f}）")
    return "\n".join(lines)


def write_prompt(fold):
    training = [field for field in eleven.FIELDS if field not in FOLDS[fold]]
    return "\n".join([
        "你在维护一份“怎样改进小型科学基座模型”的搜索经验。下一次会拿它去一个你没见过的科学方向上做同样的搜索，只有 8 轮，每轮试一个配置。",
        "模型主干固定为 2 层、2 头、32 维的 nanoGPT。先用 12,800 个无标签对象做自监督预训练，再用 32 个带标签对象训练下游读出。",
        "每一轮可以改的设置（pretrain_steps 的三个取值是起点的 1/4、1/2、1 倍）：",
        json.dumps(search.SHARED_SPACE, ensure_ascii=False),
        "含义：merge=2 把相邻两个 token 并成一个；objective=next 预测下一个 token，skip 预测隔一个的 token，masked 先把 mask_ratio 比例的输入 token 置零再预测下一个；"
        "readout=probe 冻结主干用岭回归读出（ridge_alpha），finetune 更新全部权重（finetune_steps、finetune_lr）；pool=mean 对全部 token 取平均，last 取最后一个 token。",
        "下面是在其他方向上做过的搜索的完整记录。每一轮写的是相对起点改了什么，以及开发集误差与起点之比（小于 1 表示比起点好）。"
        "每次运行最后选中开发集最好的一轮，在留出组上评一次。",
        *[evidence(field) for field in training],
        "请把这些记录总结成一份在新方向上可以直接照做的搜索经验：按输入类型（符号序列、数值块序列、晶胞）和起点的读出方式，前几轮先试什么；"
        "哪些改动通常没用或有害；8 轮怎样分配；什么时候应该停在起点。每条规则后用括号注明依据（哪个方向、几次里有几次）。"
        "只在一个方向上见过一次的现象，不要写成通用规则。",
        f"经验不超过 {SKILL_LIMIT} 字。只返回一个 JSON 对象：{{\"skill\": \"经验全文\"}}",
    ])


def write(fold, folder):
    """Ask the model for the skill of one fold, once, and lock it on disk. Returns the locked text."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"skill-{fold}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    prompt = write_prompt(fold)
    (folder / f"skill-{fold}.prompt.txt").write_text(prompt, encoding="utf-8")
    for _ in range(3):
        text, meta = ask_model(prompt, temperature=0.5, max_tokens=3000)
        try:
            skill = str(json.loads(FENCE.sub("", text.strip()))["skill"]).strip()
            if not skill or len(skill) > SKILL_LIMIT * 1.2:
                raise ValueError(f"经验有 {len(skill)} 字，超过了 {SKILL_LIMIT * 1.2:.0f} 字的上限。"
                                 f"请删掉依据最弱的规则，把全文压缩到 {SKILL_LIMIT} 字以内")
            break
        except (ValueError, KeyError, TypeError) as error:
            prompt += f"\n上一个回答没有通过检查：{error}。请只返回要求的 JSON。"
    else:
        raise RuntimeError("三次都没有得到合格的经验")
    path.write_text(skill, encoding="utf-8")
    record = {"fold": fold, "held_out": list(FOLDS[fold]), "sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
              "characters": len(skill), "model": meta.get("model"), "usage": meta.get("usage", {})}
    (folder / f"skill-{fold}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return skill


def locked(field, folder=ARCHIVE):
    """The locked skill that never saw this field."""
    return (Path(folder) / f"skill-{fold_of(field)}.md").read_text(encoding="utf-8")


def compare(ours, theirs):
    """Per-field log ratio of two methods' holdout errors, and the rule written before the study."""
    fields = [field for field in ours if field in theirs]
    d = np.array([np.log(ours[field] / theirs[field]) for field in fields])
    draws = np.random.default_rng(0).integers(0, len(d), (20000, len(d)))
    low, high = np.percentile(d[draws].mean(1), [2.5, 97.5])
    wins = int((d < 0).sum())
    if d.mean() < 0 and wins >= 8 and high < 0:
        conclusion = "有用"
    elif low > 0:
        conclusion = "反而更差"
    else:
        conclusion = "没有看到改进"
    return {"fields": fields, "d": d.tolist(), "mean": float(d.mean()), "wins": wins, "interval": [float(low), float(high)],
            "conclusion": conclusion}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", choices=list(FOLDS), help="write and lock the skill of this fold")
    parser.add_argument("--field", choices=list(eleven.FIELDS), help="search this field with the skill that never saw it")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--skills", type=Path, default=ARCHIVE, help="folder with the locked skills")
    parser.add_argument("--out", type=Path, default=Path("out/search"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.write:
        print(write(args.write, args.skills))
    if args.field:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        search.run(args.field, "skill", args.out, args.repeat, args.rounds, skill=locked(args.field, args.skills),
                   report=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()

