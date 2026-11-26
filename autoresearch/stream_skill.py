"""Skill accumulation for a parameter-fitting agent on the GD-1 orbit task (autoresearch.stream_fit).

Inside one fit, the model sees the data and every earlier evaluation and proposes the next batch of
parameter vectors (4 submissions x 10 vectors = 40 orbit evaluations). Across fits, the model keeps a
written skill: after each round it fits two new training scenarios with the current skill, then sees
their true parameters and a same-budget random search, and rewrites the skill with evidence from each
scenario. The skill is then locked and scored on eight fixed blind scenarios that the update step
never sees. Credentials come from SCNET_API_BASE, SCNET_API_KEY and SCNET_MODEL in the environment.
"""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from autoresearch import stream_fit as sf
from autoresearch.eleven_search import ask_model

CALLS, PER_CALL = 4, 10
ROUNDS = 8
BLIND_SEEDS = tuple(range(9001, 9009))
REFERENCE = np.array([216.0, 0.89, -0.99, 9.79, -8.92, -2.04, -255.0])  # large-budget fit to the real GD-1 data
SPREAD = np.array([30.0, 0.15, 0.3, 1.0, 1.0, 0.7, 40.0])
LOW = np.array([sf.PRIOR[n][0] for n in sf.NAMES])
HIGH = np.array([sf.PRIOR[n][1] for n in sf.NAMES])
FIT_TEMPERATURE, UPDATE_TEMPERATURE = 0.5, 0.5
SKILL_LIMIT = 1500
FENCE = re.compile(r"^\s*\x60{3}(?:json)?\s*|\s*\x60{3}\s*$")
OBSERVABLES = ("phi2", "dist", "pmphi1", "pmphi2", "vlos")


def scenario(seed):
    """A mock GD-1-like stream with a known truth."""
    rng = np.random.default_rng([seed, 1])
    while True:
        truth = REFERENCE + rng.uniform(-1, 1, 7) * SPREAD
        data = sf.mock(truth, seed)
        if data is not None:
            return {"seed": seed, "truth": truth.tolist(), "data": data,
                    "truth_chi2": float(sf.chi2(truth, data).sum())}


def gap(chi2_value, case):
    """How far a fit is from the truth's own chi-square: ln(best / truth)."""
    return float(np.log(chi2_value / case["truth_chi2"]))


def describe(data):
    lines = []
    for name in OBSERVABLES:
        table = data[name]
        lines.append(f"{name}: " + " ".join(f"({row[0]:g}, {row[1]:.2f}±{row[2]:g})" for row in table))
    return "\n".join(lines)


def fit_prompt(case, skill, history, call):
    box = ", ".join(f"{n} [{lo:g}, {hi:g}] {u}" for n, (lo, hi), u in zip(sf.NAMES, sf.PRIOR.values(), sf.UNITS))
    rows = [f"#{i} {json.dumps([round(v, 4) for v in h['theta']])} " +
            " ".join(f"{n}={c:.1f}" for n, c in zip(OBSERVABLES, h["chi2"])) + f" total={sum(h['chi2']):.1f}"
            for i, h in enumerate(history)]
    best = min(history, key=lambda h: sum(h["chi2"]))["theta"] if history else None
    return "\n".join([
        "你在拟合一条恒星星流（GD-1 类）的轨道模型。星流轨迹近似为前身天体的轨道，银河系势为对数晕势 Φ = vc²/2 · ln(R² + z²/q²)。",
        "前身天体位于星流坐标 φ1 = 0 处。要拟合 7 个参数，顺序和允许范围：" + box + "。",
        "依次是：圆速度 vc、晕的垂直扁率 q、前身天体的 φ2、日心距离、φ1 方向自行（已乘 cos φ2）、φ2 方向自行、视向速度。",
        "每组参数会积分轨道，在观测的 φ1 位置上和 5 类观测比较，给出每类的 χ²：phi2（星流在天上的横向位置，度）、dist（距离，kpc）、pmphi1、pmphi2（自行，mas/yr）、vlos（视向速度，km/s）。",
        "总 χ² 越小越好。轨道不能从 φ1 = -62° 单调扫到 +2° 时，每类 χ² 都记为 1000000。",
        "观测数据，每项是 (φ1, 测量值±误差)：",
        describe(case["data"]),
        "你积累的调参经验（请按它做）：" if skill else "你还没有积累调参经验。",
        skill,
        f"一共 {CALLS} 次提交，每次给 {PER_CALL} 组参数。现在是第 {call + 1} 次。",
        "已经评估过的参数（编号、参数、各类 χ²、总 χ²）：" if history else "还没有评估过任何参数。",
        *rows,
        f"目前最好的一组：{json.dumps(best)}" if best else "",
        f"只返回一个 JSON 对象：{{\"reason\": \"一两句话说明这批参数的打算\", \"candidates\": [[vc, q, phi2, dist, pmphi1, pmphi2, vlos], ...共 {PER_CALL} 组]}}",
    ])


def parse_candidates(text):
    value = json.loads(FENCE.sub("", text.strip()))
    rows = [np.asarray(c, float) for c in value["candidates"] if len(c) == 7]
    if not rows:
        raise ValueError("没有长度为 7 的参数组")
    return np.clip(np.array(rows[:PER_CALL]), LOW, HIGH), str(value.get("reason", ""))


def agent_fit(case, skill, folder, tag):
    """One fit by the model; cached on disk so an interrupted run resumes."""
    path = folder / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    history, calls = [], []
    for call in range(CALLS):
        prompt = fit_prompt(case, skill, history, call)
        for attempt in (1, 2):
            text, meta = ask_model(prompt, FIT_TEMPERATURE, max_tokens=1500)
            entry = {"call": call, "attempt": attempt, **meta}
            try:
                batch, reason = parse_candidates(text)
                entry["reason"] = reason
                calls.append(entry)
                break
            except (ValueError, KeyError, TypeError) as error:
                entry["error"] = str(error)
                calls.append(entry)
                prompt += f"\n上一个回答没有通过检查：{error}。请只返回要求的 JSON。"
        else:
            continue
        for theta, c in zip(batch, sf.chi2(batch, case["data"])):
            history.append({"call": call, "theta": theta.tolist(), "chi2": c.tolist()})
    result = {"tag": tag, "seed": case["seed"], "history": history, "calls": calls, **summarize(history, case)}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def summarize(history, case):
    totals = [sum(h["chi2"]) for h in history] or [5e6]
    best = int(np.argmin(totals))
    curve = np.minimum.accumulate(totals).tolist()
    return {"best_chi2": totals[best], "gap": gap(totals[best], case), "curve": [gap(v, case) for v in curve],
            "best_theta": history[best]["theta"] if history else None}


def random_fit(case):
    rng = np.random.default_rng([case["seed"], 7])
    batch = LOW + rng.uniform(0, 1, (CALLS * PER_CALL, 7)) * (HIGH - LOW)
    history = [{"theta": t.tolist(), "chi2": c.tolist()} for t, c in zip(batch, sf.chi2(batch, case["data"]))]
    return summarize(history, case)


def simplex_fit(case):
    """Nelder-Mead from the centre of the box, in coordinates scaled to [0, 1], same budget."""
    history = []

    def objective(u):
        theta = LOW + np.clip(u, 0, 1) * (HIGH - LOW)
        c = sf.chi2(theta, case["data"])[0]
        history.append({"theta": theta.tolist(), "chi2": c.tolist()})
        return c.sum()

    minimize(objective, np.full(7, 0.5), method="Nelder-Mead", options={"maxfev": CALLS * PER_CALL})
    return summarize(history[:CALLS * PER_CALL], case)


def ceiling_fit(case):
    result = differential_evolution(lambda x: sf.chi2(x.T, case["data"]).sum(1), list(zip(LOW, HIGH)), seed=1,
                                    maxiter=150, popsize=15, tol=1e-8, polish=False, vectorized=True)
    return {"best_chi2": float(result.fun), "gap": gap(float(result.fun), case), "best_theta": result.x.tolist()}


def update_prompt(skill, episodes, past):
    blocks = []
    for e in episodes:
        case, fit = e["case"], e["fit"]
        lines = [f"## 训练场景 {case['seed']}",
                 "真值：" + json.dumps(dict(zip(sf.NAMES, np.round(case["truth"], 3).tolist()))),
                 f"真值本身的总 χ² = {case['truth_chi2']:.1f}；你找到的最好总 χ² = {fit['best_chi2']:.1f}；"
                 f"同预算随机搜索 = {e['random']['best_chi2']:.1f}；结果：{'赢' if fit['best_chi2'] < e['random']['best_chi2'] else '输'}随机搜索",
                 "最好的一组：" + json.dumps(np.round(fit["best_theta"], 3).tolist() if fit["best_theta"] else None)]
        for c in fit["calls"]:
            if "reason" in c:
                lines.append(f"第 {c['call'] + 1} 次提交的打算：{c['reason']}")
        for i, h in enumerate(fit["history"]):
            lines.append(f"#{i} 第{h['call'] + 1}次 {json.dumps([round(v, 3) for v in h['theta']])} " +
                         " ".join(f"{n}={c:.1f}" for n, c in zip(OBSERVABLES, h["chi2"])))
        blocks.append("\n".join(lines))
    trend = [f"第 {p['round']} 轮训练：" + "，".join(f"场景 {g['seed']} 差距 {g['gap']:.2f}（随机 {g['random_gap']:.2f}）"
                                                for g in p["train"]) for p in past]
    return "\n".join([
        "你在维护一份恒星星流轨道拟合的调参经验。下一次会拿它去拟合新的模拟场景（参数真值不同），每次拟合只有 4 次提交、每次 10 组参数。",
        "拟合任务：对数晕势 Φ = vc²/2 · ln(R² + z²/q²)，前身天体在 φ1 = 0，7 个参数依次是 vc、q、phi2、dist、pmphi1、pmphi2、vlos，",
        "允许范围：" + json.dumps(sf.PRIOR) + "。观测有 phi2、dist、pmphi1、pmphi2、vlos 五类，各给出 χ²。",
        "差距 = ln(找到的最好 χ² / 真值的 χ²)，0 表示和真值一样好。",
        "当前经验：" if skill else "目前还没有经验。",
        skill,
        "以往各轮训练场景的差距：" if trend else "",
        *trend,
        "下面是用当前经验做的训练场景的完整记录：每次提交的打算、每组参数和各类 χ²，以及真值。",
        *blocks,
        f"请根据这些逐场景的证据改写经验：写成下一次拟合可以直接照做的步骤和规则，包括怎样从观测数据定初值、每次提交怎样分配 10 组参数、先调哪些参数。",
        "保留有证据支持的条目，改掉或删掉被证据否定的条目，每条规则后用括号注明依据（哪个场景、第几组）。",
        f"经验不超过 {SKILL_LIMIT} 字。只返回一个 JSON 对象：{{\"skill\": \"新的经验全文\", \"changes\": \"相对旧版改了什么、为什么\"}}",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fits = args.out / "fits"
    fits.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"rounds": []}

    def save():
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    blind = [scenario(s) for s in BLIND_SEEDS]
    if "baselines" not in state:
        state["baselines"] = {
            "truth_chi2": [c["truth_chi2"] for c in blind],
            "random": [random_fit(c) for c in blind],
            "simplex": [simplex_fit(c) for c in blind],
            "ceiling": [ceiling_fit(c) for c in blind]}
        save()

    def blind_test(skill, label):
        with ThreadPoolExecutor(args.workers) as pool:
            return list(pool.map(lambda c: agent_fit(c, skill, fits, f"{label}-blind-{c['seed']}"), blind))

    skill = ""
    if not state["rounds"]:
        results = blind_test(skill, "r0")
        state["rounds"].append({"round": 0, "skill": "", "sha256": hashlib.sha256(b"").hexdigest(),
                                "blind": [{"seed": r["seed"], "gap": r["gap"]} for r in results], "train": []})
        save()
    skill = state["rounds"][-1]["skill"]
    while len(state["rounds"]) <= ROUNDS:
        index = len(state["rounds"])
        train_cases = [scenario(100 * index + k) for k in (1, 2)]
        with ThreadPoolExecutor(2) as pool:
            train_fits = list(pool.map(lambda c: agent_fit(c, skill, fits, f"r{index}-train-{c['seed']}"), train_cases))
        episodes = [{"case": c, "fit": f, "random": random_fit(c)} for c, f in zip(train_cases, train_fits)]
        prompt = update_prompt(skill, episodes, state["rounds"][1:])
        (args.out / f"r{index}-update.prompt.txt").write_text(prompt, encoding="utf-8")
        for attempt in (1, 2, 3):
            text, meta = ask_model(prompt, UPDATE_TEMPERATURE, max_tokens=3000)
            (args.out / f"r{index}-update.response.txt").write_text(text, encoding="utf-8")
            try:
                value = json.loads(FENCE.sub("", text.strip()))
                new_skill, changes = str(value["skill"]).strip(), str(value.get("changes", ""))
                if not new_skill or len(new_skill) > SKILL_LIMIT * 1.2:
                    raise ValueError(f"经验长度 {len(new_skill)} 不在允许范围内")
                break
            except (ValueError, KeyError, TypeError) as error:
                prompt += f"\n上一个回答没有通过检查：{error}。请只返回要求的 JSON。"
        else:
            raise RuntimeError("经验更新三次都没有通过检查")
        sha = hashlib.sha256(new_skill.encode("utf-8")).hexdigest()
        (args.out / f"skill-r{index}.md").write_text(new_skill, encoding="utf-8")
        results = blind_test(new_skill, f"r{index}")
        state["rounds"].append({
            "round": index, "skill": new_skill, "sha256": sha, "changes": changes, "update_usage": meta.get("usage", {}),
            "train": [{"seed": e["case"]["seed"], "gap": e["fit"]["gap"], "random_gap": e["random"]["gap"]} for e in episodes],
            "blind": [{"seed": r["seed"], "gap": r["gap"]} for r in results]})
        save()
        skill = new_skill
        print("round", index, "blind mean gap", round(float(np.mean([r["gap"] for r in results])), 3), flush=True)
    if "frozen_repeat" not in state:
        state["frozen_repeat"] = [{"seed": r["seed"], "gap": r["gap"]} for r in blind_test("", "frozen")]
        save()
    if "real" not in state:
        real = {"seed": "GD-1", "data": sf.GD1_DATA, "truth_chi2": 1.0}
        state["real"] = {label: agent_fit(real, text, fits, f"real-{label}")["best_chi2"]
                         for label, text in (("empty", ""), ("final", skill))}
        state["real"]["ceiling"] = ceiling_fit(real)["best_chi2"]
        save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
