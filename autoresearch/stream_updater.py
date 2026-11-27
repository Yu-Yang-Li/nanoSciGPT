"""A gated, model-edited search rule for the GD-1 fitting exercise."""

import argparse
import csv
from functools import lru_cache
import itertools
import json
from pathlib import Path

import numpy as np

from . import stream_fit as sf
from . import stream_skill as s
from .stream_revision import read, save

DAMPING = (1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0)
RADIUS = (0.03, 0.08, 0.2, 0.5, 1.0)
DIFF = (0.0001, 0.001, 0.01, 0.05, 0.15)
SHAPE = np.array([len(DAMPING), len(RADIUS), len(DIFF)])
GRID = list(itertools.product(*(range(int(n)) for n in SHAPE)))
START = (4, 2, 2)
FIXED = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]]
NODES = tuple(sf.GD1_DATA)
SIGMA = np.concatenate([sf.GD1_DATA[key][:, 2] for key in NODES])
WIDTH = s.HIGH - s.LOW
CHECK = range(40101, 40141, 2)


@lru_cache(maxsize=4096)
def observables(point):
    return observe_batch(np.array([point]))[0]


def observe_batch(points):
    tracks = sf.track(s.LOW + np.asarray(points) * WIDTH)
    result = np.full((len(points), len(SIGMA)), np.nan)
    for i, track in enumerate(tracks):
        part = sf.segment(track)
        if part is not None:
            result[i] = np.concatenate(
                [np.interp(sf.GD1_DATA[key][:, 0], part[:, 0], part[:, j + 1]) for j, key in enumerate(NODES)]
            )
    return result


@lru_cache(maxsize=None)
def case(seed):
    scene = s.scenario(seed)
    return np.concatenate([scene["data"][k][:, 1] for k in NODES]), scene["truth_chi2"]


def fit(data, setting):
    """Forty charged orbit evaluations. Cache only shared deterministic forward calculations."""
    damp, radius, eps = (axis[i] for axis, i in zip((DAMPING, RADIUS, DIFF), setting))
    x = np.full(7, 0.5)
    residual = (observables(tuple(x)) - data) / SIGMA
    best = float(residual @ residual)
    for iteration in range(3):
        dx = np.where(x + eps <= 1, eps, -eps)
        probes = x + np.diag(dx)
        values = (
            np.array([observables(tuple(p)) for p in probes]) if iteration == 0 else observe_batch(probes)
        )
        r = (values - data) / SIGMA
        jac = np.nan_to_num((r - residual).T / dx)
        scale = np.maximum(np.linalg.norm(jac, axis=0), 1e-8)
        j = jac / scale
        direction = np.linalg.solve(j.T @ j + damp * np.eye(7), -j.T @ residual) / scale
        direction *= min(1.0, radius / max(np.abs(direction).max(), 1e-12))
        steps = np.clip(x + np.array([1.0, 0.5, 0.25, 1.5, 0.75, 0.125])[:, None] * direction, 0.0, 1.0)
        residuals = np.vstack([r, (observe_batch(steps) - data) / SIGMA])
        points = np.vstack([probes, steps])
        scores = np.where(np.isfinite(residuals).all(1), (residuals**2).sum(1), np.inf)
        winner = int(scores.argmin())
        if scores[winner] < best:
            best = float(scores[winner])
            x, residual = points[winner], residuals[winner]
    return best, x


@lru_cache(maxsize=None)
def score(setting, seed):
    y, truth_chi2 = case(seed)
    return float(np.log(fit(y, setting)[0] / truth_chi2))


def tune(rule, seed):
    tried = [START]
    best = START
    if rule == "random":
        rng = np.random.default_rng([seed, 401126])
        options = [g for g in GRID if g != START]
        random_configs = [options[i] for i in rng.choice(len(options), 4, replace=False)]
    for i in range(4):
        if rule == "random":
            proposed = random_configs[i]
        else:
            aim = np.clip(np.array(best) + rule[i], 0, SHAPE - 1)
            # Resolve repeats identically for every rule; charge four distinct trials.
            available = [g for g in GRID if g not in tried]
            proposed = min(available, key=lambda g: sum((int(a) - int(b)) ** 2 for a, b in zip(g, aim)))
        tried.append(proposed)
        if score(proposed, seed) < score(best, seed):
            best = proposed
    return best, [{"config": g, "gap": round(score(g, seed), 5)} for g in tried]


def assess(rule, seeds):
    results = []
    for seed in seeds:
        chosen, tried = tune(rule, seed)
        results.append(
            {
                "tune": seed,
                "test": seed + 1,
                "trials": tried,
                "chosen": chosen,
                "gap": score(chosen, seed + 1),
            }
        )
    return results


def valid(rule):
    return (
        isinstance(rule, list)
        and len(rule) == 4
        and all(
            isinstance(row, list)
            and len(row) == 3
            and any(row)
            and all(type(x) is int and abs(x) <= 6 for x in row)
            for row in rule
        )
    )


def propose(rule, history, evidence, path):
    if path.exists():
        return read(path)
    prompt = (
        "你是GD-1星流拟合智能体的自改进层。你直接改写生成下一次试验的更新规则。"
        "每个新任务从同一档位 [4,2,2] 出发。执行四条位移指令，每条相对于当时调试星流上最佳设置，"
        "给阻尼/信任半径/差分步长的档位加整数，越界截断，重复改为最近未试项。"
        "阻尼档=" + str(DAMPING) + "，步幅档=" + str(RADIUS) + "，差分步长档=" + str(DIFF) + "。"
        "每组参数均给同一个三步Gauss-Newton拟合器40次轨道积分。指标越低越好。"
        "你的目标是更新后的规则能在不曾见过的任务中用同样四次候选机会学得更好。"
        "请据不同场景的实测差距找出关键轴、方向和适合联合移动的尺度，解释一次失败或证据限制。"
        "可以提出新的跨档、多轴指令。输出JSON含rule（4×3个整数，每行非零，单项绝对值≤6）和reason。"
        "\n当前规则 "
        + json.dumps(rule)
        + "\n往轮门槛反馈 "
        + json.dumps(history, ensure_ascii=False)
        + "\n本轮训练试跑（当前规则和同预算随机探索） "
        + json.dumps(evidence, ensure_ascii=False)
    )
    for attempt in range(1, 4):
        reply, meta = s.ask_model(prompt, 0.3, max_tokens=1600)
        try:
            obj = json.loads(s.FENCE.sub("", reply.strip()))
            if not valid(obj["rule"]) or not isinstance(obj["reason"], str):
                raise ValueError("规则格式无效")
        except (ValueError, KeyError, TypeError):
            prompt += "\n请校验JSON字段、长度、整数范围。"
            continue
        result = {"rule": obj["rule"], "reason": obj["reason"], "attempts": attempt, "model": meta["model"]}
        save(path, result)
        return result
    raise ValueError("Three invalid proposals")


def decision(rows, history):
    panel = {}
    for tune_seed, test_seed, arm, *_, gap in rows:
        if test_seed != tune_seed + 1 or tune_seed in panel.setdefault(arm, {}):
            raise ValueError("Invalid paired confirmation results")
        panel[arm][tune_seed] = gap
    curves = {arm: np.array([values[seed] for seed in CHECK]) for arm, values in panel.items()}
    means = {arm: float(values.mean()) for arm, values in curves.items()}
    indices = np.random.default_rng(401126).integers(0, len(CHECK), (20000, len(CHECK)))
    comparisons = {}
    for arm in ("r0", "random", "static", "start", "r1"):
        delta = curves["r4"] - curves[arm]
        comparisons[arm] = {
            "mean": float(delta.mean()),
            "wins": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "interval": np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist(),
        }
    passed = means["r4"] < means["start"] and all(
        comparisons[arm]["mean"] <= -0.02
        and comparisons[arm]["wins"] >= 14
        and comparisons[arm]["interval"][1] < 0
        for arm in ("r0", "random")
    )
    accepted = [h["round"] for h in history if h["accepted"]]
    later_better = len(accepted) >= 2 and means["r4"] < means[f"r{accepted[0]}"]
    return {
        "means": means,
        "comparisons": comparisons,
        "passed": passed,
        "accepted_rounds": accepted,
        "later_better_than_first_accepted": later_better,
        "primary_and_progress": passed and later_better,
    }


def run(out, replay=False):
    out.mkdir(parents=True, exist_ok=True)
    recorded = (
        read(Path(__file__).resolve().parents[1] / "docs/stream-skill/updater-proposals.json")
        if replay
        else None
    )
    rules = [FIXED]
    history = []
    configs = {START}
    for rnd in range(1, 5):
        current = rules[-1]
        training = range(38100 + 10 * rnd, 38106 + 10 * rnd, 2)
        gate = range(39100 + 10 * rnd, 39106 + 10 * rnd, 2)
        evidence = {k: assess(v, training) for k, v in (("current", current), ("random", "random"))}
        proposal_path = out / f"proposal-{rnd}.json"
        if recorded is None:
            proposal = propose(current, history, evidence, proposal_path)
        else:
            proposal = recorded[rnd - 1]
            if proposal_path.exists() and read(proposal_path) != proposal:
                raise ValueError("Replay conflicts with an existing proposal in the output directory")
            if not valid(proposal["rule"]):
                raise ValueError("Invalid archived rule")
            save(proposal_path, proposal)
        incumbent = assess(current, gate)
        challenger = assess(proposal["rule"], gate)
        for group in [*evidence.values(), incumbent, challenger]:
            configs.update(tuple(t["config"]) for row in group for t in row["trials"])
        delta = np.array([a["gap"] - b["gap"] for a, b in zip(challenger, incumbent)])
        accepted = bool(delta.mean() < -0.002 and (delta < 0).sum() >= 2)
        history.append(
            {
                "round": rnd,
                "prior": current,
                "proposed": proposal["rule"],
                "delta": delta.tolist(),
                "accepted": accepted,
            }
        )
        rules.append(proposal["rule"] if accepted else current)
        save(
            out / f"dev-{rnd}.json",
            {"evidence": evidence, "current": incumbent, "challenger": challenger, "decision": history[-1]},
        )
        print("round", rnd, "accepted", accepted, "mean", round(float(delta.mean()), 4), flush=True)
    static = min(sorted(configs), key=lambda g: sum(score(g, s) for s in (39501, 39502, 39503)))
    lock = {"rules": rules, "static": list(static), "history": history, "spent_configs": len(configs)}
    path = out / "lock.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Development lock mismatch")
    save(path, lock)
    rows = []
    for seed in CHECK:
        for label, rule in [(f"r{i}", v) for i, v in enumerate(rules)] + [
            ("random", "random"),
            ("static", None),
            ("start", None),
        ]:
            chosen = tuple(static) if label == "static" else START
            if rule is not None:
                chosen, _ = tune(rule, seed)
            rows.append([seed, seed + 1, label, *chosen, score(chosen, seed + 1)])
        print("checked pair", seed, flush=True)
    with (out / "blind.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["tune_seed", "test_seed", "arm", "damping_index", "radius_index", "difference_index", "gap"]
        )
        writer.writerows(rows)
    save(out / "verdict.json", decision(rows, history))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--replay", action="store_true", help="Recompute the fits with the archived DSK edits, offline"
    )
    args = parser.parse_args()
    run(args.out, replay=args.replay)
