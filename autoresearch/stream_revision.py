"""Test one revision of the GD-1 tuning skill using the archived training evidence."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import stream_skill as s

DOC = Path(__file__).resolve().parents[1] / "docs/stream-skill"
DEV = range(27101, 27107)
BLIND = range(27201, 27213)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence(archive):
    result = []
    for r in range(1, 9):
        for i in (1, 2):
            seed = 100*r+i
            receipt = read(archive / "fits" / f"r{r}-train-{seed}.json")
            case = s.scenario(seed)
            episode = {"seed": seed, "truth": np.round(case["truth"], 3).tolist(), "batches": []}
            best = None
            for batch in range(4):
                hs = [h for h in receipt["history"] if h["call"] == batch]
                winner = min(hs, key=lambda h: sum(h["chi2"]))
                row = {"best_id": receipt["history"].index(winner), "theta": winner["theta"],
                       "chi2_components": np.round(winner["chi2"], 1).tolist()}
                if best is not None:
                    counts = [[0, 0] for _ in range(3)]
                    for h in hs:
                        n = int(np.count_nonzero(np.abs(np.subtract(h["theta"], best["theta"])) > 1e-8))
                        bucket = counts[min(n, 2)]
                        bucket[0] += int(sum(h["chi2"]) < sum(best["chi2"]))
                        bucket[1] += 1
                    row["improved_tried_repeat_single_joint"] = counts
                if best is None or sum(winner["chi2"]) < sum(best["chi2"]):
                    best = winner
                episode["batches"].append(row)
            result.append(episode)
    return result


def revised_skill(old, ledger, out):
    path = out / "candidate.json"
    identity = digest(old + json.dumps(ledger, sort_keys=True))
    if path.exists():
        record = read(path)
        if record["evidence_id"] != identity or digest(record["skill"]) != record["sha256"]:
            raise ValueError("Candidate does not match training evidence")
        return record["skill"]
    prompt = ("你要更新 GD-1 轨道拟合智能体的经验。顺序 vc,q,phi2,dist,pmphi1,pmphi2,vlos。"
              "下面是旧经验和八轮共16个训练场景。每批列最好候选及五项χ²，另统计和此前最优比较，重复、单参、多参的改善数/尝试数。"
              "请写一套下一次拟合可以执行的四批各十组的提议步骤，用新场景自身观测和各批结果定初值、步长和方向；"
              "先核对正反案例，删去证据薄弱的绝对参数范围。说明是否值得联合调整和何时回退。"
              "至少引用两个不同轮次的训练场景及候选编号，分清观察与推断。只返回 JSON，skill字段为不超过1500字的经验，changes字段简述理由。"
              "\n旧经验\n" + old + "\n训练证据\n" + json.dumps(ledger, ensure_ascii=False))
    for attempt in range(3):
        text, meta = s.ask_model(prompt, 0.5, max_tokens=3000)
        try:
            obj = json.loads(s.FENCE.sub("", text.strip()))
            skill = obj["skill"].strip()
            if not 0 < len(skill) <= 1500:
                raise ValueError("需为 1–1500 字")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            prompt += f"\n请修正输出格式或长度：{exc}"
            continue
        save(path, {"skill": skill, "changes": obj.get("changes"), "sha256": digest(skill),
                    "evidence_id": identity, "attempts": attempt+1, **meta})
        return skill
    raise RuntimeError("No valid update in three replies")


def phase(seeds, skills, out, workers):
    def fit(pair):
        seed, arm = pair
        case = s.scenario(seed)
        skill = skills[arm]
        tag = f"{seed}-{arm}-{digest(skill)[:12]}"
        fit = s.agent_fit(case, skill, out, tag, exact=True)
        if len(fit["history"]) != 40:
            raise ValueError(tag)
        print(f"completed {seed} {arm}", flush=True)
        return seed, arm, fit["gap"]

    jobs = [(seed, arm) for seed in seeds for arm in (list(skills) if seed % 2 else reversed(skills))]
    with ThreadPoolExecutor(workers) as executor:
        values = list(executor.map(fit, jobs))
    scores = {(seed, arm): val for seed, arm, val in values}
    return np.array([scores[seed,"new"]-scores[seed,"old"] for seed in seeds])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Rebuild the training evidence from original fit receipts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old = (DOC / "skill-r8.md").read_text(encoding="utf-8").strip()
    ledger = evidence(args.archive) if args.archive else read(DOC / "revision-evidence.json")
    save(args.out / "training-evidence.json", ledger)
    new = revised_skill(old, ledger, args.out)
    skills = {"old": old, "new": new}
    gate = args.out / "decision.json"
    delta = phase(DEV, skills, args.out, args.workers)
    decision = {"old_sha256": digest(old), "new_sha256": digest(new), "dev_delta": delta.tolist(),
                "accepted": bool(delta.mean() <= -0.10 and (delta < 0).sum() >= 4)}
    if gate.exists() and read(gate) != decision:
        raise ValueError("Existing decision differs")
    save(gate, decision)
    print("development:", decision["accepted"], flush=True)
    blind_delta = phase(BLIND, skills, args.out, args.workers)
    rows = []
    for seed in BLIND:
        case = s.scenario(seed)
        for arm in ("old", "new", "random", "simplex"):
            if arm in skills:
                fit = read(args.out / f"{seed}-{arm}-{digest(skills[arm])[:12]}.json")
            else:
                fit = (s.random_fit if arm == "random" else s.simplex_fit)(case)
            rows.append([seed, arm, *[fit["curve"][i] for i in (9, 19, 29, 39)]])
    with (args.out/"blind.csv").open("w", newline="", encoding="utf-8") as file:
        wr = csv.writer(file)
        wr.writerow(["seed","method","gap10","gap20","gap30","gap40"])
        wr.writerows(rows)
    rng = np.random.default_rng(271126)
    draws = rng.choice(blind_delta, (20000, len(blind_delta)), replace=True)
    low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
    wins = int((blind_delta < 0).sum())
    save(args.out / "confirmation.json", {
        "new_minus_old": float(blind_delta.mean()), "wins": wins, "n": len(blind_delta),
        "paired_95_interval": [float(low), float(high)],
        "adopt_revision": bool(decision["accepted"] and wins >= 8 and high < 0),
    })
    print("complete", flush=True)


if __name__ == "__main__":
    main()
