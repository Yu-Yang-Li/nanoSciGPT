"""Figures for the common search: one field's runs, all eleven fields, and the searches that carry a skill."""
import json

import matplotlib.pyplot as plt
import numpy as np

from nanoscigpt.style import INK, MAIN, MUTED, SECOND, THIRD, headline
from nanoscigpt.tasks import eleven

from . import common_search as search
from . import search_skill

COLORS = {"model": MAIN, "random": SECOND, "skill": THIRD}
NAMES = {"model": "DSK 提改动", "random": "随机搜索", "skill": "DSK 带着经验"}
# Fields where the archived searches found an improvement of at least 5% over the start, and those where they did not.
ROOM = ("spectrum", "protein", "smiles", "structure3d", "weather", "field", "image")
NO_ROOM = ("dna", "rna", "crystal", "text")


def holdout_ratio(runs, method):
    """Holdout error of each run's chosen configuration divided by the starting configuration's holdout error."""
    return [r["confirmation"]["A"] / r["confirmation"]["start_A"] for r in runs if r["method"] == method]


def best_so_far(run, rounds=8):
    """Best development error up to each round, relative to the start. A run that stopped early keeps its last value."""
    dev = [t["dev"] for t in run["trials"]]
    dev += [dev[-1]] * (rounds + 1 - len(dev))
    return np.minimum.accumulate(dev) / dev[0]


def show_search(field, runs):
    """Left: best development error so far in every run. Right: holdout A of the chosen configurations."""
    spec = eleven.FIELDS[field]
    result = search.verdict([r for r in runs if r["method"] != "skill"])
    methods = [m for m in ("model", "random", "skill") if any(r["method"] == m for r in runs)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[1.5, 1])
    ax = axes[0]
    for run in runs:
        dev = np.array([t["dev"] for t in run["trials"]])
        color = COLORS[run["method"]]
        first = run["repeat"] == min(r["repeat"] for r in runs if r["method"] == run["method"])
        ax.plot(dev, "o", color=color, alpha=0.3, ms=5)
        ax.step(range(len(dev)), np.minimum.accumulate(dev), where="post", color=color, lw=1.8, alpha=0.9,
                label=NAMES[run["method"]] if first else None)
    guess, tried = runs[0]["trials"][0]["guess_mean_dev"], [t["dev"] for run in runs for t in run["trials"]]
    if guess < 1.6 * max(tried):  # far above every trial, the line would squeeze the search into the bottom
        ax.axhline(guess, color=INK, ls="--", lw=1.4, label="只猜平均值")
    ax.set_xlabel("第几轮（0 是起点）")
    ax.set_ylabel("开发集上的平均绝对误差")
    ax.set_xticks(range(max(len(run["trials"]) for run in runs)))
    ax.set_title("搜索过程：圆点是每轮试的配置，线是到这一轮为止的最好成绩")
    ax.legend()

    ax = axes[1]
    ax.axhline(result["start"], color=MUTED, lw=2.4, label=f"起点 {result['start']:.3g}")
    means = {}
    for position, method in enumerate(methods):
        values = [r["confirmation"]["A"] for r in runs if r["method"] == method]
        means[method] = float(np.mean(values))
        ax.plot(np.linspace(position - 0.1, position + 0.1, len(values)), values, "o", color=COLORS[method], ms=8, alpha=0.55)
        ax.plot(position, means[method], "D", color=COLORS[method], ms=13, mec=INK)
        ax.annotate(f"{means[method]:.3g}", (position, means[method]), xytext=(16, 0), textcoords="offset points", va="center")
    ax.set_xticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.set_xlim(-0.6, len(methods) - 0.3)
    ax.margins(y=0.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("留出组上的平均绝对误差")
    ax.set_title("搜索结束后在留出组上：圆点是每次运行，菱形是平均")
    ax.legend()
    sentence = f"{spec['name']}：起点的留出误差 {result['start']:.3g}"
    if "model" in means:
        sentence += f"，自动研究后 {means['model']:.3g}"
    if "random" in means:
        sentence += f"，随机搜索 {means['random']:.3g}"
    if "skill" in means:
        sentence += f"，带着经验 {means['skill']:.3g}"
    verdicts = "；".join(result[key] for key in ("against_start", "against_random") if key in result)
    headline(fig, sentence + (f"。自动研究{verdicts}" if verdicts else "") + f"（只猜平均值 {result['guess_mean']:.3g}）")
    plt.show()
    for run in runs:
        chosen = run["confirmation"]
        changed = {k: v for k, v in chosen["config"].items() if v != search.start(field)[k]}
        print(f"{NAMES[run['method']]} 第 {run['repeat']} 次：选中第 {chosen['round']} 轮，留出误差 {chosen['A']:.4g}，"
              f"相对起点改了 {json.dumps(changed, ensure_ascii=False) if changed else '无'}")


def show_all_fields():
    """All eleven fields: holdout error of the chosen configurations relative to the starting configuration."""
    fields = list(eleven.FIELDS)
    runs = {field: [r for r in search.archived(field) if r["method"] != "skill"] for field in fields}
    results = {field: search.verdict(runs[field]) for field in fields}
    fig, ax = plt.subplots(figsize=(13, 7))
    for row, field in enumerate(fields):
        for method, shift in (("model", -0.17), ("random", 0.17)):
            values = holdout_ratio(runs[field], method)
            ax.plot(values, [row + shift] * len(values), "o", color=COLORS[method], ms=7, alpha=0.45)
            ax.plot(np.mean(values), row + shift, "D", color=COLORS[method], ms=11, mec=INK,
                    label=f"{NAMES[method]}（菱形是各次运行的平均）" if row == 0 else None)
        ax.text(1.01, row, results[field]["against_random"].replace("和随机搜索", ""), transform=ax.get_yaxis_transform(),
                va="center", fontsize=11, color=MAIN if results[field]["against_random"] == "好于随机搜索" else INK)
    ax.axvline(1, color=MUTED, lw=2, label="起点")
    ax.set_yticks(range(len(fields)), [eleven.FIELDS[f]["name"] for f in fields])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("留出组的误差 ÷ 起点的误差（小于 1 表示比起点好）")
    fig.legend(loc="outside lower center", ncols=3)
    counts = {key: sum(r["against_random"] == key for r in results.values())
              for key in ("好于随机搜索", "和随机搜索分不出高下", "没有跑赢随机搜索")}
    better = sum(r["against_start"] == "好于起点" for r in results.values())
    headline(fig, f"自动研究在 {better} 个方向上每次运行都好于起点；和同预算的随机搜索比，{counts['好于随机搜索']} 个方向更好，"
                  f"{counts['和随机搜索分不出高下']} 个分不出高下，{counts['没有跑赢随机搜索']} 个没有跑赢")
    plt.show()
    print(f"{'方向':<6}{'起点':>10}{'DSK':>10}{'随机搜索':>10}{'只猜均值':>10}  运行次数  结论")
    for field in fields:
        r = results[field]
        print(f"{eleven.FIELDS[field]['name']:<6}{r['start']:>10.4g}{r['model']:>10.4g}{r['random']:>10.4g}{r['guess_mean']:>10.4g}"
              f"  {len(holdout_ratio(runs[field], 'model')):^8}  {r['against_start']}；{r['against_random']}")


def skill_summary():
    """Per-field mean holdout ratio of each method, and the comparisons defined before the study."""
    fields = list(eleven.FIELDS)
    ratio = {method: {f: float(np.mean(holdout_ratio(search.archived(f), method))) for f in fields}
             for method in ("skill", "model", "random")}
    return {"ratio": ratio, "against_model": search_skill.compare(ratio["skill"], ratio["model"]),
            "against_random": search_skill.compare(ratio["skill"], ratio["random"])}


def show_skill_transfer():
    """Does a skill written from other fields help on a field it never saw? Per field, and round by round."""
    summary = skill_summary()
    fields = [*ROOM, *NO_ROOM]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), width_ratios=[1.25, 1])
    ax = axes[0]
    for row, field in enumerate(fields):
        for method, shift in (("skill", -0.24), ("model", 0), ("random", 0.24)):
            values = holdout_ratio(search.archived(field), method)
            ax.plot(values, [row + shift] * len(values), "o", color=COLORS[method], ms=6, alpha=0.4)
            ax.plot(np.mean(values), row + shift, "D", color=COLORS[method], ms=10, mec=INK, label=NAMES[method] if row == 0 else None)
    ax.axvline(1, color=MUTED, lw=2, label="起点")
    if ROOM and NO_ROOM:
        ax.axhline(len(ROOM) - 0.5, color=MUTED, lw=1, ls="--")
    groups = {f: "有改进空间" if f in ROOM else "空间很小" for f in fields}
    ax.set_yticks(range(len(fields)), [f"{eleven.FIELDS[f]['name']}（{search_skill.fold_of(f)} 折，{groups[f]}）" for f in fields])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("留出组的误差 ÷ 起点的误差")
    ax.set_title("没见过的方向上，搜索结束后的留出成绩")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower left", ncols=4)

    ax = axes[1]
    for method in ("skill", "model", "random"):
        curves = []
        for field in eleven.FIELDS:
            runs = [r for r in search.archived(field) if r["method"] == method]
            best = [best_so_far(r) for r in runs]
            curves.append(np.mean(best, axis=0))
        ax.plot(np.mean(curves, axis=0), "o-", color=COLORS[method], label=NAMES[method])
    ax.set_xlabel("第几轮（0 是起点）")
    ax.set_ylabel("到这一轮为止最好的开发集误差 ÷ 起点")
    ax.set_title("搜索得快不快：十一个方向的平均")
    ax.legend()

    against = summary["against_model"]
    words = {"有用": "按事先写定的标准，经验有用", "反而更差": "按事先写定的标准，带着经验反而更差",
             "没有看到改进": "按事先写定的标准，没有看到经验带来的改进"}[against["conclusion"]]
    change = np.exp(against["mean"]) - 1
    headline(fig, f"{words}：带着经验的留出误差平均比不带经验{'低' if change < 0 else '高'} {abs(change):.1%}，"
                  f"11 个方向里 {against['wins']} 个更好")
    plt.show()
    for name, group in (("有改进空间的 7 个方向", ROOM), ("改进空间很小的 4 个方向", NO_ROOM)):
        if not group:
            continue
        d = [against["d"][against["fields"].index(f)] for f in group]
        print(f"{name}：带经验比不带经验平均{'低' if np.mean(d) < 0 else '高'} {abs(np.exp(np.mean(d)) - 1):.1%}，"
              f"{sum(v < 0 for v in d)} 个方向更好")
    other = summary["against_random"]
    print(f"和随机搜索比：平均{'低' if other['mean'] < 0 else '高'} {abs(np.exp(other['mean']) - 1):.1%}，"
          f"{other['wins']} 个方向更好，结论：{other['conclusion']}")

