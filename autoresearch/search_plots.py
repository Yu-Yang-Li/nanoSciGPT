"""Figures for the common search: one field's runs, and all eleven fields side by side."""
import json

import matplotlib.pyplot as plt
import numpy as np

from nanoscigpt.tasks import eleven

from . import common_search as search

COLORS = {"model": "#0a6e78", "random": "#c46a2b"}
NAMES = {"model": "DSK 提改动", "random": "随机搜索"}


def show_search(field, runs):
    """Left: best development error so far in every run. Right: holdout A of the chosen configurations."""
    spec = eleven.FIELDS[field]
    result = search.verdict(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", width_ratios=[1.5, 1])
    ax = axes[0]
    for run in runs:
        dev = np.array([t["dev"] for t in run["trials"]])
        color = COLORS[run["method"]]
        first = run["repeat"] == min(r["repeat"] for r in runs if r["method"] == run["method"])
        ax.plot(dev, "o", color=color, alpha=0.3, ms=4)
        ax.step(range(len(dev)), np.minimum.accumulate(dev), where="post", color=color, lw=2,
                label=f"{NAMES[run['method']]}：到这一轮为止的最好成绩" if first else None)
    guess, tried = runs[0]["trials"][0]["guess_mean_dev"], [t["dev"] for run in runs for t in run["trials"]]
    if guess < 1.6 * max(tried):  # far above every trial, the line would squeeze the search into the bottom
        ax.axhline(guess, color="black", ls="--", lw=1, label="只猜平均值")
    else:
        ax.plot([], [], " ", label=f"只猜平均值是 {guess:.4g}，在图的上方")
    ax.set_xlabel("第几轮（0 是起点，圆点是每一轮试的配置）")
    ax.set_ylabel("开发集上的平均绝对误差")
    ax.set_xticks(range(max(len(run["trials"]) for run in runs)))
    ax.set_title("搜索过程：只看开发集", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.axhline(result["start"], color="#8a8a8a", lw=2, label=f"起点 {result['start']:.4g}")
    methods = [m for m in ("model", "random") if m in result]
    for position, method in enumerate(methods):
        values = [r["confirmation"]["A"] for r in runs if r["method"] == method]
        ax.plot(np.linspace(position - 0.08, position + 0.08, len(values)), values, "o", color=COLORS[method], ms=7, alpha=0.6)
        ax.plot(position, result[method], "D", color=COLORS[method], ms=11, mec="black")
        ax.annotate(f"{result[method]:.4g}", (position, result[method]), xytext=(14, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.margins(y=0.35)
    ax.set_ylabel("留出组 A 上的平均绝对误差（越低越好）")
    ax.set_title(f"搜索结束后：选中的配置在留出组上\n圆点是每次运行，菱形是平均。只猜平均值是 {result['guess_mean']:.4g}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    conclusion = "；".join(result[key] for key in ("against_start", "against_random") if key in result)
    fig.suptitle(f"{spec['name']} · {spec['question']}" + (f"　结论：{conclusion}" if conclusion else ""), fontsize=12)
    plt.show()
    for run in runs:
        chosen = run["confirmation"]
        changed = {k: v for k, v in chosen["config"].items() if v != search.start(field)[k]}
        print(f"{NAMES[run['method']]} 第 {run['repeat']} 次：选中第 {chosen['round']} 轮，留出误差 {chosen['A']:.4g}，"
              f"相对起点改了 {json.dumps(changed, ensure_ascii=False) if changed else '无'}")


def show_all_fields():
    """All eleven fields: holdout error of the chosen configurations relative to the starting configuration."""
    fields = list(eleven.FIELDS)
    results = {field: search.verdict(search.archived(field)) for field in fields}
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    for row, field in enumerate(fields):
        for method, shift in (("model", 0.16), ("random", -0.16)):
            values = [r["confirmation"]["A"] / results[field]["start"] for r in search.archived(field) if r["method"] == method]
            ax.plot(values, [row + shift] * len(values), "o", color=COLORS[method], ms=5, alpha=0.5)
            ax.plot(np.mean(values), row + shift, "D", color=COLORS[method], ms=8,
                    label=f"{NAMES[method]}（菱形是两次运行的平均）" if row == 0 else None)
        ax.text(1.0, row, "  " + "；".join(results[field].get(k, "") for k in ("against_start", "against_random")),
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(1, color="#8a8a8a", lw=1.2, label="起点（V1 默认配置）")
    ax.set_yticks(range(len(fields)), [eleven.FIELDS[f]["name"] for f in fields])
    ax.invert_yaxis()
    ax.set_xlabel("留出组 A 的误差 ÷ 起点的误差（小于 1 表示比起点好）")
    ax.set_title("十一个方向用同一套搜索：选中的配置在留出组上的成绩", fontsize=11)
    fig.legend(loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    plt.show()
    print(f"{'方向':<6}{'起点':>10}{'DSK':>10}{'随机搜索':>10}{'只猜均值':>10}  结论")
    for field in fields:
        r = results[field]
        print(f"{eleven.FIELDS[field]['name']:<6}{r['start']:>10.4g}{r['model']:>10.4g}{r['random']:>10.4g}{r['guess_mean']:>10.4g}  "
              f"{r['against_start']}；{r['against_random']}")

