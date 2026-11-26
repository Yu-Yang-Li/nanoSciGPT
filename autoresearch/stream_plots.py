"""Figures for the GD-1 orbit-fitting agent: orbits over the data, how the written skill changed, blind scores.

Everything is drawn from docs/stream-skill/rounds.json, the archived parameters of the eight-round run.
The orbits are re-integrated on CPU by autoresearch.stream_fit; no model interface is needed.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import stream_fit as sf
from .stream_skill import scenario

ARCHIVE = Path(__file__).resolve().parents[1] / "docs" / "stream-skill"
PANELS = (("phi2", "天上的横向位置 φ2（度）"), ("dist", "距离（kpc）"), ("pmphi1", "沿星流方向的自行（mas/yr）"),
          ("pmphi2", "垂直方向的自行（mas/yr）"), ("vlos", "视向速度（km/s）"))
# A rule counts as present in a round when its key phrase occurs in that round's skill text.
RULES = (("第一批围绕一个中心点撒点", "1组中心点"), ("第一批要分散开", "分散"), ("每组只动一个参数", "只动"),
         ("第 1 组复制当前最好的一组", "复制当前最优"), ("φ2 朝负方向一路扫到转折", "扫到转折"), ("距离要往两侧都试", "两侧扫描"),
         ("自行和视向速度成对微调", "成对微调"), ("迟迟没进展就大步移动", "5000"), ("把某个具体的距离值写成常用值", "常为佳"))


def archive():
    return json.loads((ARCHIVE / "rounds.json").read_text(encoding="utf-8"))


def _mean_gap(record):
    return float(np.mean([fit["gap"] for fit in record["blind"].values()]))


def _orbit(theta):
    """The part of the orbit that covers the data, or None when it misses the phi1 span."""
    return sf.segment(sf.track(np.asarray(theta, float)[None])[0])


def _frame(ax, table):
    low, high = table[:, 1].min() - table[:, 2].max(), table[:, 1].max() + table[:, 2].max()
    ax.set_xlim(-64, 4)
    ax.set_ylim(low - 0.25 * (high - low), high + 0.25 * (high - low))


def show_orbits(seed=9005, rounds=(0, 1, 2, 5, 8)):
    """One blind scenario: the best orbit found with the skill of each round, over the mock observations."""
    runs = archive()
    case = scenario(seed)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(rounds)))
    truth = _orbit(case["truth"])
    for column, (ax, (name, label)) in enumerate(zip(axes, PANELS)):
        table = case["data"][name]
        ax.errorbar(table[:, 0], table[:, 1], table[:, 2], fmt="o", color="black", ms=3, lw=0.8, zorder=3, label="模拟观测")
        ax.plot(truth[:, 0], truth[:, column + 1], color="black", lw=1, ls=":", label="生成数据用的真值轨道")
        for index, color in zip(rounds, colors):
            fit = runs["rounds"][index]["blind"][str(seed)]
            orbit = _orbit(fit["by_batch"][-1])
            if orbit is not None:
                ax.plot(orbit[:, 0], orbit[:, column + 1], color=color, lw=1.8,
                        label=f"第 {index} 轮经验，差距 {fit['gap']:.2f}")
        _frame(ax, table)
        ax.set_xlabel("沿星流的位置 φ1（度）")
        ax.set_title(label, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False, fontsize=9)
    fig.suptitle(f"盲测场景 {seed}：每一轮锁定的经验各拟合一次（40 次轨道评估），画出找到的最好轨道", fontsize=12)
    plt.show()


def show_one_fit(seed=9005, rounds=(0, 8)):
    """Inside one fit: the best orbit after each of the four submissions, without and with the final skill."""
    runs = archive()
    case = scenario(seed)
    fig, axes = plt.subplots(len(rounds), 3, figsize=(13, 3.1 * len(rounds)), layout="constrained", squeeze=False)
    colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, 4))
    for row, index in zip(axes, rounds):
        fit = runs["rounds"][index]["blind"][str(seed)]
        for ax, (name, label) in zip(row, (PANELS[0], PANELS[1], PANELS[4])):
            column = [p[0] for p in PANELS].index(name)
            table = case["data"][name]
            ax.errorbar(table[:, 0], table[:, 1], table[:, 2], fmt="o", color="black", ms=3, lw=0.8, zorder=3)
            for batch, (theta, color) in enumerate(zip(fit["by_batch"], colors)):
                orbit = None if theta is None else _orbit(theta)
                if orbit is not None:
                    ax.plot(orbit[:, 0], orbit[:, column + 1], color=color, lw=1.6, label=f"第 {batch + 1} 次提交后")
            _frame(ax, table)
            ax.set_title(f"第 {index} 轮经验 · {label}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("沿星流的位置 φ1（度）")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False, fontsize=9)
    fig.suptitle(f"一次拟合之内：每提交 10 组参数后的最好轨道（盲测场景 {seed}）", fontsize=12)
    plt.show()


def skill_text(round_index):
    """The locked skill of one round, as it was given to the model."""
    return (ARCHIVE / f"skill-r{round_index}.md").read_text(encoding="utf-8")


def show_skill_changes():
    """How the written skill was rewritten: which rules it holds in each round, its length, and its evidence."""
    rounds = archive()["rounds"]
    index = [r["round"] for r in rounds]
    present = np.array([[bool(re.search(pattern, skill_text(k))) for k in index[1:]] for _, pattern in RULES])
    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    ax.imshow(present, cmap="Greens", vmin=0, vmax=1.5, aspect="auto")
    ax.set_xticks(range(8), [f"第 {k} 轮" for k in index[1:]])
    ax.set_yticks(range(len(RULES)), [name for name, _ in RULES])
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RULES)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    ax.set_title("每一轮的经验里有哪些规则（绿色 = 写着这条规则；按关键词判断）", fontsize=11)
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), layout="constrained")
    axes[0].bar(index, [r["skill_chars"] for r in rounds], color="#0a6e78")
    axes[0].axhline(1500, color="grey", ls="--", lw=1, label="提示里要求的上限 1,500 字")
    axes[0].set_ylabel("经验的字数")
    axes[0].set_ylim(0, 2300)
    axes[0].set_title("经验的长度", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].bar(index[1:], [100 * r["kept_from_previous"] for r in rounds[1:]], color="#c46a2b")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("与上一轮相同的文字（%）")
    axes[1].set_title("每次改写保留了多少", fontsize=10)
    grid = np.zeros((8, 8))
    for r in rounds[1:]:
        for cited in r["cited_training_scenarios"]:
            grid[cited // 100 - 1, r["round"] - 1] = 1
    axes[2].imshow(grid, cmap="Greens", vmin=0, vmax=1.4, origin="lower")
    axes[2].set_xticks(range(8), range(1, 9))
    axes[2].set_yticks(range(8), range(1, 9))
    axes[2].set_xticks(np.arange(-0.5, 8), minor=True)
    axes[2].set_yticks(np.arange(-0.5, 8), minor=True)
    axes[2].grid(which="minor", color="white", lw=1.5)
    axes[2].tick_params(which="both", length=0)
    axes[2].set_xlabel("第几轮的经验")
    axes[2].set_ylabel("引用了第几轮的训练场景")
    axes[2].set_title("规则注明的依据来自哪一轮", fontsize=10)
    for ax in axes[:2]:
        ax.set_xticks(index)
        ax.set_xlabel("第几轮的经验（0 是空经验）")
    plt.show()
    for r in rounds[1:]:
        print(f"第 {r['round']} 轮改了什么：{r['changes'][:150]}{'……' if len(r['changes']) > 150 else ''}\n")


def show_blind_scores():
    """Blind-test score of the locked skill of every round, with the same-budget baselines."""
    runs = archive()
    seeds = [str(s) for s in runs["blind_seeds"]]
    gaps = np.array([[r["blind"][s]["gap"] for s in seeds] for r in runs["rounds"]])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(seeds), (4000, len(seeds)))
    low, high = np.percentile(gaps[:, draws].mean(2), [2.5, 97.5], axis=1)
    index = np.arange(len(gaps))
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    ax.fill_between(index, low, high, color="#0a6e78", alpha=0.15, label="95% 区间（对 8 个场景重抽样）")
    for row in gaps.T:
        ax.plot(index, row, color="#0a6e78", alpha=0.2, lw=0.8)
    ax.plot(index, gaps.mean(1), "o-", color="#0a6e78", lw=2.4, label="带着当轮经验去拟合")
    repeat = np.mean([runs["frozen_repeat"][s]["gap"] for s in seeds])
    ax.axhline(repeat, color="grey", ls="--", lw=1.2, label=f"空经验再测一次 {repeat:.2f}")
    for name, label, color, style in (("random", "同预算随机搜索", "#c46a2b", "-."), ("simplex", "同预算 Nelder–Mead", "#4f8a3c", ":")):
        value = np.mean([runs["baselines"][name][s]["gap"] for s in seeds])
        ax.axhline(value, color=color, ls=style, lw=1.6, label=f"{label} {value:.2f}")
    ax.set_xticks(index)
    ax.set_xlabel("经验更新到第几轮（0 是空经验）")
    ax.set_ylabel("盲测差距 ln(找到的最好 χ² / 真值的 χ²)，越低越好")
    ax.set_title("8 个没见过的场景上的成绩。细线是单个场景", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plt.show()
    best = int(gaps.mean(1).argmin())
    print(f"最好的是第 {best} 轮（{gaps.mean(1)[best]:.2f}），最后一轮是 {gaps.mean(1)[-1]:.2f}。")

