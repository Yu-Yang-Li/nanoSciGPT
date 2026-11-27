"""Figures for the GD-1 orbit-fitting agent: orbits over the data, how the written skill changed, blind scores.

Everything is drawn from docs/stream-skill/rounds.json, the archived parameters of the eight-round run.
The orbits are re-integrated on CPU by autoresearch.stream_fit; no model interface is needed.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from nanoscigpt.style import GOOD, INK, MAIN, MUTED, PANEL, SECOND, SEQUENCE, headline

from . import stream_fit as sf
from .stream_skill import scenario

ARCHIVE = Path(__file__).resolve().parents[1] / "docs" / "stream-skill"
PANELS = (("phi2", "横向位置 φ2（度）"), ("dist", "距离（kpc）"), ("pmphi1", "沿星流的自行（mas/yr）"),
          ("pmphi2", "垂直方向的自行（mas/yr）"), ("vlos", "视向速度（km/s）"))
# A rule counts as present in a round when its key phrase occurs in that round's skill text.
RULES = (("第一批围绕一个中心点撒点", "1组中心点"), ("第一批要分散开", "分散"), ("每组只动一个参数", "只动"),
         ("第 1 组复制当前最好的一组", "复制当前最优"), ("φ2 朝负方向一路扫到转折", "扫到转折"), ("距离要往两侧都试", "两侧扫描"),
         ("自行和视向速度成对微调", "成对微调"), ("迟迟没进展就大步移动", "5000"), ("把某个具体的距离值写成常用值", "常为佳"))


def archive():
    return json.loads((ARCHIVE / "rounds.json").read_text(encoding="utf-8"))


def skill_text(round_index):
    """The locked skill of one round, as it was given to the model."""
    return (ARCHIVE / f"skill-r{round_index}.md").read_text(encoding="utf-8")


def _orbit(theta):
    """The part of the orbit that covers the data, or None when it misses the phi1 span."""
    return sf.segment(sf.track(np.asarray(theta, float)[None])[0])


def _observations(ax, table):
    ax.errorbar(table[:, 0], table[:, 1], table[:, 2], fmt="o", color=INK, ms=4, lw=1, zorder=3, label="模拟观测")
    low, high = table[:, 1].min() - table[:, 2].max(), table[:, 1].max() + table[:, 2].max()
    ax.set_xlim(-64, 4)
    ax.set_ylim(low - 0.25 * (high - low), high + 0.25 * (high - low))


def show_orbits(seed=9005, rounds=(0, 1, 2, 5, 8)):
    """One blind scenario: the best orbit found with the skill of each round, over the mock observations."""
    runs = archive()
    case = scenario(seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    axes = axes.ravel()
    colors = plt.get_cmap(SEQUENCE)(np.linspace(0.0, 0.88, len(rounds)))
    truth = _orbit(case["truth"])
    for column, (ax, (name, label)) in enumerate(zip(axes, PANELS)):
        _observations(ax, case["data"][name])
        ax.plot(truth[:, 0], truth[:, column + 1], color=INK, lw=1.4, ls=":", label="生成数据用的真值轨道")
        for index, color in zip(rounds, colors):
            fit = runs["rounds"][index]["blind"][str(seed)]
            orbit = _orbit(fit["by_batch"][-1])
            if orbit is not None:
                ax.plot(orbit[:, 0], orbit[:, column + 1], color=color, label=f"第 {index} 轮经验：差距 {fit['gap']:.2f}")
        ax.set_xlabel("沿星流的位置 φ1（度）")
        ax.set_title(label)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    axes[5].legend(handles, labels, loc="center left", fontsize=12, title="每一轮锁定的经验各拟合一次")
    gaps = {index: runs["rounds"][index]["blind"][str(seed)]["gap"] for index in rounds}
    best = min(gaps, key=gaps.get)
    headline(fig, f"盲测场景 {seed}：空经验找到的轨道离观测点很远（差距 {gaps[rounds[0]]:.2f}），"
                  f"带着第 {best} 轮经验的轨道贴得最近（差距 {gaps[best]:.2f}）")
    plt.show()


def show_one_fit(seed=9005, rounds=(0, 8)):
    """Inside one fit: the best orbit after each of the four submissions, without and with the final skill."""
    runs = archive()
    case = scenario(seed)
    fig, axes = plt.subplots(len(rounds), 3, figsize=(15, 4 * len(rounds)), squeeze=False)
    colors = plt.get_cmap(SEQUENCE)(np.linspace(0.0, 0.88, 4))
    for row, index in zip(axes, rounds):
        fit = runs["rounds"][index]["blind"][str(seed)]
        for ax, (name, label) in zip(row, (PANELS[0], PANELS[1], PANELS[4])):
            column = [p[0] for p in PANELS].index(name)
            _observations(ax, case["data"][name])
            for batch, (theta, color) in enumerate(zip(fit["by_batch"], colors)):
                orbit = None if theta is None else _orbit(theta)
                if orbit is not None:
                    ax.plot(orbit[:, 0], orbit[:, column + 1], color=color, label=f"第 {batch + 1} 次提交后")
            ax.set_title(f"{'空经验' if index == 0 else f'第 {index} 轮经验'} · {label}")
    for ax in axes[-1]:
        ax.set_xlabel("沿星流的位置 φ1（度）")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5)
    def gaps(index):  # distance from the truth after each submission, from the archived best parameters
        thetas = runs["rounds"][index]["blind"][str(seed)]["by_batch"]
        return [np.log(sf.chi2(theta, case["data"])[0].sum() / case["truth_chi2"]) for theta in thetas]

    first, last = gaps(rounds[0]), gaps(rounds[-1])
    headline(fig, f"一次拟合有 4 次提交、每次 10 组参数：空经验每次提交后的差距是 {'、'.join(f'{g:.2f}' for g in first)}，"
                  f"带着第 {rounds[-1]} 轮经验是 {'、'.join(f'{g:.2f}' for g in last)}")
    plt.show()


def show_skill_changes():
    """How the written skill was rewritten: which rules it holds in each round, its length, and its evidence."""
    rounds = archive()["rounds"]
    index = [r["round"] for r in rounds]
    present = np.array([[bool(re.search(pattern, skill_text(k))) for k in index[1:]] for _, pattern in RULES])
    cells = ListedColormap([PANEL, GOOD])
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.imshow(present, cmap=cells, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(8), [f"第 {k} 轮" for k in index[1:]])
    ax.set_yticks(range(len(RULES)), [name for name, _ in RULES])
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RULES)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", lw=3)
    ax.tick_params(which="both", length=0)
    kept = int(present.all(1).sum())
    dropped = index[1:][int(np.argmin(present[0]))]      # the first round without the "one centre" rule
    added = index[1:][int(np.argmax(present[1]))]        # the first round with the "spread out" rule
    headline(fig, f"九条规则里 {kept} 条从第 1 轮留到最后；“{RULES[0][0]}”第 {dropped} 轮被删掉，"
                  f"“{RULES[1][0]}”第 {added} 轮加进来。绿色表示那一轮的经验里写着这条规则")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), width_ratios=[1, 1, 0.8])
    lengths = [r["skill_chars"] for r in rounds]
    axes[0].bar(index, lengths, color=MAIN)
    axes[0].axhline(1500, color=INK, ls="--", lw=1.4, label="提示里要求的上限 1,500 字")
    axes[0].set_ylabel("经验的字数")
    axes[0].set_ylim(0, 2300)
    axes[0].set_title("经验有多长")
    axes[0].legend(loc="upper left")
    kept_share = [100 * r["kept_from_previous"] for r in rounds[1:]]
    axes[1].bar(index[1:], kept_share, color=SECOND)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("与上一轮相同的文字（%）")
    axes[1].set_title("每次改写保留了多少")
    grid = np.zeros((8, 8))
    for r in rounds[1:]:
        for cited in r["cited_training_scenarios"]:
            grid[cited // 100 - 1, r["round"] - 1] = 1
    axes[2].imshow(grid, cmap=cells, vmin=0, vmax=1, origin="lower")
    axes[2].set_xticks(range(8), range(1, 9))
    axes[2].set_yticks(range(8), range(1, 9))
    axes[2].set_xticks(np.arange(-0.5, 8), minor=True)
    axes[2].set_yticks(np.arange(-0.5, 8), minor=True)
    axes[2].grid(False)
    axes[2].grid(which="minor", color="white", lw=2)
    axes[2].tick_params(which="both", length=0)
    axes[2].set_xlabel("第几轮的经验")
    axes[2].set_ylabel("引用了第几轮的训练场景")
    axes[2].set_title("规则的依据来自哪一轮")
    for ax in axes[:2]:
        ax.set_xticks(index)
        ax.set_xlabel("第几轮的经验（0 是空经验）")
        ax.grid(axis="x", visible=False)
    headline(fig, f"经验从第 2 轮起就写到了 {min(lengths[2:]):,} 到 {max(lengths[2:]):,} 字，在上限附近；"
                  f"第 5 轮以后每次改写保留 {min(kept_share[4:]):.0f}% 到 {max(kept_share[4:]):.0f}% 的文字")
    plt.show()
    for r in rounds[1:]:
        print(f"第 {r['round']} 轮改了什么：{r['changes'][:150]}{'……' if len(r['changes']) > 150 else ''}\n")


def show_blind_scores():
    """Blind-test score of the locked skill of every round, with the same-budget baselines."""
    runs = archive()
    seeds = [str(s) for s in runs["blind_seeds"]]
    gaps = np.array([[r["blind"][s]["gap"] for s in seeds] for r in runs["rounds"]])
    draws = np.random.default_rng(0).integers(0, len(seeds), (4000, len(seeds)))
    low, high = np.percentile(gaps[:, draws].mean(2), [2.5, 97.5], axis=1)
    index = np.arange(len(gaps))
    means = gaps.mean(1)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.fill_between(index, low, high, color=MAIN, alpha=0.15, label="95% 区间（对 8 个场景重抽样）")
    for row in gaps.T:
        ax.plot(index, row, color=MAIN, alpha=0.2, lw=1)
    ax.plot(index, means, "o-", color=MAIN, lw=2.8, label="带着当轮经验去拟合")
    repeat = np.mean([runs["frozen_repeat"][s]["gap"] for s in seeds])
    ax.axhline(repeat, color=MUTED, ls="--", label=f"空经验再测一次 {repeat:.2f}")
    baseline = {}
    for name, label, color, line in (("random", "同预算随机搜索", SECOND, "-."), ("simplex", "同预算 Nelder–Mead", INK, ":")):
        baseline[name] = np.mean([runs["baselines"][name][s]["gap"] for s in seeds])
        ax.axhline(baseline[name], color=color, ls=line, label=f"{label} {baseline[name]:.2f}")
    ax.set_xticks(index)
    ax.set_xlabel("经验更新到第几轮（0 是空经验）")
    ax.set_ylabel("盲测差距 ln(找到的最好 χ² / 真值的 χ²)")
    ax.legend(ncols=2, loc="upper right")
    best = int(means.argmin())
    headline(fig, f"带着经验比空经验好，但没有一轮比一轮好下去：盲测差距从 {means[0]:.2f} 降到第 {best} 轮的 {means[best]:.2f}，"
                  f"之后回升到 {means[-1]:.2f}，和同预算的 Nelder–Mead（{baseline['simplex']:.2f}）差不多")
    plt.show()

