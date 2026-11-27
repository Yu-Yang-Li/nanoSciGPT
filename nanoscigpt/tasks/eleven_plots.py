"""Figures for notebooks/eleven_fields.ipynb: what the data look like, how they become tokens, how training goes.

Every figure uses nanoscigpt.style and carries a one-sentence headline computed from the numbers it shows.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from .. import style
from ..style import INK, MAIN, MUTED, PANEL, SECOND, SEQUENCE, headline, plain
from . import eleven


def use_chinese_font():
    """Apply the shared figure style; returns the Chinese font in use, or None when none could be found."""
    return style.use()


def _strings(field, tokens):
    table = eleven.symbols(field)
    return ["".join(table[t] for t in row if t) for row in np.asarray(tokens)]


def _grid(patches, side, block):
    """Tokens of block x block pixels back to a side x side picture."""
    count = side // block
    return np.asarray(patches).reshape(count, count, block, block).transpose(0, 2, 1, 3).reshape(side, side)


def _weather_frames(patches):
    """16 tokens of 4 frames x 4 x 4 pixels back to four 16 x 16 frames."""
    return np.asarray(patches).reshape(4, 4, 4, 4, 4).transpose(2, 0, 3, 1, 4).reshape(4, 16, 16)


def _field_frames(patches):
    """16 tokens of 4 frames x 4 positions back to four frames of 64 positions."""
    return np.asarray(patches).reshape(16, 4, 4).transpose(1, 0, 2).reshape(4, 64)


def _token_edges(ax, side, block):
    for edge in range(block, side, block):
        ax.axhline(edge - 0.5, color="white", lw=0.9)
        ax.axvline(edge - 0.5, color="white", lw=0.9)


def _draw(field, ax, sample, tokens=False):
    """Draw one object. With tokens=True, also mark where one token ends and the next begins."""
    spec = eleven.FIELDS[field]
    if spec["kind"] == "symbol":
        text = _strings(field, [sample["tokens"]])[0]
        plain(ax)
        ax.set_frame_on(False)
        shown = " ".join(text) if tokens else text
        width = 36 if tokens else 70
        wrapped = "\n".join(shown[i:i + width] for i in range(0, len(shown), width))
        ax.text(0.0, 0.5, wrapped, family="monospace", fontsize=12, va="center", transform=ax.transAxes)
    elif field == "crystal":
        table = eleven.symbols(field)
        elements = [table[e] for e in np.asarray(sample["elements"])]
        xyz = np.asarray(sample["fractional"], dtype=float)
        depth = np.array([0.45, 0.3])  # the cell is drawn as a cube seen from the side: z shifts a point up and right
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        ax.plot(*corners.T, color=MUTED, lw=1)
        ax.plot(*(corners + depth).T, color=MUTED, lw=1)
        for corner in corners[:4]:
            ax.plot(*np.array([corner, corner + depth]).T, color=MUTED, lw=1)
        flat = xyz[:, :2] + xyz[:, 2:] * depth
        ax.scatter(flat[:, 0], flat[:, 1], s=520, color=[SECOND, MAIN, PANEL, PANEL, PANEL], edgecolor=INK, zorder=3)
        for index, (name, (x, y)) in enumerate(zip(elements, flat)):
            ax.text(x, y, name, ha="center", va="center", fontsize=11, zorder=4, color="white" if index < 2 else INK)
        ax.set_xlim(-0.2, 1.65)
        ax.set_ylim(-0.2, 1.5)
        ax.set_aspect("equal")
        plain(ax)
        ax.set_frame_on(False)
    elif field == "spectrum":
        ax.plot(np.linspace(350, 1000, 64), np.asarray(sample["patches"]).ravel(), color=MAIN)
        if tokens:
            for edge in np.linspace(350, 1000, 64)[4::4]:
                ax.axvline(edge - 5, color=MUTED, lw=0.6)
        ax.set_xlabel("波长（nm）")
        ax.set_ylabel("通量")
    elif field == "field":
        colors = plt.get_cmap(SEQUENCE)(np.linspace(0.05, 0.85, 4))
        for index, frame in enumerate(_field_frames(sample["patches"])):
            ax.plot(frame, color=colors[index], lw=1.8, label=f"第 {index + 1} 帧")
        if tokens:
            for edge in range(4, 64, 4):
                ax.axvline(edge - 0.5, color=MUTED, lw=0.6)
        ax.set_xlabel("位置")
        ax.set_ylabel("浓度")
    elif field == "image":
        ax.imshow(_grid(sample["patches"], 16, 4))
        plain(ax)
        if tokens:
            _token_edges(ax, 16, 4)
    elif field == "weather":
        ax.imshow(np.concatenate(list(_weather_frames(sample["patches"])), axis=1))
        plain(ax)
        for edge in (16, 32, 48):
            ax.axvline(edge - 0.5, color="white", lw=2)
        ax.set_xticks([8, 24, 40, 56], ["第 1 帧", "第 2 帧", "第 3 帧", "第 4 帧"])
        ax.tick_params(length=0)
        if tokens:
            for edge in range(4, 64, 4):
                ax.axvline(edge - 0.5, color="white", lw=0.5)
            for edge in range(4, 16, 4):
                ax.axhline(edge - 0.5, color="white", lw=0.5)
    else:  # structure3d: the 14 visible blocks of the 16 x 16 distance matrix
        blocks = np.full((16, 16), np.nan)
        blocks[[i for i in range(16) if i not in (3, 12)]] = np.asarray(sample["patches"])
        ax.set_facecolor(PANEL)
        ax.imshow(_grid(blocks, 16, 4))
        plain(ax)
        _token_edges(ax, 16, 4)
        ax.text(13.5, 1.5, "遮住", ha="center", va="center", color=MUTED, fontsize=10)
        ax.text(1.5, 13.5, "遮住", ha="center", va="center", color=MUTED, fontsize=10)


def show_samples(field, task, count=3):
    """A few objects of the field next to their labels, and the distribution of the labels."""
    spec = eleven.FIELDS[field]
    inputs, labels = task["A"]
    pool = task["pool"][1]
    order = np.argsort(labels)
    picks = order[np.linspace(0, len(order) - 1, count + 2).astype(int)[1:-1]]  # low, middle and high labels
    stacked = spec["kind"] == "symbol" or field == "weather"
    if stacked:
        height = 0.45 if spec["kind"] == "symbol" else 1.7
        fig, axes = plt.subplots(count + 1, 1, figsize=(11, height * count + 4.2), height_ratios=[height] * count + [3],
                                 layout="constrained")
    else:
        fig, axes = plt.subplots(1, count + 1, figsize=(3.6 * (count + 1), 4.2), layout="constrained")
    for ax, row in zip(axes, picks):
        _draw(field, ax, {name: values[row] for name, values in inputs.items()})
        label = f"标签 {labels[row]:.3g}"
        if spec["kind"] == "symbol":  # the label sits to the left of the sequence
            ax.text(-0.01, 0.5, label, ha="right", va="center", color=MAIN, fontsize=12, fontweight="bold", transform=ax.transAxes)
        else:
            ax.set_title(label, color=MAIN, fontweight="bold")
    ax = axes[-1]
    bins = np.histogram_bin_edges(np.concatenate([pool, labels]), bins=24)
    ax.hist(pool, bins=bins, density=True, color=MAIN, alpha=0.8, label=f"标签池 {len(pool)} 个，从中抽 {eleven.N_LABELS} 个训练")
    ax.hist(labels, bins=bins, density=True, histtype="step", color=SECOND, lw=2.2, label=f"留出组 {len(labels)} 个，只用来评测")
    ax.set_xlabel(spec["task"])
    ax.set_ylabel("占比（密度）")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
    ax.legend(loc="upper left")
    if spec["kind"] == "symbol":
        fig.get_layout_engine().set(rect=(0.09, 0, 0.91, 1))  # room for the labels left of the sequences
    low, high = min(pool.min(), labels.min()), max(pool.max(), labels.max())
    headline(fig, f"{spec['name']}：{spec['question']}。{spec['data']}数据，标签从 {low:.3g} 到 {high:.3g}")
    plt.show()


def show_tokens(field, task, row=0):
    """The same object twice: as the scientist sees it, and as the sequence of tokens the model reads."""
    spec = eleven.FIELDS[field]
    inputs, labels = task["A"]
    sample = {name: values[row] for name, values in inputs.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), width_ratios=[1, 1.3])
    _draw(field, axes[0], sample, tokens=True)
    axes[0].set_title("原始对象" + ("" if spec["kind"] in ("symbol", "crystal") else "，细线是 token 的边界"))
    ax = axes[1]
    if spec["kind"] == "symbol":
        table = eleven.symbols(field)
        ids = np.asarray(sample["tokens"])
        per_row = 16
        real = int((ids > 0).sum())
        shown = min(len(ids), -(-(real + 1) // per_row) * per_row)  # rows that hold only padding are left out
        grid = np.zeros(-(-shown // per_row) * per_row, dtype=int)
        grid[:shown] = ids[:shown]
        grid = grid.reshape(-1, per_row)
        ax.imshow(np.where(grid > 0, grid, np.nan), cmap="Pastel1", aspect=1.5, vmin=1, vmax=len(table))
        for (row_index, column), token in np.ndenumerate(grid):
            symbol = table[token] if token else "·"
            ax.text(column, row_index, f"{'空格' if symbol == ' ' else symbol}\n{token}", ha="center", va="center",
                    fontsize=9, color=INK if token else MUTED)
        plain(ax)
        ax.set_yticks(range(len(grid)), [f"第 {r * per_row + 1} 个起" for r in range(len(grid))])
        ax.tick_params(length=0)
        rest = f"，后面 {len(ids) - shown} 个位置都是填充" if shown < len(ids) else ""
        ax.set_title(f"模型读到的编号：上面是符号，下面是编号，“·” 是填充{rest}")
        summary = f"这条样本被切成 {real} 个 token：一个 token 是 {spec['token']}，查 {len(table)} 项的词表得到编号"
    elif field == "crystal":
        table = eleven.symbols(field)
        rows = [["晶格矩阵", " ".join(f"{v:.2f}" for v in np.asarray(sample["lattice"]).ravel())]]
        rows += [[f"原子 {table[e]}", " ".join(f"{v:.2f}" for v in xyz)]
                 for e, xyz in zip(np.asarray(sample["elements"]), np.asarray(sample["fractional"]))]
        plain(ax)
        ax.set_frame_on(False)
        cells = ax.table(cellText=rows, colLabels=["token", "包含的数"], loc="center", cellLoc="left", colWidths=[0.25, 0.75])
        cells.auto_set_font_size(False)
        cells.set_fontsize(11)
        cells.scale(1, 1.9)
        for (row_index, _), cell in cells.get_celld().items():
            cell.set_edgecolor("#dfe4ea")
            cell.set_facecolor(PANEL if row_index == 0 else "white")
        ax.set_title("模型读到的 6 个 token")
        summary = "这条样本被切成 6 个 token：晶格矩阵 1 个，之后每个原子 1 个，带着元素和它在晶胞里的位置"
    else:
        patches = np.asarray(sample["patches"])
        picture = ax.imshow(patches, aspect="auto")
        ax.grid(False)
        ax.set_xlabel(f"一个 token 里的 {patches.shape[1]} 个数")
        ax.set_ylabel("token 序号")
        ax.set_title("模型读到的 token：每行一个")
        fig.colorbar(picture, ax=ax, shrink=0.85)
        summary = f"这条样本被切成 {patches.shape[0]} 个 token：一个 token 是 {spec['token']}，共 {patches.shape[1]} 个数"
    headline(fig, summary)
    plt.show()


def show_normalization(field, raw_task, task):
    """Value distribution before and after normalization (numeric fields), or symbol frequencies (symbol fields)."""
    spec = eleven.FIELDS[field]
    if spec["kind"] == "symbol":
        table = eleven.symbols(field)
        tokens = np.asarray(raw_task["A"][0]["tokens"])
        counts = np.bincount(tokens[tokens > 0], minlength=len(table))[1:]
        lengths = (tokens > 0).sum(1)
        names = [s if s != " " else "空格" for s in table[1:]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), width_ratios=[2, 1])
        axes[0].bar(range(len(counts)), counts, color=MAIN)
        axes[0].set_xticks(range(len(counts)), names, fontsize=11 if len(names) < 12 else 9, rotation=90 if max(map(len, names)) > 2 else 0)
        axes[0].set_ylabel("出现次数")
        axes[0].set_title("每种符号出现的次数")
        axes[0].grid(axis="x", visible=False)
        axes[1].hist(lengths, bins=range(int(lengths.min()), int(lengths.max()) + 2), color=SECOND)
        axes[1].set_xlabel("一条样本的 token 数")
        axes[1].set_ylabel("样本数")
        axes[1].set_title("样本有多长")
        common = names[int(counts.argmax())]
        headline(fig, f"符号类输入不做数值标准化：最常见的符号是 {common}，样本长 {lengths.min()} 到 {lengths.max()} 个 token，不够长的位置用 0 填充")
        plt.show()
        return
    key = "lattice" if field == "crystal" else "patches"
    before, after = (np.asarray(panel["A"][0][key]).ravel() for panel in (raw_task, task))
    if field == "crystal":
        before, after = before[before != 0], after[after != 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(before, bins=60, color=MUTED)
    axes[0].set_title("标准化之前" + ("（晶格边长，Å）" if field == "crystal" else ""))
    axes[1].hist(after, bins=60, color=MAIN)
    axes[1].set_title("标准化之后")
    for ax in axes:
        ax.set_xlabel("数值")
        ax.set_ylabel("个数")
    how = "晶格边长除以训练对象里最大的边长" if field == "crystal" else "每个位置减去均值、除以标准差"
    headline(fig, f"标准化后数值落在 {after.min():.2g} 到 {after.max():.2g} 之间：{how}，统计量只取自 {eleven.N_LABELS} 个训练对象")
    plt.show()


def show_model(fields):
    """Parameter count of each model part for one or two fields: only the encoder and the position table differ."""
    names = ("编码器（随方向更换）", "位置编码（随序列长度变）", "两层 nanoGPT（结构相同）", "输出前的归一化", "读出头（一个线性层）")
    fig, ax = plt.subplots(figsize=(11, 2.4 + 1.1 * len(fields)))
    height = 0.8 / len(fields)
    totals = []
    for index, (field, color) in enumerate(zip(fields, (MAIN, SECOND))):
        model = eleven.network(field)
        counts = [sum(p.numel() for p in part.parameters())
                  for part in (model.encoder, model.wpe, model.blocks, model.ln_f, model.head)]
        totals.append(counts)
        bars = ax.barh(np.arange(len(names)) + index * height, counts, height=height, color=color,
                       label=f"{eleven.FIELDS[field]['name']}：共 {sum(counts):,} 个参数")
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
    ax.set_yticks(np.arange(len(names)) + height * (len(fields) - 1) / 2, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 31000)
    ax.set_xlabel("参数个数")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    share = totals[0][2] / sum(totals[0])
    headline(fig, f"{share:.0%} 的参数在两层 nanoGPT 里，换方向只换编码器（这个方向 {totals[0][0]:,} 个参数）和读出头")
    plt.show()


def show_training(curves, heldout):
    """Pretraining loss per update for each size, with the held-out loss at the end as a dot."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = plt.get_cmap(SEQUENCE)(np.linspace(0.05, 0.8, len(curves)))
    for (size, losses), color in zip(curves.items(), colors):
        window = max(1, len(losses) // 60)
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(smooth)) + window, smooth, color=color, label=f"{size:,} 个对象")
        ax.scatter([len(losses)], [heldout[size]], s=90, color=color, edgecolor=INK, linewidth=1.2, zorder=3)
    ax.scatter([], [], s=90, color="white", edgecolor=INK, linewidth=1.2, label="圆点：留出样本上的损失")
    ax.set_xscale("log")
    ax.set_xlim(1, None)
    ax.set_xlabel("预训练更新次数")
    ax.set_ylabel("预测下一个 token 的损失")
    ax.legend(title="曲线：训练批次上的损失")
    sizes = list(curves)
    change = heldout[sizes[-1]] / heldout[sizes[0]] - 1
    headline(fig, f"无标签对象从 {sizes[0]:,} 个加到 {sizes[-1]:,} 个，留出样本上的预训练损失从 {heldout[sizes[0]]:.3g} "
                  f"{'降' if change < 0 else '升'}到 {heldout[sizes[-1]]:.3g}")
    plt.show()


def show_predictions(truth, predictions, guess_mean, unit):
    """Predicted against true labels on holdout A, one panel per model."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.4 * len(predictions), 5), sharex=True, sharey=True, squeeze=False)
    low, high = truth.min(), truth.max()
    errors = {name: float(np.abs(values - truth).mean()) for name, values in predictions.items()}
    for ax, (name, values), color in zip(axes[0], predictions.items(), (SECOND, MAIN, MAIN)):
        ax.scatter(truth, values, s=14, alpha=0.55, color=color, linewidth=0)
        ax.plot([low, high], [low, high], color=INK, lw=1.2, ls="--")
        ax.set_title(f"{name}：平均绝对误差 {errors[name]:.3g}")
        ax.set_xlabel(f"真实值：{unit}")
    axes[0][0].set_ylabel("模型预测")
    names = list(errors)
    first, last = errors[names[0]], errors[names[-1]]
    headline(fig, f"{names[-1]}：留出误差 {last:.3g}，比{names[0]}{style.relative(last, first)}，"
                  f"比只猜平均值{style.relative(last, guess_mean)}")
    plt.show()


def show_scaling(pretrain_loss, downstream_error, guess_mean, title, unit):
    """Left: downstream error against the number of unlabeled objects, with both baselines. Right: pretraining loss."""
    sizes = list(pretrain_loss)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(sizes, [downstream_error[s] for s in sizes], "o-", color=MAIN, label="预训练后再训练读出")
    ax.axhline(downstream_error[0], color=SECOND, label="不预训练")
    ax.axhline(guess_mean, color=INK, ls="--", lw=1.4, label="只猜平均值")
    ax.set_ylabel(f"留出组的平均绝对误差\n{unit}")
    ax.set_title("下游误差，越低越好")
    ax.legend()
    axes[1].plot(sizes, [pretrain_loss[s] for s in sizes], "o-", color=MUTED)
    axes[1].set_ylabel("留出样本上预测下一个 token 的损失")
    axes[1].set_title("预训练损失，越低越好")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("预训练用的无标签对象数")
    errors = [downstream_error[s] for s in sizes]
    falling = all(a > b for a, b in zip(errors, errors[1:]))
    trend = "逐档下降" if falling else ("最大档低于最小档，中间有起伏" if errors[-1] < errors[0] else "没有随数据量下降")
    headline(fig, f"{title}：下游误差{trend}，最大档 {errors[-1]:.3g}，比不预训练{style.relative(errors[-1], downstream_error[0])}，"
                  f"比只猜平均值{style.relative(errors[-1], guess_mean)}")
    plt.show()

