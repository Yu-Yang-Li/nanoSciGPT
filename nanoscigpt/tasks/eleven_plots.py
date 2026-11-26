"""Figures for notebooks/eleven_fields.ipynb: what the data look like, how they become tokens, how training goes."""
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, ticker

from . import eleven

TEAL, ORANGE, GREY = "#0a6e78", "#c46a2b", "#8a8a8a"
CMAP = "viridis"
# Noto Sans SC (SIL Open Font License), fetched only when the machine has no Chinese font at all.
FONT_URLS = ("https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
             "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf")


def use_chinese_font():
    """Pick an installed Chinese font, so labels do not turn into boxes. Without one, download Noto Sans SC once."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    wanted = ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans CJK JP",
              "Noto Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC", "Droid Sans Fallback")
    found = [name for name in wanted if name in installed]
    if not found:
        cached = Path(matplotlib.get_cachedir()) / "NotoSansSC-Regular.otf"
        for url in FONT_URLS:
            if cached.exists():
                break
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    cached.write_bytes(response.read())
            except OSError:
                continue
        if cached.exists():
            font_manager.fontManager.addfont(cached)
            found = [font_manager.FontProperties(fname=cached).get_name()]
    plt.rcParams.update({"font.family": found[:1] or ["DejaVu Sans"], "axes.unicode_minus": False})
    return found[0] if found else None


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
        ax.axhline(edge - 0.5, color="white", lw=0.7)
        ax.axvline(edge - 0.5, color="white", lw=0.7)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw(field, ax, sample, label, tokens=False):
    """Draw one object. With tokens=True, also mark where one token ends and the next begins."""
    spec = eleven.FIELDS[field]
    title = f"标签 {label:.3g}"
    if spec["kind"] == "symbol":
        text = _strings(field, [sample["tokens"]])[0]
        ax.axis("off")
        shown = " ".join(text) if tokens else text
        width = 44 if tokens else 80
        wrapped = "\n".join(shown[i:i + width] for i in range(0, len(shown), width))
        if not tokens:  # in the list of samples, the label goes on the same line as the sequence
            ax.text(0.0, 0.5, f"{title}　　{wrapped}", family=plt.rcParams["font.family"], fontsize=10, va="center", transform=ax.transAxes)
            return
        ax.text(0.0, 0.5, wrapped, family="monospace", fontsize=10, va="center", transform=ax.transAxes)
    elif field == "crystal":
        table = eleven.symbols(field)
        elements = [table[e] for e in np.asarray(sample["elements"])]
        xyz = np.asarray(sample["fractional"], dtype=float)
        edge = float(np.asarray(sample["lattice"])[0, 0])
        depth = np.array([0.45, 0.3])  # the cell is drawn as a cube seen from the side: z shifts a point up and right
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        ax.plot(*corners.T, color=GREY, lw=1)
        ax.plot(*(corners + depth).T, color=GREY, lw=1)
        for corner in corners[:4]:
            ax.plot(*np.array([corner, corner + depth]).T, color=GREY, lw=1)
        flat = xyz[:, :2] + xyz[:, 2:] * depth
        ax.scatter(flat[:, 0], flat[:, 1], s=380, color=["#d98c5f", "#5f9bd9", "#dddddd", "#dddddd", "#dddddd"], edgecolor="black", zorder=3)
        for name, (x, y) in zip(elements, flat):
            ax.text(x, y, name, ha="center", va="center", fontsize=9, zorder=4)
        ax.set_xlim(-0.2, 1.65)
        ax.set_ylim(-0.2, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        title += f"，晶胞边长 {edge:.2f} Å"
    elif field == "spectrum":
        flux = np.asarray(sample["patches"]).ravel()
        ax.plot(np.linspace(350, 1000, 64), flux, color=TEAL)
        if tokens:
            for edge in np.linspace(350, 1000, 64)[4::4]:
                ax.axvline(edge - 5, color=GREY, lw=0.5)
        ax.set_xlabel("波长（nm）")
    elif field == "field":
        frames = _field_frames(sample["patches"])
        for index, frame in enumerate(frames):
            ax.plot(frame, color=plt.get_cmap(CMAP)(index / 3.5), label=f"第 {index + 1} 帧")
        if tokens:
            for edge in range(4, 64, 4):
                ax.axvline(edge - 0.5, color=GREY, lw=0.5)
        ax.set_xlabel("位置")
    elif field == "image":
        ax.imshow(_grid(sample["patches"], 16, 4), cmap=CMAP)
        _token_edges(ax, 16, 4) if tokens else (ax.set_xticks([]), ax.set_yticks([]))
    elif field == "weather":
        frames = _weather_frames(sample["patches"])
        ax.imshow(np.concatenate(list(frames), axis=1), cmap=CMAP)
        for edge in (16, 32, 48):
            ax.axvline(edge - 0.5, color="white", lw=1.5)
        ax.set_xticks([8, 24, 40, 56], ["第 1 帧", "第 2 帧", "第 3 帧", "第 4 帧"])
        ax.set_yticks([])
        if tokens:
            for edge in range(4, 64, 4):
                ax.axvline(edge - 0.5, color="white", lw=0.4)
            for edge in range(4, 16, 4):
                ax.axhline(edge - 0.5, color="white", lw=0.4)
    else:  # structure3d: the 14 visible blocks of the 16 x 16 distance matrix
        blocks = np.full((16, 16), np.nan)
        blocks[[i for i in range(16) if i not in (3, 12)]] = np.asarray(sample["patches"])
        ax.imshow(_grid(blocks, 16, 4), cmap=CMAP)
        _token_edges(ax, 16, 4)
        ax.text(13.5, 1.5, "遮住", ha="center", va="center", color="black", fontsize=8)
        ax.text(1.5, 13.5, "遮住", ha="center", va="center", color="black", fontsize=8)
    ax.set_title(title, fontsize=10)


def show_samples(field, task, count=3):
    """A few objects of the field next to their labels, and the distribution of the labels."""
    spec = eleven.FIELDS[field]
    inputs, labels = task["A"]
    order = np.argsort(labels)
    picks = order[np.linspace(0, len(order) - 1, count + 2).astype(int)[1:-1]]  # low, middle and high labels
    wide = spec["kind"] == "symbol" or field == "weather"
    if wide:
        row_height = 0.3 if spec["kind"] == "symbol" else 1.5
        fig, axes = plt.subplots(count + 1, 1, figsize=(9, row_height * count + 3), layout="constrained",
                                 height_ratios=[row_height] * count + [2.4])
    else:
        fig, axes = plt.subplots(1, count + 1, figsize=(3.3 * (count + 1), 3.3), layout="constrained")
    for ax, row in zip(axes, picks):
        _draw(field, ax, {name: values[row] for name, values in inputs.items()}, labels[row])
    ax = axes[-1]
    pool = task["pool"][1]
    bins = np.histogram_bin_edges(np.concatenate([pool, labels]), bins=24)
    ax.hist(pool, bins=bins, density=True, color=TEAL, alpha=0.75, label=f"标签池（{len(pool)} 个，从中抽 {eleven.N_LABELS} 个训练）")
    ax.hist(labels, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.8, label=f"留出组 A（{len(labels)} 个，只用来评测）")
    ax.set_xlabel(spec["task"])
    ax.set_ylabel("占比（密度）")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{spec['name']}：{spec['question']}（{spec['data']}数据）", fontsize=12)
    plt.show()
    print(f"标签范围 {min(pool.min(), labels.min()):.3g} 到 {max(pool.max(), labels.max()):.3g}，"
          f"标签池平均 {pool.mean():.3g}，标准差 {pool.std():.3g}。")


def show_tokens(field, task, row=0):
    """The same object twice: as the scientist sees it, and as the sequence of tokens the model reads."""
    spec = eleven.FIELDS[field]
    inputs, labels = task["A"]
    sample = {name: values[row] for name, values in inputs.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained", width_ratios=[1, 1.25])
    _draw(field, axes[0], sample, labels[row], tokens=True)
    axes[0].set_title("原始对象，细线是 token 的边界", fontsize=10)
    ax = axes[1]
    if spec["kind"] == "symbol":
        table = eleven.symbols(field)
        ids = np.asarray(sample["tokens"])
        per_row = 16
        grid = np.zeros(-(-len(ids) // per_row) * per_row, dtype=int)
        grid[:len(ids)] = ids
        grid = grid.reshape(-1, per_row)
        ax.imshow(np.where(grid > 0, grid, np.nan), cmap="Pastel1", aspect="auto", vmin=1, vmax=len(table))
        for (row_index, column), token in np.ndenumerate(grid):
            symbol = table[token] if token else "·"
            ax.text(column, row_index, f"{'空格' if symbol == ' ' else symbol}\n{token}", ha="center", va="center", fontsize=7,
                    color="black" if token else GREY)
        ax.set_xticks([])
        ax.set_yticks(range(len(grid)), [f"第 {r * per_row + 1} 个起" for r in range(len(grid))], fontsize=8)
        ax.set_title(f"模型读到的 {len(ids)} 个位置：上面是符号，下面是编号，“·” 是填充（词表 {len(table)} 项）", fontsize=10)
    elif field == "crystal":
        table = eleven.symbols(field)
        rows = [["晶格矩阵", " ".join(f"{v:.2f}" for v in np.asarray(sample["lattice"]).ravel())]]
        rows += [[f"原子 {table[e]}", " ".join(f"{v:.2f}" for v in xyz)]
                 for e, xyz in zip(np.asarray(sample["elements"]), np.asarray(sample["fractional"]))]
        ax.axis("off")
        cells = ax.table(cellText=rows, colLabels=["token", "包含的数"], loc="center", cellLoc="left", colWidths=[0.25, 0.75])
        cells.auto_set_font_size(False)
        cells.set_fontsize(8)
        ax.set_title("模型读到的 6 个 token", fontsize=10)
    else:
        patches = np.asarray(sample["patches"])
        picture = ax.imshow(patches, cmap=CMAP, aspect="auto")
        ax.set_xlabel(f"一个 token 里的 {patches.shape[1]} 个数")
        ax.set_ylabel(f"token 序号（共 {patches.shape[0]} 个）")
        ax.set_title("模型读到的 token：每行一个", fontsize=10)
        fig.colorbar(picture, ax=ax, shrink=0.8)
    plt.show()


def show_normalization(field, raw_task, task):
    """Value distribution before and after normalization (numeric fields), or symbol frequencies (symbol fields)."""
    spec = eleven.FIELDS[field]
    if spec["kind"] == "symbol":
        table = eleven.symbols(field)
        tokens = np.asarray(raw_task["A"][0]["tokens"])
        counts = np.bincount(tokens[tokens > 0], minlength=len(table))[1:]
        lengths = (tokens > 0).sum(1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained", width_ratios=[2, 1])
        axes[0].bar(range(len(counts)), counts, color=TEAL)
        axes[0].set_xticks(range(len(counts)), [s if s != " " else "空格" for s in table[1:]], fontsize=7, rotation=90 if len(table) > 12 else 0)
        axes[0].set_title("每种符号出现的次数（留出组 A）", fontsize=10)
        axes[1].hist(lengths, bins=range(int(lengths.min()), int(lengths.max()) + 2), color=ORANGE)
        axes[1].set_title("每条样本的 token 数", fontsize=10)
        plt.show()
        print("符号类输入不做数值标准化：每个符号查词表得到一个向量，长度不够的位置用 0 号填充。")
        return
    key = "lattice" if field == "crystal" else "patches"
    before, after = (np.asarray(panel["A"][0][key]).ravel() for panel in (raw_task, task))
    if field == "crystal":
        before, after = before[before != 0], after[after != 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    axes[0].hist(before, bins=60, color=GREY)
    axes[0].set_title("标准化之前的数值" + ("（晶格边长，Å）" if field == "crystal" else ""), fontsize=10)
    axes[1].hist(after, bins=60, color=TEAL)
    axes[1].set_title("标准化之后", fontsize=10)
    plt.show()
    print("标准化用的均值和标准差只从 32 个带标签的训练对象上算，再套用到预训练和评测数据上。")


def show_model(fields):
    """Parameter count of each model part for one or two fields: only the encoder and the position table differ."""
    names = ("编码器（随方向更换）", "位置编码（随序列长度变）", "两层 nanoGPT（结构相同）", "输出前的归一化", "读出头（一个线性层）")
    fig, ax = plt.subplots(figsize=(8.5, 1.6 + 0.9 * len(fields)), layout="constrained")
    height = 0.8 / len(fields)
    for index, (field, color) in enumerate(zip(fields, (TEAL, ORANGE))):
        model = eleven.network(field)
        counts = [sum(p.numel() for p in part.parameters())
                  for part in (model.encoder, model.wpe, model.blocks, model.ln_f, model.head)]
        bars = ax.barh(np.arange(len(names)) + index * height, counts, height=height, color=color,
                       label=f"{eleven.FIELDS[field]['name']}：共 {sum(counts):,} 个参数")
        ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
    ax.set_yticks(np.arange(len(names)) + height * (len(fields) - 1) / 2, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 30000)
    ax.set_xlabel("参数个数")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    plt.show()


def show_training(curves, heldout):
    """Pretraining loss per update for each size, with the held-out loss at the end as a dot."""
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    colors = plt.get_cmap(CMAP)(np.linspace(0.1, 0.85, len(curves)))
    for (size, losses), color in zip(curves.items(), colors):
        window = max(1, len(losses) // 60)
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(smooth)) + window, smooth, color=color, label=f"{size:,} 个对象（训练批次）")
        ax.scatter([len(losses)], [heldout[size]], color=color, edgecolor="black", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, None)
    ax.set_xlabel("预训练更新次数")
    ax.set_ylabel("预测下一个 token 的损失")
    ax.set_title("曲线是训练批次上的损失，圆点是留出样本上的损失", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plt.show()


def show_predictions(truth, predictions, guess_mean, unit):
    """Predicted against true labels on holdout A, one panel per model."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(4.2 * len(predictions), 3.9), layout="constrained", sharex=True, sharey=True)
    low, high = truth.min(), truth.max()
    for ax, (name, values) in zip(np.atleast_1d(axes), predictions.items()):
        ax.scatter(truth, values, s=8, alpha=0.5, color=TEAL)
        ax.plot([low, high], [low, high], color=GREY, lw=1)
        ax.set_title(f"{name}\n平均绝对误差 {np.abs(values - truth).mean():.4g}", fontsize=10)
        ax.set_xlabel(f"真实值：{unit}")
    np.atleast_1d(axes)[0].set_ylabel("模型预测")
    fig.suptitle(f"留出组 A 上的预测。只猜平均值的误差是 {guess_mean:.4g}", fontsize=11)
    plt.show()


def show_scaling(pretrain_loss, downstream_error, guess_mean, title, unit):
    """Left: downstream error against the number of unlabeled objects, with both baselines. Right: pretraining loss."""
    sizes = list(pretrain_loss)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    ax = axes[0]
    ax.plot(sizes, [downstream_error[s] for s in sizes], "o-", color=TEAL, lw=2, label="预训练后再训练读出")
    ax.axhline(downstream_error[0], color=ORANGE, lw=2.5, alpha=0.6, label="不预训练")
    ax.axhline(guess_mean, color="black", ls="--", lw=1, label="只猜平均值")
    ax.set_ylabel(f"留出组 A 的平均绝对误差：{unit}")
    ax.set_title("下游误差（越低越好）", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    axes[1].plot(sizes, [pretrain_loss[s] for s in sizes], "^--", color=GREY, lw=2)
    axes[1].set_ylabel("留出样本上预测下一个 token 的损失")
    axes[1].set_title("预训练损失（越低越好）", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("预训练用的无标签对象数")
    fig.suptitle(title, fontsize=12)
    plt.show()

