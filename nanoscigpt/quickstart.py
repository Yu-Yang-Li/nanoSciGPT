"""Three-minute start: pretrain one field at three sizes, train the readout, save one figure.

    python -m nanoscigpt.quickstart            # the diffusion field
    python -m nanoscigpt.quickstart protein    # any of the eleven fields

The figure is written to quickstart-<field>.png in the current folder.
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")  # write the figure to a file; no window is needed
import matplotlib.pyplot as plt
import torch

from .tasks import eleven, eleven_plots as plots

SIZES = (200, 800, 3200)
SEED = 53


def main():
    field = sys.argv[1] if len(sys.argv) > 1 else "field"
    if field not in eleven.FIELDS:
        raise SystemExit("可选的方向：" + "、".join(eleven.FIELDS))
    torch.set_num_threads(2)
    plots.use_chinese_font()
    spec = eleven.FIELDS[field]
    print(f"{spec['name']}（{spec['data']}数据）：{spec['question']}")
    start = time.perf_counter()
    parts, task = eleven.normalize(field, *eleven.load(field), SEED)
    pretrain_loss, error = {}, {}
    for size in (0, *SIZES):
        model = eleven.pretrain(field, parts, size, SEED)
        if size:
            pretrain_loss[size] = eleven.heldout_loss(model, parts["A"][0])
        error[size], guess_mean = eleven.downstream(field, model, task, SEED)
        print(f"{'不预训练' if size == 0 else f'{size:,} 个无标签对象预训练'}：留出误差 {error[size]:.4g}")
    print(f"只猜平均值：{guess_mean:.4g}")
    path = f"quickstart-{field}.png"
    plt.show = lambda: (plt.savefig(path), plt.close("all"))
    plots.show_scaling(pretrain_loss, error, guess_mean, f"{spec['name']} · {spec['question']}", spec["task"])
    print(f"用时 {time.perf_counter() - start:.0f} 秒，图存在 {path}")


if __name__ == "__main__":
    main()

