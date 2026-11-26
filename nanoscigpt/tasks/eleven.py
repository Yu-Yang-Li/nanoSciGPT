"""Eleven scientific inputs, one nanoGPT: data, tokens, pretraining and a downstream readout.

Used by notebooks/eleven_fields.ipynb. Simulated fields are regenerated from fixed random
streams; measured or curated inputs are read from data/eleven/<field>.npz. With seed 53 the
pretraining losses and downstream errors reproduce the five-seed study for that seed.
"""
from pathlib import Path
import zlib

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from ..scientific.encoders import CrystalEncoder, PatchEncoder, SymbolEncoder
from ..scientific.scigpt import SciGPT
from . import one_backbone as train
from .structure_scale import chain_panel

DATA = Path(__file__).resolve().parents[2] / "data" / "eleven"

# readout: "probe" = frozen backbone + ridge regression; "finetune" = all weights, 200 steps.
# budget: "fixed" = 1,600 pretraining updates at every size; "passes" = size // 2 updates.
# data, question, label: the plain-language description shown to students and the public.
FIELDS = {
    "spectrum": dict(name="光谱", kind="patch", width=4, positions=16, task="紫外/可见通量比（log10）", readout="probe", budget="passes",
                     token="连续 4 个波长采样点的通量",
                     data="模拟", question="看可见光部分，推测紫外有多亮", label="模拟设定"),
    "protein": dict(name="蛋白质", kind="symbol", vocab=19, positions=32, task="色谱保留时间（分钟）", readout="probe", budget="fixed",
                    token="1 个氨基酸",
                    data="真实", question="预测蛋白质片段在仪器里多久流出来", label="实验测的"),
    "dna": dict(name="DNA", kind="symbol", vocab=6, positions=64, task="发夹熔解温度（°C）", readout="finetune", budget="fixed",
                token="1 个碱基",
                data="真实", question="预测一小段 DNA 加热到多少度会解开", label="实验测的"),
    "rna": dict(name="RNA", kind="symbol", vocab=6, positions=32, task="三核苷酸熵（bit）", readout="probe", budget="fixed",
                token="1 个碱基",
                data="真实", question="判断一段序列有多少重复", label="直接数出来的"),
    "smiles": dict(name="小分子", kind="symbol", vocab=27, positions=22, task="sp³ 杂化碳比例", readout="probe", budget="fixed",
                   token="1 个 SMILES 词法单元",
                   data="真实分子", question="预测分子里饱和碳占多少", label="由结构式算出来的"),
    "crystal": dict(name="晶体", kind="crystal", positions=6, task="形成能（eV/晶胞）", readout="finetune", budget="passes",
                    token="晶格矩阵 1 个，之后每个原子 1 个（元素 + 分数坐标）",
                    data="计算", question="预测晶体稳不稳定", label="量子力学计算"),
    "structure3d": dict(name="三维结构", kind="patch", width=16, positions=16, task="首尾距（键长）", readout="probe", budget="fixed",
                        token="距离矩阵中的 1 个 4×4 块（遮去含首尾距的两块）",
                        data="模拟", question="根据部分原子间的距离，推测分子链两端相距多远", label="模拟设定"),
    "weather": dict(name="天气场", kind="patch", width=64, positions=16, task="平均云团亮度", readout="probe", budget="fixed",
                    token="同一 4×4 区域连续四帧",
                    data="简化模拟", question="看几帧云团移动的画面，估计云团有多亮", label="模拟设定"),
    "field": dict(name="扩散场", kind="patch", width=16, positions=16, task="扩散系数（log10）", readout="finetune", budget="passes",
                  token="4 个相邻位置连续四帧",
                  data="模拟", question="看几帧画面，推测扩散得多快", label="模拟设定"),
    "image": dict(name="天文图像", kind="patch", width=16, positions=16, task="星像半高全宽（像素）", readout="probe", budget="fixed",
                  token="4×4 像素块",
                  data="模拟", question="估计星点有多模糊", label="模拟设定"),
    "text": dict(name="文本", kind="symbol", vocab=28, positions=62, task="去空格后的词数", readout="finetune", budget="fixed",
                 token="1 个字母或空格",
                 data="真实", question="去掉空格后，数原来有几个词", label="直接数出来的"),
}
SIZES = {"unlabeled": 12800, "pool": 256, "dev": 480, "A": 480, "B": 480}
N_LABELS = 32  # labeled objects used to train the downstream readout


# ---------- simulated generators (same random streams as the experiments) ----------

def _spectrum(n, rng):
    wave = np.linspace(350, 1000, 64)

    def curve(w, temperature):
        planck = lambda x: x**-5.0 / np.expm1(1.4388e7 / (x * temperature))
        return planck(w) / planck(700.0)

    cool = rng.uniform(3000, 5600, (n, 1))
    hot = rng.uniform(6500, 14500, (n, 1))
    share = rng.uniform(0.1, 0.94, (n, 1))
    mix = lambda w: share * curve(w, cool) + (1 - share) * curve(w, hot)
    flux = mix(wave)
    flux = flux / flux.max(1, keepdims=True) + rng.normal(0, 0.008, flux.shape)
    return flux.reshape(n, 16, 4), np.log10(mix(250.0)).ravel()


def _stars(n, rng):
    grid = np.arange(16)
    count = rng.integers(3, 23, n)
    fwhm = rng.uniform(1.5, 4.0, n)
    images = np.zeros((n, 16, 16))
    for i in range(n):
        x, y = rng.uniform(0, 15, (2, count[i]))
        flux = rng.lognormal(0, 0.5, count[i])
        sigma = fwhm[i] / 2.355
        px = np.exp(-0.5 * ((grid[None] - x[:, None]) / sigma) ** 2)
        py = np.exp(-0.5 * ((grid[None] - y[:, None]) / sigma) ** 2)
        images[i] = np.einsum("k,ky,kx->yx", flux / (2 * np.pi * sigma**2), py, px)
    images += 0.05 + rng.normal(0, 0.02, images.shape)
    return images.reshape(n, 4, 4, 4, 4).transpose(0, 1, 3, 2, 4).reshape(n, 16, 16), fwhm


def _clouds(n, rng):
    axis = np.linspace(-1, 1, 16)
    xx, yy = np.meshgrid(axis, axis)
    speed = rng.uniform(0.02, 0.2, n)
    angle = rng.uniform(0, 2 * np.pi, n)
    frames = np.zeros((n, 4, 16, 16))
    brightness = []
    for _ in range(2):
        cx, cy = rng.uniform(-0.6, 0.6, (2, n))
        width = rng.uniform(0.15, 0.35, n)[:, None, None]
        amplitude = rng.uniform(0.5, 1.5, n)[:, None, None]
        brightness.append(amplitude.ravel())
        for t in range(4):
            dx = xx[None] - (cx + speed * np.cos(angle) * t)[:, None, None]
            dy = yy[None] - (cy + speed * np.sin(angle) * t)[:, None, None]
            frames[:, t] += amplitude * np.exp(-0.5 * (dx**2 + dy**2) / width**2)
    frames += rng.normal(0, 0.02, frames.shape)
    patches = frames.reshape(n, 4, 4, 4, 4, 4).transpose(0, 1, 2, 4, 3, 5).reshape(n, 64, 16)
    tokens = patches.reshape(n, 4, 16, 16).transpose(0, 2, 1, 3).reshape(n, 16, 64)
    return tokens, np.mean(brightness, 0)


def _diffusion(n, rng):
    modes = np.arange(1, 4)
    coefficient = np.exp(rng.uniform(np.log(0.018), np.log(0.085), n))
    amplitude = rng.normal(0, 1, (n, 1, 3))
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 1, 3))
    time = np.arange(4)[None, :, None]
    decayed = amplitude * np.exp(-time * modes**2 * coefficient[:, None, None])
    waves = np.sin(2 * np.pi * np.arange(64)[None, None, :, None] * modes / 64 + phase)
    fields = (decayed[:, :, None, :] * waves).sum(-1) + rng.normal(0, 0.025, (n, 4, 64))
    tokens = fields.reshape(n, 32, 8).reshape(n, 4, 16, 4).transpose(0, 2, 1, 3).reshape(n, 16, 16)
    return tokens, np.log10(coefficient)


def _simulated(field):
    parts = {}
    for index, (part, n) in enumerate(SIZES.items()):
        if part in ("dev", "B"):
            continue
        if field in ("spectrum", "image"):
            rng = np.random.default_rng([zlib.crc32(field.encode()), index])
            x, y = (_spectrum if field == "spectrum" else _stars)(n, rng)
        else:  # weather and field were drawn 256 objects at a time from one stream per part
            rng = np.random.default_rng([815530, {"weather": 2, "field": 3}[field], index])
            batches = [(_clouds if field == "weather" else _diffusion)(min(256, n - s), rng) for s in range(0, n, 256)]
            x, y = np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches])
        parts[part] = ({"patches": torch.from_numpy(x.astype(np.float32))}, y.astype(np.float64))
    return parts


def load(field):
    """Return the pretraining panel and the task panel, each as {part: (inputs, labels)}."""
    if field in ("spectrum", "image", "weather", "field"):
        parts = _simulated(field)
        return parts, {g: parts[g] for g in ("pool", "A")}
    if field == "structure3d":
        streams = {"unlabeled": (12800, [264130, 0]), "pool": (256, [264130, 1]), "A": (480, [26112607, 31])}
        parts = {}
        for g, (n, stream) in streams.items():
            x, y = chain_panel(n, stream)
            parts[g] = ({"patches": x}, y.astype(np.float64))
        return parts, {g: parts[g] for g in ("pool", "A")}
    with np.load(DATA / f"{field}.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    def inputs(prefix):
        keys = [k for k in arrays if k.startswith(prefix + ":") and not k.endswith(":y")]
        return {k.split(":", 1)[1]: torch.from_numpy(arrays[k].astype(np.float32 if arrays[k].dtype.kind == "f" else np.int64))
                for k in keys}

    parts = {g: (inputs(g), None) for g in ("unlabeled", "pool", "A")}
    task = {g: (inputs("task_" + g) or parts[g][0], arrays["task_" + g + ":y"].astype(np.float64)) for g in ("pool", "A")}
    return parts, task


def symbols(field):
    """Token id -> symbol for the five symbol fields and the element table for crystals."""
    with np.load(DATA / f"{field}.npz", allow_pickle=False) as z:
        return [str(s) for s in z["symbols"]]


def labeled_rows(n_pool, seed):
    return np.random.default_rng(seed).choice(n_pool, N_LABELS, replace=False)


def show_tokens(field, sample):
    """Print how one object is written as tokens. sample maps input names to one object's arrays."""
    spec = FIELDS[field]
    if spec["kind"] == "symbol":
        ids = [int(t) for t in sample["tokens"] if t]
        table = symbols(field)
        pieces = [table[t] for t in ids]
        print("原始对象：", "".join(pieces))
        print(f"{len(ids)} 个 token：", " | ".join(pieces))
        print("token 编号：", ids)
    elif spec["kind"] == "crystal":
        table = symbols(field)
        lattice = sample["lattice"].numpy().astype(np.float64)
        print("token 1，晶格矩阵（Å）：", np.round(lattice, 3).tolist())
        for i, (element, xyz) in enumerate(zip(sample["elements"].tolist(), sample["fractional"].numpy().astype(np.float64)), start=2):
            print(f"token {i}，原子 {table[element]}，分数坐标 {np.round(xyz, 3).tolist()}")
    else:
        patches = sample["patches"].numpy().astype(np.float64)
        print(f"{patches.shape[0]} 个 token，每个 token 是 {patches.shape[1]} 个数")
        for i, token in enumerate(patches[:3], start=1):
            print(f"token {i}：", np.round(token[:8], 3).tolist(), "..." if len(token) > 8 else "")


def normalize(field, parts, task, seed):
    """Patch statistics and the lattice scale come from the 32 labeled pool objects only."""
    kind = FIELDS[field]["kind"]
    if kind == "symbol":
        return parts, task
    pool = parts["pool"][0]
    rows = torch.from_numpy(labeled_rows(len(next(iter(pool.values()))), seed))
    if kind == "crystal":
        unit = pool["lattice"][rows].abs().max()
        scale = lambda feed: {**feed, "lattice": feed["lattice"] / unit}
    else:
        sample = pool["patches"][rows]
        center = sample.mean((0, 1))
        spread = sample.std((0, 1), unbiased=False).clamp_min(1e-6)
        scale = lambda feed: {"patches": (feed["patches"] - center) / spread}
    parts = {g: (scale(x), y) for g, (x, y) in parts.items()}
    task = {g: (scale(x), y) for g, (x, y) in task.items()}
    return parts, task


def network(field):
    spec = FIELDS[field]
    if spec["kind"] == "symbol":
        encoder = SymbolEncoder(spec["vocab"], train.N_EMBD)
    elif spec["kind"] == "crystal":
        encoder = CrystalEncoder(train.N_EMBD)
    else:
        encoder = PatchEncoder(spec["width"], train.N_EMBD)
    return SciGPT(encoder, spec["positions"], n_embd=train.N_EMBD)


@torch.no_grad()
def heldout_loss(model, feed):
    """Mean next-token loss over every predicted target (symbols, elements or patch values)."""
    model.eval()
    n = len(next(iter(feed.values())))
    total, count = 0.0, 0
    for start in range(0, n, 128):
        chunk = {k: v[start:start + 128] for k, v in feed.items()}
        hidden = model.hidden(chunk)[0][:, :-1]
        if "patches" in chunk:
            target = chunk["patches"][:, 1:]
            total += float(F.mse_loss(model.encoder.to_patch(hidden), target, reduction="sum"))
            count += target.numel()
        else:
            target = chunk["elements"] if "elements" in chunk else chunk["tokens"][:, 1:]
            head = model.encoder.to_element if "elements" in chunk else model.encoder.to_symbol
            total += float(F.cross_entropy(head(hidden).flatten(0, 1), target.flatten(), ignore_index=0, reduction="sum"))
            count += int((target != 0).sum())
    return total / count


def pretrain(field, parts, size, seed, curve=None):
    """Pretrain on the first size objects of this seed's fixed ordering of the unlabeled pool.

    Pass a list as curve to receive the loss of every update.
    """
    torch.manual_seed(seed)
    model = network(field)
    if size:
        order = torch.from_numpy(np.random.default_rng(seed).permutation(12800))
        train.SEED = seed
        train.PRETRAIN_STEPS = 1600 if FIELDS[field]["budget"] == "fixed" else size // 2
        losses = train.pretrain(model, train.select(parts["unlabeled"][0], order[:size]))
        if curve is not None:
            curve.extend(losses)
    return model


@torch.no_grad()
def _pooled(model, feed):
    n = len(next(iter(feed.values())))
    return torch.cat([model.pooled(*model.hidden({k: v[i:i + 256] for k, v in feed.items()})) for i in range(0, n, 256)]).numpy()


def downstream(field, model, task, seed, predictions=None, readout=None):
    """Train the readout on 32 labels; return (MAE on holdout A, MAE of predicting the label mean).

    Pass a list as predictions to receive the predicted value of every holdout object.
    readout overrides the field's default: "probe" or "finetune".
    """
    (pool_x, pool_y), (test_x, test_y) = task["pool"], task["A"]
    rows = labeled_rows(len(pool_y), seed)
    y = pool_y[rows]
    x = {k: v[torch.from_numpy(rows)] for k, v in pool_x.items()}
    if (readout or FIELDS[field]["readout"]) == "probe":
        model.eval()
        a = _pooled(model, x)
        center, scale = a.mean(0), a.std(0).clip(1e-6)
        ridge = Ridge(alpha=1.0).fit((a - center) / scale, y)
        prediction = ridge.predict((_pooled(model, test_x) - center) / scale)
    else:
        target = torch.from_numpy(((y - y.mean()) / y.std()).astype(np.float32))
        model.train()
        train.finetune(model, x, target)
        model.eval()
        prediction = train.predict(model, test_x) * y.std() + y.mean()
    if predictions is not None:
        predictions.extend(prediction.tolist())
    return float(np.abs(prediction - test_y).mean()), float(np.abs(y.mean() - test_y).mean())


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Pretrain one field at several sizes, then train its downstream readout.")
    parser.add_argument("--field", default="protein", choices=list(FIELDS))
    parser.add_argument("--sizes", default="200,800,3200", help="comma-separated numbers of unlabeled objects")
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    torch.set_num_threads(2)
    spec = FIELDS[args.field]
    print(f"{spec['name']}（{spec['data']}数据）：{spec['question']}。标签：{spec['label']}")
    print(f"一个 token = {spec['token']}；下游任务：{spec['task']}")
    parts, task = load(args.field)
    show_tokens(args.field, {k: v[0] for k, v in task["A"][0].items()})
    parts, task = normalize(args.field, parts, task, args.seed)
    print("size,pretrain_loss,downstream_mae,seconds")
    for size in [0, *map(int, args.sizes.split(","))]:
        start = time.perf_counter()
        model = pretrain(args.field, parts, size, args.seed)
        loss = heldout_loss(model, parts["A"][0]) if size else float("nan")
        mae, guess_mean = downstream(args.field, model, task, args.seed)
        print(f"{size},{loss:.5f},{mae:.5f},{time.perf_counter() - start:.0f}")
    print(f"guess_mean,,{guess_mean:.5f},")


if __name__ == "__main__":
    main()
