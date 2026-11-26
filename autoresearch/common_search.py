"""One search space for all eleven fields: a model or a random sampler proposes changes, the harness runs them.

The nanoGPT backbone is fixed (2 layers, 2 heads, 32 dimensions). A proposal may change how base tokens
are grouped into one token, the pretraining objective and masking, the pretraining schedule, and the
downstream readout. Round 0 is the V1 default of the field (nanoscigpt.tasks.eleven), and it reproduces
the numbers of the five-seed study.

During the search every trial is scored on development objects: the labeled pool objects that are not
among the 32 training labels of that seed. Holdout A is scored once per run, after the search, for the
configuration with the lowest development error.

The model is reached through SCNET_API_BASE, SCNET_API_KEY and SCNET_MODEL in the process environment.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from autoresearch.eleven_search import FENCE, ask_model
from nanoscigpt.scientific.encoders import CrystalEncoder, PatchEncoder, SymbolEncoder
from nanoscigpt.scientific.scigpt import SciGPT
from nanoscigpt.tasks import eleven

BACKBONE = {"n_layer": 2, "n_head": 2, "n_embd": 32}
DEV_SEEDS = (53, 71, 89)
CONFIRM_SEEDS = (53, 71, 89, 107, 131)
ARCHIVE = Path(__file__).resolve().parents[1] / "docs" / "autoresearch-eleven" / "results.json"
SHARED_SPACE = {
    "merge": [1, 2],
    "objective": ["next", "skip", "masked"],
    "mask_ratio": [0.15, 0.3, 0.5],
    "pretrain_lr": [0.0003, 0.001, 0.003],
    "readout": ["probe", "finetune"],
    "pool": ["mean", "last"],
    "ridge_alpha": [0.1, 1.0, 10.0],
    "finetune_steps": [100, 200, 400],
    "finetune_lr": [0.0003, 0.001],
}
_PANELS = {}


def default_steps(field):
    """Pretraining updates of the V1 default at 12,800 unlabeled objects."""
    return 1600 if eleven.FIELDS[field]["budget"] == "fixed" else 6400


def space(field):
    full = default_steps(field)
    return {**SHARED_SPACE, "pretrain_steps": [full // 4, full // 2, full]}


def start(field):
    """The V1 default configuration of this field."""
    return {"merge": 1, "objective": "next", "mask_ratio": 0.15, "pretrain_lr": 0.001,
            "readout": eleven.FIELDS[field]["readout"], "pool": "mean", "ridge_alpha": 1.0,
            "finetune_steps": 200, "finetune_lr": 0.001, "pretrain_steps": default_steps(field)}


def validate(field, config):
    allowed = space(field)
    if not isinstance(config, dict) or set(config) != set(allowed):
        raise ValueError("配置必须恰好包含这些键：" + ", ".join(allowed))
    for key, values in allowed.items():
        if config[key] not in values:
            raise ValueError(f"{key} 只能取 {values}")
    return config


def canonical(field, config):
    """Reset settings that cannot change the result, so equivalent configurations compare equal."""
    config, default = dict(config), start(field)
    if eleven.FIELDS[field]["kind"] == "crystal":  # a cell has one lattice token and five atom tokens
        config["merge"] = 1
    if config["objective"] != "masked":
        config["mask_ratio"] = default["mask_ratio"]
    if config["readout"] == "probe":
        config["finetune_steps"], config["finetune_lr"] = default["finetune_steps"], default["finetune_lr"]
    else:
        config["ridge_alpha"] = default["ridge_alpha"]
    return config


def regroup(field, feed, merge):
    """Join each pair of neighbouring base tokens into one token (merge = 2)."""
    spec = eleven.FIELDS[field]
    if merge == 1 or spec["kind"] == "crystal":
        return feed
    if spec["kind"] == "symbol":  # a pair of symbols becomes one symbol of a vocabulary of size vocab * vocab
        tokens = feed["tokens"]
        return {"tokens": tokens[:, 0::2] * spec["vocab"] + tokens[:, 1::2]}
    patches = feed["patches"]
    return {"patches": patches.reshape(len(patches), patches.shape[1] // 2, -1)}


def network(field, merge=1):
    spec = eleven.FIELDS[field]
    width = BACKBONE["n_embd"]
    if spec["kind"] == "crystal":
        return SciGPT(CrystalEncoder(width), spec["positions"], **BACKBONE)
    if spec["kind"] == "symbol":
        encoder = SymbolEncoder(spec["vocab"] ** merge, width)
    else:
        encoder = PatchEncoder(spec["width"] * merge, width)
    return SciGPT(encoder, spec["positions"] // merge, **BACKBONE)


def select(feed, rows):
    return {name: values[rows] for name, values in feed.items()}


def pretrain_loss(model, batch, config, generator):
    """next: predict the following token. skip: predict the token after it. masked: hide a share of the inputs."""
    gap = 2 if config["objective"] == "skip" else 1
    if config["objective"] == "masked":  # same steps as SciGPT.hidden, with the hidden tokens' embeddings set to zero
        embedded, padding = model.encoder(batch)
        keep = torch.rand(embedded.shape[:2], generator=generator) >= config["mask_ratio"]
        hidden = embedded * keep[..., None] + model.wpe(torch.arange(embedded.size(1)))
        for block in model.blocks:
            hidden = block(hidden, padding)
        hidden = model.ln_f(hidden)
    else:
        hidden = model.hidden(batch)[0]
    hidden = hidden[:, :-gap]
    encoder = model.encoder
    if "patches" in batch:
        return F.mse_loss(encoder.to_patch(hidden), batch["patches"][:, gap:])
    if "elements" in batch:  # hidden state i (lattice, then atoms) predicts the element of atom i
        logits, targets = encoder.to_element(hidden), batch["elements"][:, gap - 1:]
    else:
        logits, targets = encoder.to_symbol(hidden), batch["tokens"][:, gap:]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)


def summary(model, feed, pool):
    """One vector per object: the mean over real tokens, or the hidden state of the last real token."""
    hidden, padding = model.hidden(feed)
    if pool == "mean":
        return model.pooled(hidden, padding)
    if padding is None:
        return hidden[:, -1]
    last = (~padding).sum(1) - 1
    return hidden[torch.arange(len(hidden)), last]


def panel(field):
    if field not in _PANELS:
        _PANELS[field] = eleven.load(field)
    return _PANELS[field]


def evaluate(field, config, seed, confirm=False):
    """Pretrain and train the readout for one seed; return the development error (and holdout A if asked)."""
    parts, task = eleven.normalize(field, *panel(field), seed)
    unlabeled, pool, holdout = (regroup(field, feed, config["merge"])
                                for feed in (parts["unlabeled"][0], task["pool"][0], task["A"][0]))
    pool_y, holdout_y = task["pool"][1], task["A"][1]
    torch.manual_seed(seed)
    model = network(field, config["merge"])
    n_objects = len(next(iter(unlabeled.values())))
    unlabeled = select(unlabeled, torch.from_numpy(np.random.default_rng(seed).permutation(n_objects)))
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["pretrain_lr"])
    losses = []
    for _ in range(config["pretrain_steps"]):
        batch = select(unlabeled, torch.randint(n_objects, (64,), generator=generator))
        loss = pretrain_loss(model, batch, config, generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    rows = eleven.labeled_rows(len(pool_y), seed)
    dev_rows = np.setdiff1d(np.arange(len(pool_y)), rows)
    y = pool_y[rows]
    train_x = select(pool, torch.from_numpy(rows))

    @torch.no_grad()
    def vectors(feed):
        n = len(next(iter(feed.values())))
        return torch.cat([summary(model, select(feed, slice(i, i + 256)), config["pool"]) for i in range(0, n, 256)])

    if config["readout"] == "probe":
        model.eval()
        a = vectors(train_x).numpy()
        center, scale = a.mean(0), a.std(0).clip(1e-6)
        ridge = Ridge(alpha=config["ridge_alpha"]).fit((a - center) / scale, y)
        predict = lambda feed: ridge.predict((vectors(feed).numpy() - center) / scale)
    else:
        target = torch.from_numpy(((y - y.mean()) / y.std()).astype(np.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["finetune_lr"])
        model.train()
        for _ in range(config["finetune_steps"]):
            loss = F.mse_loss(model.head(summary(model, train_x, config["pool"])).flatten(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

        @torch.no_grad()
        def predict(feed):
            return model.head(vectors(feed)).flatten().numpy() * y.std() + y.mean()

    dev_x, dev_y = select(pool, torch.from_numpy(dev_rows)), pool_y[dev_rows]
    result = {"dev": float(np.abs(predict(dev_x) - dev_y).mean()),
              "guess_mean_dev": float(np.abs(y.mean() - dev_y).mean()),
              "pretrain_loss": float(np.mean(losses[-50:]))}
    if confirm:
        result["A"] = float(np.abs(predict(holdout) - holdout_y).mean())
        result["guess_mean_A"] = float(np.abs(y.mean() - holdout_y).mean())
    return result


def score(field, config, seeds, cache, confirm=False):
    """Evaluate one configuration on several seeds. Results are kept on disk and reused by later runs."""
    config = canonical(field, config)
    key = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cache) / field
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        full, dev_only = folder / f"{key}-{seed}-A.json", folder / f"{key}-{seed}.json"
        path = full if confirm or full.exists() else dev_only
        if not path.exists():
            record = {"config": config, "seed": seed, **evaluate(field, config, seed, confirm)}
            scratch = path.with_suffix(f".{time.time_ns()}.tmp")
            scratch.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            scratch.replace(path)
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def proposal_prompt(field, history, dev_seeds):
    spec = eleven.FIELDS[field]
    lines = [json.dumps({"round": t["round"], "config": t["config"], "dev_mae": round(t["dev"], 5),
                         "dev_mae_by_seed": [round(v, 5) for v in t["dev_by_seed"]],
                         "pretrain_loss": round(t["pretrain_loss"], 5), "reason": t.get("reason", "")}, ensure_ascii=False)
             for t in history]
    no_merge = "这个方向的 merge 不起作用。" if spec["kind"] == "crystal" else ""
    return "\n".join([
        f"你在做一个小型科学机器学习研究。方向：{spec['name']}（{spec['data']}数据）。任务：{spec['question']}，专业名称是{spec['task']}。",
        f"输入被切成 token：一个 token = {spec['token']}，每条样本 {spec['positions']} 个位置。",
        "模型主干固定为 2 层、2 头、32 维的 nanoGPT，不能改。先用 12,800 个无标签对象做自监督预训练，再用 32 个带标签对象训练下游读出。",
        f"目标：让开发集上的下游平均绝对误差（dev_mae，{len(dev_seeds)} 个种子平均）尽量低。只猜标签平均值时的开发集误差约为 {history[0]['guess_mean_dev']:.5f}。",
        "你只能改下面这些设置，每个键只能取列出的值：",
        json.dumps(space(field), ensure_ascii=False),
        "含义：merge=2 把相邻两个 token 并成一个（序列长度减半）；objective=next 预测下一个 token，skip 预测隔一个的 token，"
        "masked 先把 mask_ratio 比例的输入 token 置零再预测下一个 token；pretrain_steps 和 pretrain_lr 是预训练的更新次数和学习率；",
        "readout=probe 冻结主干、用岭回归读出（ridge_alpha），finetune 更新全部权重（finetune_steps、finetune_lr）；"
        "pool=mean 对全部 token 的表示取平均，last 取最后一个 token 的表示。",
        "这些设置在下列条件下不起作用，只改它们等于重复已做过的实验，会被拒绝：objective 不是 masked 时 mask_ratio 不起作用；"
        "readout=probe 时 finetune_steps 和 finetune_lr 不起作用；readout=finetune 时 ridge_alpha 不起作用。" + no_merge,
        "dev_mae_by_seed 是各个种子的误差。两个配置的平均差距小于种子之间的波动时，不要把它当成确定的改进。"
        "pretrain_loss 是预训练最后 50 步的平均损失，不同 objective 和 merge 之间不能直接比较。",
        "已经做过的实验（按轮次，第 0 轮是起点）：",
        *lines,
        "请根据这些结果提出下一个要试的配置，不要和已做过的配置相同。只返回一个 JSON 对象，形如",
        '{"reason": "一两句话说明依据和预期", "config": {...全部 10 个键...}}',
    ])


def archived(field):
    """The journals of the archived study for one field (two model runs and two random runs)."""
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))["fields"][field]


def verdict(runs):
    """Apply the rule written before the study to the holdout errors of a field's runs."""
    held = {method: [r["confirmation"]["A"] for r in runs if r["method"] == method] for method in ("model", "random")}
    first = runs[0]["confirmation"]
    result = {"start": first["start_A"], "guess_mean": first["guess_mean_A"],
              **{method: float(np.mean(values)) for method, values in held.items() if values}}
    if held["model"]:
        chosen = [r["confirmation"]["round"] for r in runs if r["method"] == "model"]
        if not any(chosen):
            result["against_start"] = "没有找到更好的配置"
        elif all(value < result["start"] for value in held["model"]):
            result["against_start"] = "好于起点"
        else:
            result["against_start"] = "没有稳定好于起点"
    if len(held["model"]) > 1 and len(held["random"]) > 1:
        spread = max(np.ptp(held["model"]), np.ptp(held["random"]))
        if result["model"] >= result["random"]:
            result["against_random"] = "没有跑赢随机搜索"
        elif result["random"] - result["model"] > spread:
            result["against_random"] = "好于随机搜索"
        else:
            result["against_random"] = "和随机搜索分不出高下"
    return result


def run(field, method, out, repeat=1, rounds=8, dev_seeds=DEV_SEEDS, confirm_seeds=CONFIRM_SEEDS, report=print):
    """Search one field. method is "model" or "random"; returns the journal, which is also saved under out."""
    out = Path(out)
    folder = out / field / f"{method}-{repeat}"
    folder.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    journal_path = folder / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {
        "field": field, "method": method, "repeat": repeat, "backbone": BACKBONE, "dev_seeds": list(dev_seeds),
        "space": space(field), "trials": [], "calls": []}
    if journal["dev_seeds"] != list(dev_seeds):
        raise ValueError(f"{journal_path} 是用种子 {journal['dev_seeds']} 跑的，换种子请换一个输出目录")

    def save():
        journal_path.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")

    sampler = random.Random(f"{field}-{repeat}")
    drawn = [start(field)]  # the random sequence is replayed from the beginning, so a resumed run continues it
    while len(journal["trials"]) <= rounds:
        index = len(journal["trials"])
        seen = [t["config"] for t in journal["trials"]]
        if index == 0:
            config, reason = start(field), "起点：V1 默认配置"
        elif method == "random":
            while len(drawn) <= index:
                candidate = canonical(field, {key: sampler.choice(values) for key, values in space(field).items()})
                if candidate not in drawn:
                    drawn.append(candidate)
            config, reason = drawn[index], "均匀随机抽样"
        else:
            prompt = proposal_prompt(field, journal["trials"], dev_seeds)
            config = None
            for attempt in (1, 2, 3):
                tag = f"r{index:02d}-a{attempt}"
                (folder / f"{tag}.prompt.txt").write_text(prompt, encoding="utf-8")
                tick = time.perf_counter()
                text, meta = ask_model(prompt, temperature=0.7)
                (folder / f"{tag}.response.txt").write_text(text, encoding="utf-8")
                call = {"round": index, "attempt": attempt, "seconds": round(time.perf_counter() - tick, 2), **meta}
                try:
                    value = json.loads(FENCE.sub("", text.strip()))
                    candidate = canonical(field, validate(field, value["config"]))
                    if candidate in seen:
                        raise ValueError("这个配置已经做过")
                    config, reason, call["accepted"] = candidate, str(value.get("reason", "")), True
                except (ValueError, KeyError, TypeError) as error:
                    call["accepted"], call["error"] = False, str(error)
                    prompt += f"\n上一个回答没有通过检查：{error}。请改正后重新返回完整 JSON。"
                journal["calls"].append(call)
                save()
                if config:
                    break
            if config is None:
                raise RuntimeError("模型三次都没有给出合格配置")
        tick = time.perf_counter()
        # the starting configuration is also the holdout reference, so its holdout error is computed in the same pass
        results = score(field, config, dev_seeds, cache, confirm=index == 0)
        dev_by_seed = [r["dev"] for r in results]
        journal["trials"].append({
            "round": index, "config": config, "reason": reason, "dev": float(np.mean(dev_by_seed)),
            "dev_by_seed": dev_by_seed, "guess_mean_dev": float(np.mean([r["guess_mean_dev"] for r in results])),
            "pretrain_loss": float(np.mean([r["pretrain_loss"] for r in results])),
            "seconds": round(time.perf_counter() - tick, 1)})
        save()
        report(f"{field} {method}-{repeat} 第 {index} 轮：开发集误差 {journal['trials'][-1]['dev']:.5f}")
    if "confirmation" not in journal:
        best = min(journal["trials"], key=lambda t: t["dev"])
        chosen = score(field, best["config"], confirm_seeds, cache, confirm=True)
        origin = score(field, start(field), confirm_seeds, cache, confirm=True)
        journal["confirmation"] = {
            "round": best["round"], "config": best["config"], "seeds": list(confirm_seeds),
            "A_by_seed": [r["A"] for r in chosen], "A": float(np.mean([r["A"] for r in chosen])),
            "start_A": float(np.mean([r["A"] for r in origin])),
            "guess_mean_A": float(np.mean([r["guess_mean_A"] for r in origin]))}
        save()
    confirmation = journal["confirmation"]
    report(f"{field} {method}-{repeat} 留出组 A：选中第 {confirmation['round']} 轮，误差 {confirmation['A']:.5f}"
           f"（起点 {confirmation['start_A']:.5f}，只猜均值 {confirmation['guess_mean_A']:.5f}）")
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", choices=list(eleven.FIELDS), required=True)
    parser.add_argument("--method", choices=("model", "random"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("out/search"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    run(args.field, args.method, args.out, args.repeat, args.rounds, report=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()

