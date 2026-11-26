"""Automated research on one nanoSciGPT field: a model proposes changes, the harness runs them.

The backbone is fixed (2 layers, 2 heads, 32 dimensions). A proposal may only change how the
diffusion field is cut into tokens, the pretraining objective and masking, the pretraining
schedule, and the downstream readout. Every trial is scored on the development set; the two
confirmation groups are scored once, after the search, for the best development configuration.

Credentials are read from the process environment (SCNET_API_BASE, SCNET_API_KEY, SCNET_MODEL)
and never written to disk.
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from nanoscigpt.scientific.encoders import PatchEncoder
from nanoscigpt.scientific.scigpt import SciGPT

SPACE = {
    "space_per_token": [4, 8, 16],
    "frames_per_token": [1, 2, 4],
    "order": ["space", "time"],
    "objective": ["next", "skip", "masked"],
    "mask_ratio": [0.15, 0.3, 0.5],
    "pretrain_steps": [800, 1600, 3200, 6400],
    "pretrain_lr": [0.0003, 0.001, 0.003],
    "readout": ["probe", "finetune"],
    "ridge_alpha": [0.1, 1.0, 10.0],
    "finetune_steps": [100, 200, 400],
    "finetune_lr": [0.0003, 0.001],
}
START = {"space_per_token": 4, "frames_per_token": 4, "order": "space", "objective": "next", "mask_ratio": 0.15,
         "pretrain_steps": 6400, "pretrain_lr": 0.001, "readout": "finetune", "ridge_alpha": 1.0,
         "finetune_steps": 200, "finetune_lr": 0.001}
BACKBONE = {"n_layer": 2, "n_head": 2, "n_embd": 32}
DEV_SEEDS = (53, 71, 89)
CONFIRM_SEEDS = (53, 71, 89, 107, 131)
SIZES = {"unlabeled": 12800, "pool": 256, "dev": 480, "A": 480, "B": 480}
FENCE = re.compile(r"^\s*\x60{3}(?:json)?\s*|\s*\x60{3}\s*$")


def fields(part):
    """Raw diffusion fields (n, 4 frames, 64 positions) and log10 diffusivity, same streams as the study."""
    index = list(SIZES).index(part)
    n = SIZES[part]
    rng = np.random.default_rng([815530, 3, index])
    xs, ys = [], []
    for start in range(0, n, 256):
        m = min(256, n - start)
        modes = np.arange(1, 4)
        coefficient = np.exp(rng.uniform(np.log(0.018), np.log(0.085), m))
        amplitude = rng.normal(0, 1, (m, 1, 3))
        phase = rng.uniform(0, 2 * np.pi, (m, 1, 1, 3))
        decayed = amplitude * np.exp(-np.arange(4)[None, :, None] * modes**2 * coefficient[:, None, None])
        waves = np.sin(2 * np.pi * np.arange(64)[None, None, :, None] * modes / 64 + phase)
        xs.append((decayed[:, :, None, :] * waves).sum(-1) + rng.normal(0, 0.025, (m, 4, 64)))
        ys.append(np.log10(coefficient))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def tokenize(raw, config):
    """Cut (n, 4, 64) fields into tokens of frames_per_token x space_per_token values."""
    s, f = config["space_per_token"], config["frames_per_token"]
    n = len(raw)
    blocks = raw.reshape(n, 4 // f, f, 64 // s, s)
    if config["order"] == "space":
        blocks = blocks.transpose(0, 3, 1, 2, 4)
    else:
        blocks = blocks.transpose(0, 1, 3, 2, 4)
    return blocks.reshape(n, (4 // f) * (64 // s), f * s)


def validate(config):
    if not isinstance(config, dict) or set(config) != set(SPACE):
        raise ValueError("配置必须恰好包含这些键：" + ", ".join(SPACE))
    for key, allowed in SPACE.items():
        if config[key] not in allowed:
            raise ValueError(f"{key} 只能取 {allowed}")
    return config


def pretrain_loss(model, batch, config, generator):
    patches = batch["patches"]
    if config["objective"] == "masked":
        keep = (torch.rand(patches.shape[:2], generator=generator) >= config["mask_ratio"]).float()[..., None]
        hidden = model.hidden({"patches": patches * keep})[0]
    else:
        hidden = model.hidden(batch)[0]
    gap = 2 if config["objective"] == "skip" else 1
    return F.mse_loss(model.encoder.to_patch(hidden[:, :-gap]), patches[:, gap:])


@torch.no_grad()
def pooled(model, x):
    return torch.cat([model.pooled(*model.hidden({"patches": x[i:i + 256]})) for i in range(0, len(x), 256)]).numpy()


def evaluate(config, seeds, groups, data):
    """Mean MAE over seeds on each requested group, plus held-out pretraining loss on the dev inputs."""
    tokens = {g: torch.from_numpy(tokenize(data[g][0], config)) for g in data}
    width, length = tokens["pool"].shape[2], tokens["pool"].shape[1]
    scores = {g: [] for g in groups}
    losses = []
    for seed in seeds:
        rows = np.random.default_rng(seed).choice(SIZES["pool"], 32, replace=False)
        sample = tokens["pool"][rows]
        center, spread = sample.mean((0, 1)), sample.std((0, 1), unbiased=False).clamp_min(1e-6)
        x = {g: (t - center) / spread for g, t in tokens.items()}
        torch.manual_seed(seed)
        model = SciGPT(PatchEncoder(width, BACKBONE["n_embd"]), length, n_layer=BACKBONE["n_layer"],
                       n_head=BACKBONE["n_head"], n_embd=BACKBONE["n_embd"])
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["pretrain_lr"])
        model.train()
        for _ in range(config["pretrain_steps"]):
            idx = torch.randint(len(x["unlabeled"]), (64,), generator=generator)
            loss = pretrain_loss(model, {"patches": x["unlabeled"][idx]}, config, generator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_hidden = model.hidden({"patches": x["dev"]})[0]
            losses.append(float(F.mse_loss(model.encoder.to_patch(dev_hidden[:, :-1]), x["dev"][:, 1:])))
        y = data["pool"][1][rows]
        if config["readout"] == "probe":
            a = pooled(model, x["pool"][rows])
            m, sd = a.mean(0), a.std(0).clip(1e-6)
            ridge = Ridge(alpha=config["ridge_alpha"]).fit((a - m) / sd, y)
            predict = lambda g: ridge.predict((pooled(model, x[g]) - m) / sd)
        else:
            target = torch.from_numpy(((y - y.mean()) / y.std()).astype(np.float32))
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["finetune_lr"])
            model.train()
            for _ in range(config["finetune_steps"]):
                loss = F.mse_loss(model({"patches": x["pool"][rows]}), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()

            def predict(g):
                with torch.no_grad():
                    out = torch.cat([model({"patches": x[g][i:i + 256]}) for i in range(0, len(x[g]), 256)])
                return out.numpy() * y.std() + y.mean()
        for g in groups:
            scores[g].append(float(np.abs(predict(g) - data[g][1]).mean()))
    return {"mae": {g: float(np.mean(v)) for g, v in scores.items()}, "mae_by_seed": scores,
            "dev_pretrain_loss": float(np.mean(losses)), "tokens": length, "token_width": width}


def ask_model(prompt, temperature):
    names = ("SCNET_API_BASE", "SCNET_API_KEY", "SCNET_MODEL")
    base, key, model = (os.environ.get(n, "").strip() for n in names)
    if not (base and key and model):
        raise RuntimeError("缺少环境变量 SCNET_API_BASE / SCNET_API_KEY / SCNET_MODEL")
    url = base.rstrip("/")
    url = url if url.endswith("/chat/completions") else url + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
            "thinking": {"type": "disabled"}, "max_tokens": 1200, "stream": False}
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                payload = json.load(response)
            text = payload["choices"][0]["message"].get("content") or ""
            return text, {"model": payload.get("model"), "usage": payload.get("usage", {})}
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 2:
                raise RuntimeError(f"模型接口连续失败：{type(error).__name__}") from None
            time.sleep(2 ** attempt)


def proposal_prompt(history):
    lines = [json.dumps({"round": h["round"], "config": h["config"], "dev_mae": round(h["mae"]["dev"], 5),
                         "dev_pretrain_loss": round(h["dev_pretrain_loss"], 5), "reason": h.get("reason", "")},
                        ensure_ascii=False) for h in history]
    return "\n".join([
        "你在做一个小型科学机器学习研究。任务：从一维扩散场的 4 帧观测（每帧 64 个位置）估计扩散系数的 log10。",
        "模型主干固定为 2 层、2 头、32 维的 nanoGPT，不能改。先用 12,800 个无标签场做自监督预训练，再用 32 个带标签样本训练下游读出。",
        "目标：让开发集上的下游平均绝对误差（dev_mae，3 个种子平均）尽量低。",
        "你只能改下面这些设置，每个键只能取列出的值：",
        json.dumps(SPACE, ensure_ascii=False),
        "含义：space_per_token 和 frames_per_token 决定一个 token 包含几个空间位置、几帧；order 决定 token 按空间还是按时间排列；",
        "objective=next 预测下一个 token，skip 预测隔一个的 token，masked 先把 mask_ratio 比例的输入 token 置零再预测下一个 token（mask_ratio 只在 masked 时起作用）；",
        "readout=probe 冻结主干用岭回归读出（ridge_alpha），finetune 全参数微调（finetune_steps, finetune_lr）。",
        "已经做过的实验（按轮次）：",
        *lines,
        "请根据这些结果提出下一个要试的配置，不要和已做过的配置完全相同。只返回一个 JSON 对象，形如",
        '{"reason": "一两句话说明依据和预期", "config": {...全部 11 个键...}}',
    ])


def parse(text):
    value = json.loads(FENCE.sub("", text.strip()))
    return value["config"], str(value.get("reason", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=("model", "random"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    run = args.out / f"{args.method}-{args.repeat}"
    run.mkdir(parents=True, exist_ok=True)
    journal_path = run / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {
        "method": args.method, "repeat": args.repeat, "backbone": BACKBONE, "dev_seeds": DEV_SEEDS,
        "space": SPACE, "trials": [], "calls": []}

    def save():
        journal_path.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")

    data = {g: fields(g) for g in ("unlabeled", "pool", "dev")}
    sampler = random.Random(1000 + args.repeat)
    for t in journal["trials"]:  # replay the sampler so a resumed random run draws the same sequence
        if t["round"] and args.method == "random":
            while {k: sampler.choice(v) for k, v in SPACE.items()} != t["config"]:
                pass
    while len(journal["trials"]) <= args.rounds:
        round_index = len(journal["trials"])
        seen = [t["config"] for t in journal["trials"]]
        if round_index == 0:
            config, reason = dict(START), "起点：README 中扩散场的配置"
        elif args.method == "random":
            while True:
                config = {k: sampler.choice(v) for k, v in SPACE.items()}
                if config not in seen:
                    break
            reason = "均匀随机抽样"
        else:
            prompt = proposal_prompt(journal["trials"])
            config = None
            for attempt in (1, 2):
                tag = f"r{round_index:02d}-a{attempt}"
                (run / f"{tag}.prompt.txt").write_text(prompt, encoding="utf-8")
                tick = time.perf_counter()
                text, meta = ask_model(prompt, temperature=0.7)
                (run / f"{tag}.response.txt").write_text(text, encoding="utf-8")
                entry = {"round": round_index, "attempt": attempt, "seconds": round(time.perf_counter() - tick, 2), **meta}
                try:
                    candidate, reason = parse(text)
                    validate(candidate)
                    if candidate in seen:
                        raise ValueError("这个配置已经做过")
                    config = candidate
                    entry["accepted"] = True
                except (ValueError, KeyError, TypeError) as error:
                    entry["accepted"] = False
                    entry["error"] = str(error)
                    prompt += f"\n上一个回答没有通过检查：{error}。请改正后重新返回完整 JSON。"
                journal["calls"].append(entry)
                save()
                if config:
                    break
            if config is None:
                raise RuntimeError("模型两次都没有给出合格配置")
        tick = time.perf_counter()
        result = evaluate(config, DEV_SEEDS, ["dev"], data)
        journal["trials"].append({"round": round_index, "config": config, "reason": reason,
                                  "seconds": round(time.perf_counter() - tick, 1), **result})
        save()
        print(args.method, args.repeat, round_index, round(result["mae"]["dev"], 5), flush=True)
    if "confirmation" not in journal:
        best = min(journal["trials"], key=lambda t: t["mae"]["dev"])
        confirm = {g: fields(g) for g in ("unlabeled", "pool", "dev", "A", "B")}
        journal["confirmation"] = {"round": best["round"], "config": best["config"],
                                   **evaluate(best["config"], CONFIRM_SEEDS, ["A", "B"], confirm)}
        save()
    print("confirmation", journal["confirmation"]["mae"], flush=True)


if __name__ == "__main__":
    main()
