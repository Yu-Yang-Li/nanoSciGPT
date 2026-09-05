"""Attach a small task head to a classroom checkpoint.

The tasks are deliberately small and CPU-friendly. Text, protein, and DNA use
transparent teaching labels derived from the input; SMILES uses the measured
ESOL solubility column bundled with the repository. These runs prove that the
pretraining-to-task path is executable, not that the tiny checkpoint is
scientifically competitive.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..core.gpt import GPT, GPTConfig
from ..core.tokenizer import CharTokenizer
from .structured_demo import STRUCTURED_DOMAINS


RUNNABLE_DOMAINS = ("text", "protein", "dna", "smiles") + STRUCTURED_DOMAINS


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "causal")
    config = GPTConfig(**{key: ckpt["model_args"][key] for key in keys if key in ckpt["model_args"]})
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt


def pad_sequences(sequences, block_size, pad_id):
    x = torch.full((len(sequences), block_size), pad_id, dtype=torch.long)
    pad = torch.ones((len(sequences), block_size), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        values = np.asarray(sequence, dtype=np.int64)[:block_size]
        x[row, : len(values)] = torch.from_numpy(values)
        pad[row, : len(values)] = False
    return x, pad


def sequence_features(model, x, pad):
    positions = torch.arange(x.size(1), device=x.device)
    hidden = model.transformer.wte(x) + model.transformer.wpe(positions)
    for block in model.transformer.h:
        hidden = block(hidden, pad)
    hidden = model.transformer.ln_f(hidden)
    keep = (~pad).float().unsqueeze(-1)
    return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)


def extract_features(model, sequences, block_size, pad_id, batch_size=64):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            x, pad = pad_sequences(sequences[start : start + batch_size], block_size, pad_id)
            outputs.append(sequence_features(model, x, pad))
    return torch.cat(outputs, dim=0)


def capped(values, limit):
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return values[indices]


def composition_fraction(sequence, target_ids, ignored_ids=()):
    values = [int(token) for token in sequence if int(token) not in ignored_ids]
    return sum(token in target_ids for token in values) / max(1, len(values))


def load_protein_task(data_dir, max_samples):
    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("student_data"):
        arrays = []
        for split, cap in (("train", max_samples), ("val", max(16, max_samples // 4))):
            sequences = np.load(data_dir / f"{split}_seqs.npy", allow_pickle=True)
            labels = np.load(data_dir / f"{split}_y.npy")
            if len(sequences) != len(labels):
                raise ValueError("student sequences and labels are not aligned")
            indices = np.linspace(0, len(sequences) - 1, min(cap, len(sequences)), dtype=int)
            arrays.extend((sequences[indices], labels[indices]))
        return *arrays, tok.stoi["<pad>"], {
            "task_name": f"student protein regression: {meta['target_column']}", "task_type": "regression",
            "label_source": f"student CSV column: {meta['target_column']}",
            "source_sha256": meta["source_sha256"], "source": meta["source"],
        }
    train = capped(np.load(data_dir / "train_seqs.npy", allow_pickle=True), max_samples)
    val = capped(np.load(data_dir / "val_seqs.npy", allow_pickle=True), max(16, max_samples // 4))
    hydrophobic = {tok.stoi[aa] for aa in "AILMFWV" if aa in tok.stoi}
    ignored = {tok.stoi["<pad>"], tok.stoi["<eos>"]}

    def property_value(sequence):
        return composition_fraction(sequence, hydrophobic, ignored)

    train_property = np.asarray([property_value(sequence) for sequence in train])
    threshold = float(np.median(train_property))
    train_y = (train_property >= threshold).astype(np.int64)
    val_y = np.asarray([property_value(sequence) >= threshold for sequence in val], dtype=np.int64)
    return train, train_y, val, val_y, tok.stoi["<pad>"], {
        "task_name": "protein composition teaching classification",
        "task_type": "classification",
        "label_source": "sequence-derived teaching label",
    }


def stream_windows(path, block_size, count):
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    if len(stream) <= block_size:
        raise ValueError(f"stream too short for block_size={block_size}: {path}")
    starts = np.linspace(0, len(stream) - block_size - 1, min(count, len(stream) - block_size), dtype=int)
    return np.stack([np.asarray(stream[start : start + block_size], dtype=np.int64) for start in starts])


def load_text_task(data_dir, block_size, max_samples):
    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    train = stream_windows(data_dir / "train.bin", block_size, max_samples)
    val = stream_windows(data_dir / "val.bin", block_size, max(16, max_samples // 4))
    punctuation_ids = {tok.stoi[char] for char in ".,;:!?-" if char in tok.stoi}

    def punctuation_fraction(sequence):
        return sum(int(token) in punctuation_ids for token in sequence) / len(sequence)

    train_property = np.asarray([punctuation_fraction(sequence) for sequence in train])
    threshold = float(np.median(train_property))
    train_y = (train_property >= threshold).astype(np.int64)
    val_y = np.asarray(
        [punctuation_fraction(sequence) >= threshold for sequence in val], dtype=np.int64
    )
    return train, train_y, val, val_y, 0, {
        "task_name": "text punctuation-density teaching classification",
        "task_type": "classification",
        "label_source": "text-derived teaching label",
    }


def load_dna_task(data_dir, block_size, max_samples):
    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    train = stream_windows(data_dir / "train.bin", block_size, max_samples)
    val = stream_windows(data_dir / "val.bin", block_size, max(16, max_samples // 4))
    gc_ids = {tok.stoi[base] for base in "GC"}

    def gc_fraction(sequence):
        return sum(int(token) in gc_ids for token in sequence) / len(sequence)

    train_property = np.asarray([gc_fraction(sequence) for sequence in train])
    threshold = float(np.median(train_property))
    train_y = (train_property >= threshold).astype(np.int64)
    val_y = np.asarray([gc_fraction(sequence) >= threshold for sequence in val], dtype=np.int64)
    return train, train_y, val, val_y, 0, {
        "task_name": "DNA GC-content teaching classification",
        "task_type": "classification",
        "label_source": "sequence-derived teaching label",
    }


def load_smiles_labels(csv_path):
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("smiles", "").strip():
                labels.append(float(row["measured log solubility in mols per litre"]))
    return np.asarray(labels, dtype=np.float32)


def load_smiles_task(data_dir, max_samples):
    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    train_all = np.load(data_dir / "train_seqs.npy", allow_pickle=True)
    val_all = np.load(data_dir / "val_seqs.npy", allow_pickle=True)
    labels = load_smiles_labels(data_dir / "delaney-processed.csv")
    split = len(train_all)
    if len(labels) != len(train_all) + len(val_all):
        raise ValueError("ESOL labels and prepared SMILES sequences are not aligned")

    train_indices = np.linspace(0, len(train_all) - 1, min(max_samples, len(train_all)), dtype=int)
    val_indices = np.linspace(0, len(val_all) - 1, min(max(16, max_samples // 4), len(val_all)), dtype=int)
    train = train_all[train_indices]
    val = val_all[val_indices]
    train_y = labels[:split][train_indices]
    val_y = labels[split:][val_indices]
    return train, train_y, val, val_y, tok.stoi["<pad>"], {
        "task_name": "ESOL aqueous-solubility teaching regression",
        "task_type": "regression",
        "label_source": "measured log solubility in bundled Delaney ESOL data",
    }


def fit_classification(train_x, train_y, val_x, val_y, epochs, seed):
    torch.manual_seed(seed)
    head = nn.Linear(train_x.size(1), 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    targets = torch.as_tensor(train_y, dtype=torch.long)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(train_x), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = head(val_x).argmax(dim=1)
        score = (predictions == torch.as_tensor(val_y)).float().mean().item()
    return "accuracy", round(score, 4)


def fit_regression(train_x, train_y, val_x, val_y, epochs, seed):
    torch.manual_seed(seed)
    mean = float(np.mean(train_y))
    scale = float(np.std(train_y)) or 1.0
    train_targets = torch.as_tensor((train_y - mean) / scale, dtype=torch.float32).view(-1, 1)
    head = nn.Linear(train_x.size(1), 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(head(train_x), train_targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = head(val_x).view(-1).numpy() * scale + mean
    mae = float(np.mean(np.abs(prediction - val_y)))
    return "mae", round(mae, 4)


def finetune_task(model, config, payload, epochs, seed, destination, checkpoint, provenance):
    torch.manual_seed(seed)
    train_sequences, train_y, val_sequences, val_y, pad_id, task_meta = payload
    regression = task_meta["task_type"] == "regression"
    head = nn.Linear(config.n_embd, 1 if regression else 2)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    mean = float(np.mean(train_y)) if regression else 0.0
    scale = (float(np.std(train_y)) or 1.0) if regression else 1.0
    if "head" in checkpoint:
        if checkpoint.get("task_sampling") != provenance["task_sampling"]:
            raise ValueError("task sampling changed; keep the same sample limit when continuing this task")
        if checkpoint.get("task") != task_meta or checkpoint.get("data_fingerprint") != provenance["data_fingerprint"]:
            raise ValueError("the task or data changed; do not silently reuse the previous task head")
        head.load_state_dict(checkpoint["head"])
        mean, scale = checkpoint["target_mean"], checkpoint["target_scale"]
    targets = torch.as_tensor((train_y - mean) / scale, dtype=torch.float32) if regression else torch.as_tensor(train_y)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=3e-4)

    def evaluate():
        model.eval()
        with torch.no_grad():
            outputs = head(extract_features(model, val_sequences, config.block_size, pad_id, batch_size=8))
            if regression:
                predictions = outputs.flatten().numpy() * scale + mean
                return float(np.mean(np.abs(predictions - val_y)))
            return float((outputs.argmax(-1) == torch.as_tensor(val_y)).float().mean())

    before = evaluate()
    for _ in range(epochs):
        model.train()
        for indices in torch.randperm(len(train_sequences)).split(8):
            x, pad = pad_sequences([train_sequences[i] for i in indices.tolist()], config.block_size, pad_id)
            outputs = head(sequence_features(model, x, pad))
            loss = nn.functional.mse_loss(outputs.flatten(), targets[indices]) if regression else nn.functional.cross_entropy(outputs, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
    after = evaluate()
    delta = sum(float((value.detach() - original[name]).square().sum()) for name, value in model.named_parameters()) ** 0.5
    torch.save({"domain": checkpoint["domain"], **provenance,
                "model": model.state_dict(), "model_args": vars(config), "head": head.state_dict(),
                "task": task_meta, "target_mean": mean, "target_scale": scale, "pad_id": pad_id}, destination)
    return {"metric_name": "mae" if regression else "accuracy", "metric_value": round(after, 4),
            "metric_before_finetune": round(before, 4), "encoder_delta_l2": delta,
            "task_checkpoint": str(destination.resolve()), "pretrained_parameters_updated": delta > 0}


def run_downstream(domain, ckpt_path, data_root, out_dir, epochs=20, max_samples=128, seed=1337, adaptation="frozen"):
    if adaptation not in ("frozen", "finetune") or epochs < 1 or max_samples < 2:
        raise ValueError("choose frozen or finetune, at least one epoch and two samples")
    if adaptation == "finetune" and (Path(out_dir) / "finetuned.pt").exists():
        raise FileExistsError("a fine-tuned checkpoint already exists; choose a new --out_dir")
    if domain not in RUNNABLE_DOMAINS:
        raise ValueError(f"downstream classroom task is unavailable for domain={domain}")
    if domain in STRUCTURED_DOMAINS:
        if adaptation != "finetune":
            raise ValueError("use structured_demo for frozen-head training, or select --adaptation finetune")
        from .structured_finetune import run
        return run(domain, ckpt_path, data_root, out_dir, epochs, max_samples, seed)
    model, config, checkpoint = load_checkpoint(ckpt_path)
    if checkpoint.get("domain") != domain:
        raise ValueError(f"checkpoint domain={checkpoint.get('domain')} does not match requested {domain}")

    data_dir = Path(data_root) / domain
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("student_data") and not meta.get("target_column"):
        result = {"status": "skipped_no_labels", "domain": domain, "source": meta["source"],
                  "reason": "pretraining ran; no student target column was supplied, so no supervised score is produced"}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "downstream_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if domain == "text":
        payload = load_text_task(data_dir, config.block_size, max_samples)
    elif domain == "protein":
        payload = load_protein_task(data_dir, max_samples)
    elif domain == "dna":
        payload = load_dna_task(data_dir, config.block_size, max_samples)
    else:
        payload = load_smiles_task(data_dir, max_samples)
    train_sequences, train_y, val_sequences, val_y, pad_id, task_meta = payload

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if adaptation == "finetune":
        provenance = {
            "parent_checkpoint_sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
            "task_sampling": {"max_samples": max_samples},
            "data_fingerprint": task_data_fingerprint(data_dir),
        }
        task_result = finetune_task(model, config, payload, epochs, seed, out_dir / "finetuned.pt", checkpoint, provenance)
    else:
        train_x = extract_features(model, train_sequences, config.block_size, pad_id)
        val_x = extract_features(model, val_sequences, config.block_size, pad_id)
        if task_meta["task_type"] == "classification":
            metric_name, metric_value = fit_classification(train_x, train_y, val_x, val_y, epochs, seed)
        else:
            metric_name, metric_value = fit_regression(train_x, train_y, val_x, val_y, epochs, seed)
        task_result = {"metric_name": metric_name, "metric_value": metric_value, "pretrained_parameters_updated": False}

    result = {
        "status": "completed",
        "domain": domain,
        **task_meta,
        **task_result,
        "train_samples": len(train_sequences),
        "val_samples": len(val_sequences),
        "adaptation": adaptation,
        "encoder_frozen": adaptation == "frozen",
        "teaching_only": True,
    }
    result_path = out_dir / "downstream_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("downstream task: completed")
    print(f"task: {task_meta['task_name']}")
    print(f"result saved: {result_path}")
    return result


def task_data_fingerprint(data_dir):
    names = ("meta.json", "tokenizer.json", "train.bin", "val.bin", "train_seqs.npy", "val_seqs.npy",
             "train_y.npy", "val_y.npy", "delaney-processed.csv", "fixture.npz")
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in names if (data_dir / name).exists()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=RUNNABLE_DOMAINS)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--adaptation", choices=("frozen", "finetune"), default="frozen")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out_dir = args.out_dir or Path("out") / "classroom" / args.domain / "downstream"
    run_downstream(
        args.domain,
        args.ckpt,
        args.data_root,
        out_dir,
        epochs=args.epochs,
        max_samples=args.max_samples,
        seed=args.seed,
        adaptation=args.adaptation,
    )


if __name__ == "__main__":
    main()
