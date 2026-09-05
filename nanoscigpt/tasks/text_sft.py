"""Answer-only fine-tuning of the tiny text checkpoint; not a capable chatbot."""

import argparse
import json
from pathlib import Path

import torch

from ..core.tokenizer import CharTokenizer
from .downstream_demo import load_checkpoint


TRAIN_PAIRS = (
    ("What is DNA?", "A sequence of bases."),
    ("What is a protein?", "A chain of amino acids."),
    ("What is a spectrum?", "Light measured by wavelength."),
    ("What does a telescope collect?", "Light."),
    ("What is a label?", "A target for a sample."),
    ("What is training?", "Learning from data."),
    ("What is validation?", "Checking on held out data."),
    ("What is a model?", "A rule learned from data."),
)
VAL_PAIRS = (
    ("Describe DNA.", "A sequence of bases."),
    ("Describe a protein.", "A chain of amino acids."),
    ("Describe a spectrum.", "Light measured by wavelength."),
)


def encode_pair(question, answer, tokenizer, block_size):
    prefix = f"Q:{question}\nA:"
    tokens = tokenizer.encode(prefix + answer + "\n")
    if len(tokens) - 1 > block_size:
        raise ValueError("question and answer exceed the checkpoint context; use the classroom text profile")
    x, y = torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])
    y[:len(tokenizer.encode(prefix)) - 1] = -1
    return x, y


def run(ckpt_path, data_root, out_dir, steps=200, seed=1337):
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("choose a new SFT output directory")
    model, config, checkpoint = load_checkpoint(ckpt_path)
    if checkpoint.get("domain") != "text" or not config.causal:
        raise ValueError("answer-only SFT needs the causal text checkpoint")
    tokenizer = CharTokenizer.load(Path(data_root) / "text/tokenizer.json")
    if config.vocab_size != len(tokenizer.stoi):
        raise ValueError("checkpoint and text tokenizer vocabularies differ")
    train = [encode_pair(q, a, tokenizer, config.block_size) for q, a in TRAIN_PAIRS]
    val = [encode_pair(q, a, tokenizer, config.block_size) for q, a in VAL_PAIRS]
    original = {name: value.detach().clone() for name, value in model.named_parameters()}

    def evaluate():
        model.eval()
        with torch.no_grad():
            losses = [(model(x[None], y[None])[1], int(y.ne(-1).sum())) for x, y in val]
            return sum(float(loss) * count for loss, count in losses) / sum(count for _, count in losses)

    def sample(question):
        model.eval()
        prefix = tokenizer.encode(f"Q:{question}\nA:")
        ids = torch.tensor([prefix])
        with torch.no_grad():
            for _ in range(32):
                logits, _ = model(ids[:, -config.block_size:])
                token = logits[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat((ids, token), dim=1)
                if int(token.item()) == tokenizer.stoi["\n"]:
                    break
        return tokenizer.decode(ids[0, len(prefix):].tolist())

    before = evaluate()
    samples = [{"question": q, "reference": a, "before": sample(q)} for q, a in VAL_PAIRS]
    training_samples = [{"question": q, "reference": a, "before": sample(q)} for q, a in TRAIN_PAIRS]
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_losses = []
    for _ in range(steps):
        model.train()
        batch = torch.randint(len(train), (4,)).tolist()
        optimizer.zero_grad()
        loss = sum(model(train[i][0][None], train[i][1][None])[1] for i in batch) / len(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        training_losses.append(float(loss.detach()))
    after = evaluate()
    for record in samples + training_samples:
        record["after"] = sample(record["question"])
    delta = sum(float((value.detach() - original[name]).square().sum()) for name, value in model.named_parameters()) ** 0.5
    out_dir.mkdir(parents=True)
    torch.save({**checkpoint, "model": model.state_dict(), "sft_steps": steps}, out_dir / "sft.pt")
    result = {"status": "completed", "domain": "text", "loss_scope": "answer_tokens_only", "steps": steps,
              "answer_val_loss_before": before, "answer_val_loss_after": after, "training_losses": training_losses,
              "encoder_delta_l2": delta, "pretrained_parameters_updated": delta > 0, "samples": samples,
              "training_samples": training_samples,
              "train_pairs": TRAIN_PAIRS, "validation_pairs": VAL_PAIRS, "teaching_only": True,
              "data_source": "handwritten teaching Q&A; validation rephrases the same concepts, not new scientific knowledge",
              "checkpoint": str((out_dir / "sft.pt").resolve()),
              "claim_boundary": "demonstrates answer-only training; does not establish general question-answering ability"}
    (out_dir / "sft_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_dir", default="out/classroom/text/sft")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run(args.ckpt, args.data_root, args.out_dir, args.steps)
