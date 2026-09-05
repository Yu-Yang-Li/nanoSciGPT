"""Continue a course causal GPT in the original AI Scientist v1 experiment template."""

import argparse
import difflib
import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path

import torch

from .upstream import PROJECTS, assignments, git


def prepare(checkout, checkpoint_path, data_root, name, steps=30):
    checkout, checkpoint_path, data_root = map(lambda p: Path(p).resolve(), (checkout, checkpoint_path, data_root))
    if not re.fullmatch(r"[A-Za-z0-9_]+", name) or steps < 1:
        raise ValueError("use a simple template name and positive training steps")
    if git(checkout, "rev-parse", "HEAD") != PROJECTS["v1"][1]:
        raise ValueError("prepare the pinned original v1 checkout first")
    if git(checkout, "diff", "HEAD", "--", "templates/nanoGPT"):
        raise ValueError("the original nanoGPT template has local changes")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    domain, config = checkpoint.get("domain"), checkpoint["model_args"]
    if domain not in ("text", "protein", "dna", "smiles") or not config.get("causal", True):
        raise ValueError("this bridge supports the four course causal sequence GPTs, not masked or structured encoders")
    if "head" in checkpoint:
        from .native_task import prepare as prepare_task
        return prepare_task(checkout, checkpoint_path, checkpoint, data_root, name, steps)
    source = checkout / "templates/nanoGPT"
    target, data = checkout / "templates" / name, checkout / "data" / name
    if target.exists() or data.exists():
        raise FileExistsError("choose a new template name; previous experiments are preserved")
    chars = json.loads((data_root / domain / "tokenizer.json").read_text(encoding="utf-8"))["chars"]
    if len(chars) != config["vocab_size"]:
        raise ValueError("course tokenizer does not match checkpoint vocabulary")
    original = (source / "experiment.py").read_text(encoding="utf-8")
    content = assignments(original, {
        **{key: repr(config[key]) for key in ("n_layer", "n_head", "n_embd", "block_size")},
        "bias": "True", "dropout": "0.0", "device": "'cpu'", "batch_size": "8", "compile": "False",
        "max_iters": str(steps), "warmup_iters": str(max(1, min(5, steps // 4))),
        "eval_interval": str(max(1, steps // 2)), "eval_iters": "3", "log_interval": "10",
        "num_samples": "1", "max_new_tokens": "16", "never_save_checkpoint": "False",
        "start": repr("M" if "M" in chars else next(c for c in chars if len(c) == 1)),
    }, indent="    ")
    content, count = re.subn(r'    dtype = \([\s\S]*?\n    \) [^\n]+', '    dtype = "float32"', content, count=1)
    if count != 1:
        raise ValueError("upstream precision configuration changed")
    content, count = re.subn(r'    def get_batch\(split\):[\s\S]*?        return x, y',
                            '    def get_batch(split):\n        return course_batch(split, data_dir, batch_size, block_size, device)', content, count=1)
    if count != 1:
        raise ValueError("upstream data loader changed")
    content = "from course_data import get_batch as course_batch, load_initial_model\n" + content
    content = content.replace('print("Initializing a new model from scratch")', 'print("Loading the course checkpoint; optimizer is restarted")')
    content = content.replace('    model = GPT(gptconf)\n',
                              '    model = GPT(gptconf)\n    load_initial_model(model, os.path.join(os.path.dirname(__file__), "initial_model.pt"))\n')
    content, count = re.subn(r'    num_seeds = \{[\s\S]*?\n    \}', f'    num_seeds = {{{name!r}: 1}}', content, count=1)
    if count != 1:
        raise ValueError("upstream seed configuration changed")
    content = content.replace('for dataset in ["shakespeare_char", "enwik8", "text8"]:', f'for dataset in [{name!r}]:')
    compile(content, "experiment.py", "exec")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("run_*", "__pycache__"))
    (target / "experiment.py").write_text(content, encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("native_data.py"), target / "course_data.py")
    shutil.copyfile(checkpoint_path, target / "initial_model.pt")
    plot = (source / "plot.py").read_text(encoding="utf-8").replace(
        'datasets = ["shakespeare_char", "enwik8", "text8"]', f'datasets = [{name!r}]')
    (target / "plot.py").write_text(plot, encoding="utf-8")
    prompt = json.loads((source / "prompt.json").read_text(encoding="utf-8"))
    prompt["task_description"] = (f"Investigate improvements to this pretrained {domain} sequence GPT. "
                                  "Each experiment starts from the same initial_model.pt and a fresh optimizer. "
                                  "Preserve the data boundaries, vocabulary, evaluation, and checkpoint provenance. "
                                  "Report actual results, including failures. Architecture changes need an explicitly justified new baseline.")
    (target / "prompt.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    data.mkdir()
    for filename in (("train.bin", "val.bin") if domain in ("text", "dna") else ("train_seqs.npy", "val_seqs.npy")):
        shutil.copyfile(data_root / domain / filename, data / filename)
    with (data / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": len(chars), "stoi": {c: i for i, c in enumerate(chars)}, "itos": dict(enumerate(chars))}, handle)
    (target / "course_changes.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), content.splitlines(True),
                                                     fromfile="original/experiment.py", tofile="course/experiment.py")), encoding="utf-8")
    receipt = {"status": "prepared_not_run", "upstream_commit": PROJECTS["v1"][1], "domain": domain,
               "template": str(target), "data": str(data), "initial_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
               "optimizer": "fresh optimizer; model weights are continued", "experiment_steps": steps,
               "native_command": f"python launch_scientist.py --experiment {name} --num-ideas 1 --parallel 0 --model <supported-model>"}
    (target / "course_bridge.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default="out/upstream/v1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkout, args.ckpt, args.data_root, args.name, args.steps), indent=2))
