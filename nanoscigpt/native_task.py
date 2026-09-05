"""Export a supervised course task as an original v1-compatible experiment template."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .tasks.downstream_demo import (load_protein_task, load_smiles_task, load_text_task,
                                   load_dna_task, pad_sequences, task_data_fingerprint)
from .upstream import PROJECTS


def prepare(checkout, checkpoint_path, checkpoint, data_root, name, steps):
    domain = checkpoint["domain"]
    data_dir = data_root / domain
    if checkpoint.get("data_fingerprint") != task_data_fingerprint(data_dir):
        raise ValueError("task data changed or this checkpoint lacks data provenance; fine-tune with the current importer first")
    cap = checkpoint["task_sampling"]["max_samples"]
    if domain in ("text", "dna"):
        loader = load_text_task if domain == "text" else load_dna_task
        payload = loader(data_dir, checkpoint["model_args"]["block_size"], cap)
    else:
        loader = load_protein_task if domain == "protein" else load_smiles_task
        payload = loader(data_dir, cap)
    train, train_y, val, val_y, pad_id, task = payload
    if task != checkpoint["task"]:
        raise ValueError("the exported task must match the fine-tuned task")
    target = checkout / "templates" / name
    if target.exists():
        raise FileExistsError("choose a new template name")
    source = checkout / "templates/nanoGPT"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("run_*", "__pycache__"))
    shutil.copyfile(source / "experiment.py", target / "native_gpt.py")
    for asset, filename in (("native_task_experiment.py", "experiment.py"), ("native_task_plot.py", "plot.py")):
        shutil.copyfile(Path(__file__).with_name(asset), target / filename)
    shutil.copyfile(checkpoint_path, target / "initial_model.pt")
    arrays = {}
    for split, sequences, labels in (("train", train, train_y), ("val", val, val_y)):
        x, padding = pad_sequences(sequences, checkpoint["model_args"]["block_size"], pad_id)
        arrays.update({f"{split}_x": x.numpy(), f"{split}_pad": padding.numpy(), f"{split}_y": labels})
    np.savez(target / "task_data.npz", **arrays)
    (target / "task_setup.json").write_text(json.dumps({"task": task, "experiment_steps": steps,
                                                      "data_fingerprint": checkpoint["data_fingerprint"]}, indent=2), encoding="utf-8")
    prompt = json.loads((target / "prompt.json").read_text(encoding="utf-8"))
    prompt["task_description"] = (f"Improve this supervised {domain} task: {task['task_name']}. "
                                  "Continue the supplied encoder AND task head. Preserve task_data.npz, task_setup.json, "
                                  "initial_model.pt and evaluation. Implement experiments in experiment.py, then compare actual results. "
                                  "The GPT class is the upstream implementation; this is a task-specific teaching template. "
                                  "Report negative results and the single-seed small-data limitations. Do not replace the task with language modeling.")
    (target / "prompt.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    receipt = {"status": "prepared_not_run", "upstream_commit": PROJECTS["v1"][1], "domain": domain,
               "template": str(target), "data": str(target / "task_data.npz"), "task_type": task["task_type"],
               "adaptation": "task-specific training template; original GPT class and v1 research workflow",
               "initial_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
               "native_gpt_sha256": hashlib.sha256((target / "native_gpt.py").read_bytes()).hexdigest(),
               "optimizer": "fresh optimizer; encoder and head continued", "experiment_steps": steps,
               "native_command": f"python launch_scientist.py --experiment {name} --num-ideas 1 --parallel 0 --model <supported-model>"}
    (target / "course_bridge.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
