"""Fetch pinned original research systems and prepare reviewable teaching settings.

No dependencies, training, or model API requests are started by this module.
"""

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "autoresearch": ("https://github.com/karpathy/autoresearch.git", "228791fb499afffb54b46200aca536f79142f117"),
    "v1": ("https://github.com/SakanaAI/AI-Scientist.git", "1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb"),
    "v2": ("https://github.com/SakanaAI/AI-Scientist-v2.git", "96bd51617cfdbb494a9fc283af00fe090edfae48"),
}


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()


def checkout(project, destination):
    url, commit = PROJECTS[project]
    if destination.exists():
        if git(destination, "remote", "get-url", "origin") != url:
            raise ValueError("existing origin does not match the pinned upstream")
        try:
            existing = git(destination, "rev-parse", "--verify", "HEAD")
        except subprocess.CalledProcessError:
            if git(destination, "status", "--porcelain"):
                raise ValueError("unfinished checkout contains local files; choose a new --root")
        else:
            if existing != commit:
                raise ValueError(f"{destination} is at another revision; choose a new --root")
            return
    else:
        destination.mkdir(parents=True)
        git(destination, "init")
        git(destination, "remote", "add", "origin", url)
    git(destination, "fetch", "--depth", "1", "origin", commit)
    git(destination, "checkout", "--detach", "FETCH_HEAD")
    if git(destination, "rev-parse", "HEAD") != commit:
        raise ValueError("upstream revision verification failed")


def assignments(text, replacements, indent=""):
    for name, value in replacements.items():
        pattern = rf"(?m)^{re.escape(indent)}{re.escape(name)} = [^\n]+"
        text, count = re.subn(pattern, lambda m: f"{indent}{name} = {value}", text)
        if count != 1:
            raise ValueError(f"expected one assignment for {name}, found {count}")
    return text


def prepare(project, root, device="cpu"):
    destination = Path(root).resolve() / project
    checkout(project, destination)
    receipt = destination / "teaching_setup.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous["device"] != device:
            raise ValueError("existing teaching device differs; choose another --root")
        return previous
    if git(destination, "status", "--porcelain"):
        raise ValueError("upstream has local changes; choose another --root")
    changes = []

    def edited(source, target, content):
        original = source.read_text(encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        changes.extend(difflib.unified_diff(original.splitlines(True), content.splitlines(True),
                       fromfile=str(source.relative_to(destination)), tofile=str(target.relative_to(destination))))

    if project == "autoresearch":
        # Keep the same time allowance for baseline and every candidate.
        for filename, values in {
            "train.py": {"DEPTH": "4", "DEVICE_BATCH_SIZE": "4", "TOTAL_BATCH_SIZE": "2**14", "WINDOW_PATTERN": "\"L\""},
            "prepare.py": {"MAX_SEQ_LEN": "256", "TIME_BUDGET": "60", "EVAL_TOKENS": "16384"},
        }.items():
            path = destination / filename
            edited(path, path, assignments(path.read_text(encoding="utf-8"), values))
        (destination / "TEACHING.md").write_text(
            "# Teaching run\n\nRead the upstream program.md and source first. This session uses a fixed 60-second training budget per candidate, rather than the upstream 300 seconds. "
            "Establish a fresh baseline after these setup changes. Keep prepare.py and the evaluator fixed during the experiment. "
            "Run the baseline and at most two agent-proposed experiments, sequentially. The agent should propose and edit train.py based on the actual evidence. "
            "Record results.tsv, code diffs, logs, and keep/discard/crash decisions. Stop after the agreed session budget; the upstream indefinite-loop instruction does not apply to this lesson. "
            "Use this dedicated checkout for experiments; preserve failed changes as patches or commits. "
            "These smaller-model results are not comparable to the unmodified upstream benchmark. CUDA is still required.\n",
            encoding="utf-8")
        commands = ["uv sync", "uv run prepare.py --num-shards 1 --download-workers 1", "uv run train.py"]
        effective_device = "cuda"
    elif project == "v1":
        source = destination / "templates" / "nanoGPT"
        target = destination / "templates" / "nanoSciGPT_teaching"
        original = (source / "experiment.py").read_text(encoding="utf-8")
        modified = assignments(original, {
            "batch_size": "8", "block_size": "64", "eval_interval": "15", "log_interval": "10",
            "eval_iters": "5", "n_layer": "2", "n_head": "2", "n_embd": "64", "max_iters": "30",
            "warmup_iters": "5", "device": repr(device), "compile": "False", "num_samples": "1", "max_new_tokens": "32",
        }, indent="    ")
        if device == "cpu":
            modified, count = re.subn(r'    dtype = \([\s\S]*?\n    \) [^\n]+', '    dtype = "float32"', modified, count=1)
            if count != 1:
                raise ValueError("upstream dtype setup changed")
        modified = modified.replace('"shakespeare_char": 3,', '"shakespeare_char": 1,')
        modified = modified.replace('for dataset in ["shakespeare_char", "enwik8", "text8"]:', 'for dataset in ["shakespeare_char"]:')
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("run_*", "__pycache__"))
        edited(source / "experiment.py", target / "experiment.py", modified)
        plot = (source / "plot.py").read_text(encoding="utf-8").replace(
            'datasets = ["shakespeare_char", "enwik8", "text8"]', 'datasets = ["shakespeare_char"]')
        edited(source / "plot.py", target / "plot.py", plot)
        # The upstream experiment reads its own metadata format. Reuse identical course token ids.
        data = destination / "data" / "shakespeare_char"
        data.mkdir(exist_ok=True)
        for filename in ("train.bin", "val.bin"):
            shutil.copyfile(ROOT / "data" / "text" / filename, data / filename)
        chars = json.loads((ROOT / "data" / "text" / "tokenizer.json").read_text(encoding="utf-8"))["chars"]
        with (data / "meta.pkl").open("wb") as handle:
            pickle.dump({"vocab_size": len(chars), "stoi": {c: i for i, c in enumerate(chars)}, "itos": dict(enumerate(chars))}, handle)
        prompt = json.loads((target / "prompt.json").read_text(encoding="utf-8"))
        prompt["task_description"] = "Study a small character language model on the bundled Shakespeare subset. The starting model has two layers and trains for 30 iterations. Propose a feasible teaching-scale idea, run genuine comparisons, and report limitations and negative results."
        (target / "prompt.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")
        commands = ["python -m pip install -r requirements.txt",
                    "cd templates/nanoSciGPT_teaching && python experiment.py --out_dir run_0",
                    "python launch_scientist.py --experiment nanoSciGPT_teaching --num-ideas 1 --parallel 0 --model <supported-model>"]
        effective_device = device
    else:
        config = destination / "bfts_config.yaml"
        original = config.read_text(encoding="utf-8")
        modified = original
        # Preserve all four stages and original tree-search code.
        for key, value in {"timeout": "180", "num_workers": "1", "stage1_max_iters": "3", "stage2_max_iters": "3",
                           "stage3_max_iters": "3", "stage4_max_iters": "3", "num_seeds": "1"}.items():
            modified, count = re.subn(rf"(?m)^(\s*{key}:) [^\n]+", rf"\g<1> {value}", modified)
            if count != 1:
                raise ValueError(f"expected one upstream YAML key: {key}")
        edited(config, config, modified)
        (destination / "teaching_topic.md").write_text(
            "# Title\nSmall scientific language models\n\n# Keywords\npretraining, transfer, small data\n\n# TL;DR\nInvestigate one tractable question about a small language model and design measurable experiments.\n\n# Abstract\nUse the student's selected scientific object and available data, with its source and representation explicitly described. "
            "Begin with a tiny model, one worker, and runs within 180 seconds. Compare validation results and report actual negative results. "
            "Use the original experiment manager and tree search to propose and revise experiments. Do not prescribe two fixed routes.\n",
            encoding="utf-8")
        commands = ["python -m pip install -r requirements.txt",
                    "python ai_scientist/perform_ideation_temp_free.py --workshop-file teaching_topic.md --max-num-generations 1 --num-reflections 2 --model <supported-model>",
                    "python launch_scientist_bfts.py --load_ideas teaching_topic.json --idea_idx 0 --writeup-retries 1 --num_cite_rounds 3"]
        effective_device = "cuda"
    (destination / "teaching_changes.diff").write_text("".join(changes), encoding="utf-8")
    result = {"project": project, "url": PROJECTS[project][0], "commit": PROJECTS[project][1],
              "device": device, "required_device": effective_device, "checkout": str(destination),
              "status": "prepared_not_run", "commands": commands, "teaching_changes": "teaching_changes.diff"}
    receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def doctor():
    modules = ["torch", "openai", "anthropic", "aider", "omegaconf"]
    info = {"platform": sys.platform, "python": sys.executable,
            "modules": {m: importlib.util.find_spec(m) is not None for m in modules},
            "commands": {m: shutil.which(m) for m in ("git", "uv", "pdflatex", "chktex")},
            "credentials_present": {m: bool(os.environ.get(m)) for m in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "S2_API_KEY")}}
    if info["modules"]["torch"]:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
    info["note"] = "Readiness inventory only. CLI login is not a model API key; upstream model routing and v2 visual feedback must be tested."
    return info


def configure_v1_api(checkout, model, review_model):
    """Opt in named models to an OpenAI-compatible endpoint; never store credentials."""
    checkout = Path(checkout).resolve()
    if not model or not review_model or not model.strip() or not review_model.strip():
        raise ValueError("both research and review model names are required")
    pin = PROJECTS["v1"][1]
    if git(checkout, "rev-parse", "HEAD") != pin:
        raise ValueError("use the pinned v1 revision in another checkout")
    receipt = checkout / "teaching_api_setup.json"
    names = ("ai_scientist/llm.py", "launch_scientist.py")
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if (previous["model"], previous["review_model"]) != (model, review_model):
            raise ValueError("model selection differs; use another checkout")
        if any(hashlib.sha256((checkout / name).read_bytes()).hexdigest() != previous["output_sha256"][name] for name in names):
            raise ValueError("configured source changed; preserve it and use another checkout")
        return previous
    if (checkout / "teaching_api_changes.diff").exists():
        raise ValueError("existing API patch without a receipt; preserve it and use another checkout")
    sources = {name: (checkout / name).read_text(encoding="utf-8") for name in names}
    for name, source in sources.items():
        if source.rstrip("\n") != git(checkout, "show", f"{pin}:{name}").rstrip("\n"):
            raise ValueError(f"{name} has local changes; use another checkout")

    def replace(text, old, new, count=1):
        if text.count(old) != count:
            raise ValueError(f"unexpected upstream routing: {old}")
        return text.replace(old, new)

    models = tuple(dict.fromkeys((model, review_model)))
    llm = replace(sources[names[0]], "AVAILABLE_LLMS = [\n",
                  f"COURSE_OPENAI_MODELS = {models!r}\nCOURSE_REVIEW_MODEL = {review_model!r}\n\nAVAILABLE_LLMS = [\n" +
                  "".join(f"    {name!r},\n" for name in models))
    llm = replace(llm, "elif 'gpt' in model:", "elif model in COURSE_OPENAI_MODELS or 'gpt' in model:")
    llm = replace(llm, 'elif model == "llama-3-1-405b-instruct":',
                  'elif model == "llama-3-1-405b-instruct" and model not in COURSE_OPENAI_MODELS:')
    llm = replace(llm, 'if "claude" in model:', 'if "claude" in model and model not in COURSE_OPENAI_MODELS:')
    llm = replace(llm, "def create_client(model):\n", "def create_client(model):\n    if model in COURSE_OPENAI_MODELS:\n        return openai.OpenAI(), model\n")
    launcher = replace(sources[names[1]], "from ai_scientist.llm import create_client, AVAILABLE_LLMS",
                       "from ai_scientist.llm import create_client, AVAILABLE_LLMS, COURSE_OPENAI_MODELS, COURSE_REVIEW_MODEL")
    for indent in ("        ", "            "):
        old = indent + 'if model == "deepseek-coder-v2-0724":'
        new = (indent + "if model in COURSE_OPENAI_MODELS:\n" + indent +
               '    main_model = Model("openai/" + model, weak_model="openai/" + model)\n' +
               indent + 'elif model == "deepseek-coder-v2-0724":')
        # Match whole lines so the eight-space pattern cannot match inside twelve spaces.
        launcher = replace(launcher, "\n" + old, "\n" + new)
    launcher = replace(launcher, 'model="gpt-4o-2024-05-13",', "model=COURSE_REVIEW_MODEL,", count=2)
    changed = dict(zip(names, (llm, launcher)))
    for name, text in changed.items():
        compile(text, name, "exec")
    patch = "".join("".join(difflib.unified_diff(sources[name].splitlines(True), text.splitlines(True),
                                                fromfile="original/" + name, tofile="teaching/" + name))
                    for name, text in changed.items())
    (checkout / "teaching_api_changes.diff").write_text(patch, encoding="utf-8")
    for name, text in changed.items():
        (checkout / name).write_text(text, encoding="utf-8")
    result = {"status": "configured_not_run", "commit": pin, "model": model, "review_model": review_model,
              "endpoint_configuration": "Set OPENAI_BASE_URL and OPENAI_API_BASE to the same verified endpoint in the launch process; OPENAI_API_KEY is not saved here.",
              "output_sha256": {name: hashlib.sha256((checkout / name).read_bytes()).hexdigest() for name in names},
              "changes": "teaching_api_changes.diff"}
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def configure_v1_failures(checkout):
    """Preserve failed attempts in pinned v1; keep its research prompts and sequence."""
    checkout = Path(checkout).resolve()
    pin = PROJECTS["v1"][1]
    if git(checkout, "rev-parse", "HEAD") != pin:
        raise ValueError("use the pinned v1 revision in another checkout")
    source = checkout / "ai_scientist/perform_experiments.py"
    receipt = checkout / "teaching_failure_setup.json"
    patch = checkout / "teaching_failure_changes.diff"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if hashlib.sha256(source.read_bytes()).hexdigest() != previous["output_sha256"]:
            raise ValueError("configured source changed; preserve it and use another checkout")
        return previous
    if patch.exists():
        raise ValueError("existing failure patch without a receipt; use another checkout")
    original = source.read_text(encoding="utf-8")
    if original.rstrip("\n") != git(checkout, "show", f"{pin}:ai_scientist/perform_experiments.py").rstrip("\n"):
        raise ValueError("executor has local changes; preserve it and use another checkout")
    modified = original

    def replace(old, new):
        nonlocal modified
        if modified.count(old) != 1:
            raise ValueError(f"unexpected upstream executor: {old}")
        modified = modified.replace(old, new)

    helper = '''def preserve_failed_attempt(cwd, run_num, reason, stderr):
    # Teaching adaptation: keep partial files and exact attempted code before retry.
    from pathlib import Path
    import tempfile

    root = Path(cwd).resolve()
    failed = root / "failed_runs"
    failed.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"run_{run_num}_", dir=failed))
    run = root / f"run_{run_num}"
    if run.exists():
        if run.is_symlink() or run.resolve().parent != root:
            raise ValueError("refusing to move results outside the experiment directory")
        shutil.move(str(run), str(attempt / "artifacts"))
    shutil.copy(root / f"run_{run_num}.py", attempt / "experiment.py")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    (attempt / "failure.json").write_text(
        json.dumps({"run": run_num, "reason": reason, "stderr": stderr or ""}, indent=2),
        encoding="utf-8",
    )


'''
    replace("# RUN EXPERIMENT\ndef run_experiment", helper + "# RUN EXPERIMENT\ndef run_experiment")
    replace('            if osp.exists(osp.join(cwd, f"run_{run_num}")):\n'
            '                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))',
            '            preserve_failed_attempt(cwd, run_num, "nonzero_exit", result.stderr)')
    replace('    except TimeoutExpired:\n        print(f"Run {run_num} timed out after {timeout} seconds")\n'
            '        if osp.exists(osp.join(cwd, f"run_{run_num}")):\n'
            '            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))',
            '    except TimeoutExpired as error:\n        print(f"Run {run_num} timed out after {timeout} seconds")\n'
            '        preserve_failed_attempt(cwd, run_num, "timeout", error.stderr)')
    replace('    current_iter = 0\n    next_prompt = """\nGreat job!',
            '    if run == 1:\n        print("No new experiment completed.")\n        return False\n\n'
            '    current_iter = 0\n    next_prompt = """\nGreat job!')
    replace('    next_prompt = """\nPlease modify `notes.txt`',
            '    if return_code != 0:\n        print("Plotting did not complete.")\n        return False\n'
            '    next_prompt = """\nPlease modify `notes.txt`')
    compile(modified, str(source), "exec")
    patch.write_text("".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                                fromfile="original/ai_scientist/perform_experiments.py",
                                                tofile="teaching/ai_scientist/perform_experiments.py")), encoding="utf-8")
    source.write_text(modified, encoding="utf-8")
    result = {"status": "configured_not_run", "commit": pin,
              "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "changes": patch.name,
              "scope": "Preserve failed run artifacts/code/stderr; reject incomplete experiments or plotting. No API or whole-session budget changes."}
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "doctor", "configure-api", "configure-failures"))
    parser.add_argument("project", nargs="?", choices=tuple(PROJECTS))
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "upstream")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", help="Explicit compatible-API research model (configure-api only)")
    parser.add_argument("--review-model", help="Explicit compatible-API reviewer (configure-api only)")
    args = parser.parse_args()
    if args.action == "prepare" and not args.project:
        parser.error("prepare requires a project")
    if args.action == "configure-api" and (args.project != "v1" or not args.model or not args.review_model):
        parser.error("configure-api requires v1, --model and --review-model")
    if args.action == "configure-failures" and args.project != "v1":
        parser.error("configure-failures requires v1")
    try:
        if args.action == "configure-failures":
            result = configure_v1_failures(args.root / "v1")
        elif args.action == "configure-api":
            result = configure_v1_api(args.root / "v1", args.model, args.review_model)
        else:
            result = doctor() if args.action == "doctor" else prepare(args.project, args.root, args.device)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
