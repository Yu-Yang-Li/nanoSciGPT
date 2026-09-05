"""Record a sequential Codex CLI teaching test from a supplied list of turns."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="不提供时使用Codex CLI自身的默认模型")
    args = parser.parse_args()
    prompts = json.loads(args.inputs.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    codex = shutil.which("codex")
    if not codex:
        raise FileNotFoundError("codex CLI is not available")
    session_id = None
    completed_turns = 0
    model_label = args.model or "CLI当前默认（本脚本未覆盖模型设置）"
    transcript = ["# 连续CLI教学实测", "", f"模型：{model_label}", ""]
    for index, prompt in enumerate(prompts, 1):
        command = [codex, "exec"]
        if session_id:
            command += ["resume", session_id]
        else:
            command += ["-C", str(ROOT), "--sandbox", "workspace-write"]
        command += ["-c", 'sandbox_mode="workspace-write"', "-c", 'approval_policy="never"',
                    "-c", "features.memories=false"]
        if args.model:
            command += ["--model", args.model]
        command += ["--json", "-o", str(output / f"turn-{index}-reply.md"), prompt]
        environment = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        log = output / f"turn-{index}.jsonl"
        (output / f"turn-{index}-input.txt").write_text(prompt, encoding="utf-8")
        (output / f"turn-{index}-command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        timed_out = False
        with log.open("w", encoding="utf-8") as stdout, (output / f"turn-{index}-stderr.txt").open("w", encoding="utf-8") as stderr:
            try:
                run = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=600)
            except subprocess.TimeoutExpired:
                timed_out = True
                run = subprocess.CompletedProcess(command, 124)
                stderr.write("\nCLI exceeded the 600-second capture limit; partial events retained.\n")
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                session_id = event["thread_id"]
        reply_path = output / f"turn-{index}-reply.md"
        reply = reply_path.read_text(encoding="utf-8") if reply_path.exists() else "未得到最终回复；见原始日志。"
        if timed_out:
            reply += "\n\n本轮CLI超过600秒记录时限，已停止等待；以上内容不作为完整回复验收。"
        transcript += [f"## 第{index}轮", "", "学生原文：", "", prompt, "", "CLI实际回复：", "", reply,
                       "", f"原始日志：{log.name}；退出码：{run.returncode}", ""]
        (output / "dialogue.md").write_text("\n".join(transcript), encoding="utf-8")
        print(f"turn {index}: exit={run.returncode}, session={session_id}", flush=True)
        sandbox_error = "apply deny-read ACLs" in ((output / f"turn-{index}-stderr.txt").read_text(encoding="utf-8")
                                                  + log.read_text(encoding="utf-8"))
        failed = bool(run.returncode or not session_id or not reply_path.exists() or sandbox_error)
        completed_turns += int(not failed)
        (output / "session.json").write_text(json.dumps({"session_id": session_id, "model": args.model,
                                                        "model_selection": "explicit" if args.model else "cli_default",
                                                        "sandbox": "workspace-write",
                                                        "persistent_memory_enabled": False,
                                                        "stop_reason": "cli_timeout" if timed_out else None,
                                                        "attempted_turns": index, "completed_turns": completed_turns,
                                                        "classroom_execution_passed": False,
                                                        "note": "Dialogue capture only; inspect actual tool executions and outputs separately."}, indent=2), encoding="utf-8")
        if sandbox_error:
            (output / "blocked.json").write_text(json.dumps({"reason": "windows_sandbox_acl", "turn": index,
                                                           "classroom_execution_passed": False}), encoding="utf-8")
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
