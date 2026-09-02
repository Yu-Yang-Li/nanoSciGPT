---
name: autoresearch-model-iteration
description: Use when a student already has a completed nanoSciGPT run_report.json and wants to test one controlled model change, compare it with V0, or decide whether that branch should continue.
metadata:
  short-description: 从模型V0完成一次同口径迭代
---

# AutoResearch 模型迭代

从学生已经跑出的模型继续：先把下一轮写成一份可检查的设置，再运行一个候选模型，最后按同一评价项比较 V0 与 V1。这个 Skill 只负责一条路线的一次改动；研究想法、文献、写作和多路线探索留给后续课程。

## 接住上一份结果

学生说“继续优化”时，沿用前文的领域和 `run_report.json`。先读报告中的实际命令、训练配置、数据路径和评价项，用一句话复述当前 V0。找不到报告，或者状态不是 `completed`，就说明还缺一份可比较的 V0，不生成候选分数。

先从仓库根目录生成计划。下面以蛋白质为例：

```powershell
python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --plan_only
```

默认 `python` 不可用时，运行 `scripts/find_course_python.ps1 -RequiredModules numpy,torch`。打开 `out/autoresearch/protein/iteration_spec.json`，和学生一起看三件事：V0 的原始设置、本轮唯一改变的训练步数、保持不变的数据与模型参数。此时还没有启动新训练。

## 学生同意以后再运行

学生明确同意这份设置后，执行：

```powershell
python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --auto_approve
```

候选模型写在本轮目录，不会回写 V0。命令完成后依次读取：

- `candidate_run_report.json`：V1 的真实命令、改动项、返回状态和产物；
- `comparison.json`：V0、V1、差值、方向、门槛和下一步；
- `research_state.json`：这一轮证据以及 `retain_candidate` 或 `stop_branch`。

结果没有达到门槛时，仍然是一轮完整实验：说明这次增加训练预算没有带来足够改善，并保留停止记录。输出目录里已经有候选结果时，换一个新的 `--out_root`，不覆盖旧轮次。

## 一次自然的接续

学生：`蛋白质模型已经跑完，帮我继续优化。`

助教：

> 我先读取刚才的 V0，用同一份数据、模型结构和评价项，只把训练步数增加一倍。先生成设置，不开始训练；设置出来后我们再决定是否运行。
