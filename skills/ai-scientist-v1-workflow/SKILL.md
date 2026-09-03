---
name: ai-scientist-v1-workflow
description: Use when a student has a completed AutoResearch comparison and wants to extend that single computational route into related work, a result table and figure, a traceable draft, and a classroom evidence review.
metadata:
  short-description: 把一轮模型比较接成长一些的计算研究过程
---

# AI Scientist v1 工作流

沿用学生刚刚完成的 AutoResearch，不再训练新模型。这个 Skill 把同一条路线向前接到研究问题和相关工作，向后接到结果表、图、证据映射、短稿和审查。它复现的是 v1 的工作顺序，不是官方系统的 GPU 规模、在线文献查重、LLM 写代码或会议审稿。

## 先看现有证据

从 `out/autoresearch/<domain>/` 读取 `iteration_spec.json`、`candidate_run_report.json`、`comparison.json` 和 `research_state.json`。先说清研究问题、唯一改动、V0与V1结果以及当前是继续还是停止。缺少正式比较时，回到 AutoResearch，不用虚构结果填论文框架。

先生成一条研究路线和离线相关工作：

```powershell
python -m autoresearch.v1 --domain protein --autoresearch-dir out/autoresearch/protein --out-dir out/ai-scientist-v1/protein --plan-only
```

查看 `plan.json` 和 `related_work.json`。相关工作来自仓库内已经列明来源的课程目录，`novelty_assessment` 固定为 `not_performed_offline`；它提供阅读入口，不代表完成新颖性检索。

如果输出目录里已经有一份完成的研究材料，先请学生换一个新目录。确实要重做时才显式加入 `--overwrite`；该选项只替换本工作流生成的文件。由 `--plan-only` 生成且内容未变的计划，可以在同一目录继续 `--confirm-plan`。

## 确认路线以后整理结果

学生确认这条路线后运行：

```powershell
python -m autoresearch.v1 --domain protein --autoresearch-dir out/autoresearch/protein --out-dir out/ai-scientist-v1/protein --confirm-plan
```

这一步只整理已有证据。依次查看 `results.json`、`results.csv`、`figures/v0-v1.svg`、`evidence_map.json`、`draft.md`、`review.json` 和 `claim_boundary.md`。每个数字都回指同一份 `comparison.json`；没有 `evaluated` 结果时只留下 `workflow_status.json`，不会生成稿件。

结果没有达到门槛也可以进入短稿，但必须写成这条路线的负结果或停止依据。规则审查的最高结论是 `ready_for_human_review`，表示材料可以交给人继续检查，不表示论文已经通过评审。

学生说“把刚才结果接成 AI Scientist v1”时，可以这样回应：

> 我先沿用刚才唯一的一项改动，把研究问题、相关工作和已有比较接成一条路线；这一步只生成计划，不重跑模型，也不开始写稿。计划出来后我们再决定是否整理成完整材料。
