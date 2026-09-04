---
name: ai-scientist-research-loop
description: Use when a student has a runnable model and wants to improve it, compare a research route, or connect computation to a traceable research record.
metadata:
  short-description: 从一次模型运行走到可复查的计算研究
---

# AI Scientist研究流程

这个入口把三件事接在一起：先围绕已有模型做一次可比较的改动，再把一条路线整理成计算研究，最后让多个路线在同一评价标准下竞争。学生不需要知道这些阶段的名字，也不需要一次输入完整命令；助教根据已有文件推进下一步。

## 先看学生已经有什么

先检查当前领域的`out/classroom/<domain>/run_report.json`。没有运行报告时，回到基线或科学语言入口；不要凭空开始迭代。

有运行报告时，用一句话说明当前模型、数据和评价项，然后只问一个问题：

> 你想先改模型、改数据表示，还是先把这次结果整理成一条研究路线？

如果学生已经说“继续优化”，直接进入下一节；如果只给了一个想法，先把它写成可比较的设置，不立即训练。

## 第一步：AutoResearch，一次可比较的模型迭代

学生已有`run_report.json`并想继续优化时，先生成设置：

```powershell
python -m autoresearch.experiment --domain <domain> --baseline_run out/classroom/<domain>/run_report.json --out_root out/autoresearch --plan_only
```

打开`iteration_spec.json`，和学生核对：本轮改哪一处、哪些内容保持不变、用什么评价、最多运行多久、什么情况下停止。学生同意后才运行：

```powershell
python -m autoresearch.experiment --domain <domain> --baseline_run out/classroom/<domain>/run_report.json --out_root out/autoresearch --fresh --auto_approve
```

运行后先读`candidate_run_report.json`、`comparison.json`和`research_state.json`。比较没有达到预先约定的门槛，也要保存这一轮并说明停止原因；不把小幅波动写成“优化成功”。

## 第二步：AI Scientist v1，把一条路线整理成计算研究

只有已有完整的AutoResearch比较后才进入v1。先生成计划，不重跑模型：

```powershell
python -m autoresearch.v1 --domain <domain> --autoresearch-dir out/autoresearch/<domain> --out-dir out/ai-scientist-v1/<domain> --plan-only
```

学生确认路线后再整理：

```powershell
python -m autoresearch.v1 --domain <domain> --autoresearch-dir out/autoresearch/<domain> --out-dir out/ai-scientist-v1/<domain> --confirm-plan
```

依次查看`results.json`、`results.csv`、`figures/v0-v1.svg`、`evidence_map.json`、`draft.md`、`review.json`和`claim_boundary.md`。这些文件整理的是已有运行证据；离线相关工作只提供阅读入口，不代表完成新颖性检索或论文评审。

## 第三步：AI Scientist v2，比较多条研究路线

只有v1留下`workflow_state.json`时才建立研究树：

```powershell
python -m autoresearch.v2 init --from-v1 <v1目录>/workflow_state.json --out-root out/ai-scientist-v2
```

学生看过树状态并同意后，一次执行一条路线：

```powershell
python -m autoresearch.v2 run-next --state out/ai-scientist-v2/<domain>/tree_state.json --approve
```

中断后先查看状态，不重新开始：

```powershell
python -m autoresearch.v2 status --state out/ai-scientist-v2/<domain>/tree_state.json
```

如果状态显示当前路线仍是`running`，沿用同一份`tree_state.json`再次执行上一条`run-next --approve`命令。程序会把它记为中断后的续跑、增加`attempts`，并继续保留已经完成的路线；不要重新`init`一棵树。

所有路线完成或明确停止后再作取舍：

```powershell
python -m autoresearch.v2 decide --state out/ai-scientist-v2/<domain>/tree_state.json
```

v2只比较课程中已经定义的路线和评价器，不声称复现原版The AI Scientist v2，也不把课程规模运行写成科学发现。

## 每轮怎样和学生说

一次只推进一个动作：生成设置、运行一轮、查看比较、建立路线、执行一条路线、恢复状态或作出取舍。先说明已经存在的文件，再给下一条短命令；结果回来以后再解释，不预演成功结果。

所有输出都标明它属于“设计”“已运行”“已比较”还是“已整理”。候选、模拟、体外验证和科学结论分开记录；没有真实结果时不生成分数、图表或论文结论。

## 参考材料

- [一次模型迭代](references/autoresearch-model-iteration.md)：AutoResearch的详细字段和边界。
- [单路线研究](references/ai-scientist-v1-workflow.md)：v1的文件顺序和证据映射。
- [多路线比较](references/ai-scientist-v2-tree-search.md)：v2的状态恢复和取舍规则。
