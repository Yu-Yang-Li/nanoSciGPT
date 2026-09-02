---
name: ai-scientist-research-loop
description: Use when a student has a nanoSciGPT model result and wants to iterate it with autoresearch, compare candidate changes, or connect the model to a calculation, experiment, or observation that can influence the next research step.
metadata:
  short-description: 从一次模型训练走到可复查的研究迭代
---

# AI Scientist Research Loop

从学生刚刚跑出的模型接着做：提出一次改动，用同一种评价方法重新运行，再根据结果决定继续还是停止。等这一小段真正走通以后，再讨论更长的 AI Scientist 研究过程。

## 先接住上一份结果

学生说“继续优化”“进入 autoresearch”时，先沿用前文的领域和运行目录。通常从 `out/classroom/<domain>/run_report.json` 开始；找不到这份文件时，直接说明还缺第一份可比较结果，并带他回到 nanoSciGPT 示例，而不是凭空设计下一轮。

先用一句话概括现状：哪个领域、当前评价项是什么、目前结果是多少。然后和学生商量这一轮想改什么。课程仓库的基础版本目前演示的是增加训练预算；数据、划分、评价方法和其他设置保持不变，因此它适合讲“同口径迭代”，不代表已经覆盖所有研究改动。

## 先看计划，再决定要不要跑

先生成计划：

```powershell
python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --plan_only
```

把 `protein` 换成学生的领域。默认 `python` 不可用时，先运行 `scripts/find_course_python.ps1 -RequiredModules numpy,torch` 找到可用环境。打开生成的 `iteration_spec.json`，用自然语言告诉学生：这轮保持了什么、改变了什么、怎样比较。学生明确说可以运行以后，再执行：

```powershell
python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --fresh --auto_approve
```

命令完成后读取 `comparison.json` 和 `research_state.json`。说明基线值、候选值、差值、比较门槛和 `next_action`；其中任何一项缺失时先补齐记录，不凭印象宣布改进。评价没有达到门槛时，也把它当作一份完整结果：说明这次改动没有带来足够变化，并依据 `next_action` 停止或重新提出一个小问题。

## 从 autoresearch 讲到 AI Scientist

学生已经看懂“改动—运行—比较—下一步”以后，再把研究过程拉长：

- The AI Scientist v1 进一步把研究想法、相关工作、实验安排、图表和写作连接成一条计算研究路线；
- v2 进一步比较多条候选路线，并决定保留、合并或停止。

这两项在当前仓库中是教学拆解，不是原项目的完整复现。讨论它们时沿用学生刚才的模型、评价项和失败记录，帮助他画出研究路线即可，不生成不存在的论文评阅或实验结果。

## 接入真实科学过程

学生想把模型接到自己的研究时，只补问一个现实问题：

> 下一步你实际能够得到的是计算/模拟结果、实验测量，还是观测数据？

把这个返回结果写成下一轮能够读取的量，并说明什么变化会让路线继续、修改或停止。没有可返回的结果时，这是一份研究设计；得到机器可读结果并且它真的改变下一步时，才开始形成持续迭代。

天文教学可以接 StarWhisper 的观测规划或虚拟观测，但要沿用工具实际返回的身份：真实观测、模拟、回放或设计。人的工作始终保留在问题设定、高风险授权、关键结果复核和最终结论上。

## 一次自然的接续

学生：`蛋白质模型已经跑完了，帮我继续优化。`

助教：

> 我们先用刚才的验证损失作为比较标准，只增加训练预算，其他设置保持原样。我先生成这一轮计划，不开始训练；计划出来后我们一起看是否值得跑。
