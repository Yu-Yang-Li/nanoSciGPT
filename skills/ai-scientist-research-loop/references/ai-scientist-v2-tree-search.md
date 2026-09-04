---
name: ai-scientist-v2-tree-search
description: 用于学生已经完成 AI Scientist v1 单路线研究后，继续建立两条可比较的研究路线、逐条执行、恢复状态并按同一评价标准作出取舍。不要用于首次训练模型或尚无 v1 workflow_state.json 的情况。
---

# AI Scientist v2：比较多条研究路线

这节课从 v1 已完成的一条路线继续。先把已经跑完的路线和一个备选路线放进同一棵研究树，再由学生逐条批准执行。这里复现的是“多路线搜索与取舍”这一教学过程，并不是原版 The AI Scientist v2 的完整系统。

## 开始时先问

如果学生只说“继续 v2”，先请他提供 v1 目录中的 `workflow_state.json`。拿到文件后，说明将沿用同一个 V0、同一个评价指标和同一个门槛，不重新定义研究问题。

## 第一步：建立两条路线

```powershell
python -m autoresearch.v2 init --from-v1 [v1目录]/workflow_state.json --out-root out/ai-scientist-v2
```

这一步只生成 `tree_state.json`，不训练模型。请向学生说明：`route-1` 是 v1 已完成的路线，`route-2` 是等待执行的备选路线。

## 第二步：执行下一条路线

学生看过 `tree_state.json` 并同意后再运行：

```powershell
python -m autoresearch.v2 run-next --state out/ai-scientist-v2/[领域]/tree_state.json --approve
```

一次只运行一条路线。完成后指出这次改动、模型结果、是否达到原有门槛，以及结果保存在哪里。不要把一次课程规模运行写成科学发现。

## 中断后继续

```powershell
python -m autoresearch.v2 status --state out/ai-scientist-v2/[领域]/tree_state.json
```

`tree_state.json` 记录每条路线的状态和执行次数。已完成的路线不会再次执行。

如果当前路线显示为 `running`，说明上一次进程可能在写出最终结果前中断。不要重新建立研究树；沿用同一份状态文件再次执行 `run-next --approve`。续跑后 `attempts` 会增加，并记录 `resumed_after_interruption: true`。

## 最后取舍

```powershell
python -m autoresearch.v2 decide --state out/ai-scientist-v2/[领域]/tree_state.json
```

`route_decision.json` 只按 v1 留下的数值评价器选择保留路线；若没有路线达到门槛，则保留 V0。课堂实现不合并不同代码分支，也不声称复现原系统。状态中的 `reproduces_original_system` 始终为 `false`。

## 对学生的回答方式

每轮只推进一项：建立路线、执行一条路线、查看状态或作出取舍。先说当前已有哪份证据，再给出下一条短命令；如果缺少 v1 状态文件，就停在文件定位，不猜测实验结果。
