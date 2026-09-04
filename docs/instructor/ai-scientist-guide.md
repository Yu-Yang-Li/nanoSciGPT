# AI Scientist 讲义：三段式教学指南

面向：AI4S 实训营"科学模型专题实训（二）"B 线（AI Scientist 系统）
时长：约 25 分钟（对应 PPT P11–P16 的实操部分）
前置：学生已跑通 A 线 nanoSciGPT 四域训练；autoresearch/ 目录可直接运行

---

## 课程定位：一句话

> "一个 AI Scientist = 假设生成（想）+ 实验闭环（做）+ 论文与审稿返修（说）。三段能力缺一不可，而且每一段都有它自己的失败模式。"

这不是"Agent 越多越自主"的叙事。每一阶段的核心问题都是：**这一段解决了什么人工瓶颈？引入了什么新的失败模式？人站在哪里？**

---

## 时间线与代码的对应关系

| 阶段 | 讲授主线（时间线） | 代码模块 | 参考系统 |
|---|---|---|---|
| S1 假设生成 | 科研入口从"人读文献"到"系统生成+专家评分" | `autoresearch/hypothesis.py` | AstroInsight (EPJ Data Science 2026) |
| S2 实验闭环 | 反馈进入下一步；工具从被调用变为被编排 | `autoresearch/experiment.py` | StarWhisper Telescope (Comms Eng 2025) |
| S3 论文工作流 | 从实验结束到"能声称什么"的文字治理 | `autoresearch/paper.py` | 讲师自身 agentic research 实践 |

三段共用同一个 `ResearchState`（`state.py`）——**跨阶段的研究状态是三段成为一个系统的关键**，而不是三段各自为政。

---

## S1 假设生成（约 6 分钟）

### 讲什么

AstroInsight 的四阶段：conception（从数据统计生成想法）→ iterative refinement（打分、批评、重写）→ expert validation（人类专家 1-6 分评 novelty/feasibility）→ knowledge integration（接受的进入 idea bank）。

核心教学点：**LLM 生成的想法不是假设，经过评分、重写、专家验证的才算。** 新颖度 3+/6 是 AstroInsight 的接受线。

### 操作（现场跑）

```bash
python -m autoresearch.hypothesis --domain protein --auto_approve
```

指着输出讲：

> "三个想法全部基于真实的数据统计（vocab=22, 450 条序列）——这不是 LLM 幻觉，是从你的数据里长出来的想法。IDEA-C 是最有教学价值的：它预测'这个规模做不出基座模型'——这个'负向假设'比正向假设更接近真实科研。"

### 板书要点

- conception 的输入是**数据统计**，不是空想。每个想法都带 grounding（用什么 metric、和什么 baseline 比）。
- refinement 是**结构化的**：novelty + feasibility - vagueness，不是"让 GPT 再想想"。
- expert validation 是人工门：低分想法死掉，不进假设库。

### 停止边界

S1 的产物是"经过验证的假设"，不是"科学发现"。想法被接受≠想法被支持；支持要靠 S2 的实验证据。

---

## S2 实验闭环（约 9 分钟）

### 讲什么

StarWhisper Telescope 的闭环：观测计划（site/time-specific）→ function call 执行（望远镜控制+图像获取）→ 实时分析（pipeline 提取 transient）→ 动态 follow-up（探测触发下一轮观测）。

核心教学点：**闭环的判定标准不是"跑了多少步"，而是"反馈是否改变了下一步"。** 如果分析结果不改变动作，那是流水线不是闭环。

### 操作（现场跑）

```bash
python -m autoresearch.experiment --domain protein --auto_approve
```

指着输出讲三个点：

1. **计划是数据结构**：`plan: ['prepare', 'train_v0', 'train_extended', 'sample']`——每个步骤都有 reason，不是自由文本。
2. **人工门真实存在**：train_extended 需要批准。拒绝它，实验停在合同边界，这是设计而非故障。
3. **失败被保留**：gain +0.0000（fail）——200 iter 没比 100 iter 更好，这个小数据上的收益饱和本身就是发现。

### 板书要点

- **工具合同**（`tools.py` 的 `CONTRACTS`）：系统只能调用声明过的工具，每个工具有命令模板、产物声明、预算上限、是否需要人工批准。不在合同里的操作被拒绝——这是 AI Scientist 安全边界的第一层。
- **形式化评价器**（`evaluator.py`）：design / ran / evaluated 三级，绝不混层。"工具跑过"和"科学通过"是两回事。
- **反馈改变下一步**（`decide()`）：训练失败→停止；评估通过→加预算；增益不够→继续采样（失败也是发现）。

### 停止边界

S2 证明了"循环可执行且反馈改变行动"。但它不能证明"科学增量"——那是外部验证的事，本仓库的教学 fixture 永远到不了那一层。

---

## S3 论文工作流（约 7 分钟）

### 讲什么

这第三段没有参考已发表论文，而是讲师自己的 agentic research 实践：研究状态（locked evidence）→ 草稿组装（只用记录的证据，绝不编数字）→ 结构化审稿（逐条 claim 对照证据等级）→ 逐条返修（每处修改对应一个 review finding）→ 事实审计（每个数字必须溯源到 run artifact）→ 结论边界（能声称什么/不能声称什么）。

核心教学点：**写作 Agent 的金规则——可以用锁定证据，但不能重新分析数据或强化结论。**

### 操作（现场跑）

```bash
python -m autoresearch.paper --domain protein
```

指着输出讲：

> "看这三个 review findings：F1 和 F3 都是 'overstated'——'ran' 级别的证据被写成了结果，审稿把它降级。F2 是 'boundary'——负结果被显式保留而不是删掉。这就是 agentic 写稿和普通生成的区别：**每处修改都可追溯到一条审稿意见**。"

### 板书要点

- 草稿组装只从 `state.data` 生成，不从想象生成——**规则在构造层强制执行**。
- 逐条返修：每处改动对应一条 finding，和 LAMOST 论文逐条对照的做法同构。
- 事实审计：孤儿数字（无法溯源的数字）会 fail——这就是 `audit_manuscript_numbers.py` 的教学版。

### 停止边界

S3 产出的是"治理过的稿子"，不是"同行评审过的论文"。没有外部验证，任何科学声明都不能升级。

---

## 全流程串联演示（约 3 分钟）

```bash
python -m autoresearch.pipeline --domain protein --fresh --auto_approve
```

指着完整输出串一遍：

> "S1 想出三个假设并让专家评分（IDEA-C 说'做不出基座'）→ S2 执行了四个合同工具、经历一次人工门、保留一次失败 → S3 把这些证据组装成稿、发现三处 overstate/边界问题、逐条返修、事实审计通过、最后写下'能声称什么/不能声称什么'。**这就是一个完整的、每一层都可追溯的 AI Scientist 最小闭环。**"

---

## 课堂判断卡：系统到底属于哪一层？

给学生一个系统描述，让他们判断属于哪一层：

| 系统行为 | S1 | S2 | S3 |
|---|---|---|---|
| LLM 生成 100 个想法但没有评分机制 | ✗ | | |
| 自动运行 10 个实验但结果不影响下一步 | | ✗ | |
| 生成论文但数字无法溯源 | | | ✗ |
| 想法经过专家评分后进入实验 | ✓ | | |
| 实验结果改变下一轮实验设计 | | ✓ | |
| 每处修改可追溯到审稿意见 | | | ✓ |

答案规则：**没过 S1 的评分门，想法不能进实验；没过 S2 的反馈门，实验不能写成结果；没过 S3 的审计门，稿子不能声称结论。**

---

## 诚实边界（必须讲）

1. **本仓库刻意不用 LLM**。规则驱动让每一步可读、可追溯，学生能逐行理解"反馈怎么改变下一步"。真实 AI Scientist 的策略层由 LLM 承担，但架构不变。
2. **教学数据是 fixture**。450 条蛋白序列不会产生真实迁移增益——IDEA-C 的"负向假设"被 S2 的 gain=0 部分支持、被 A 线的 transfer_probe 的 -5% 完全支持。**负结果就是这门课最重要的教学结果。**
3. **S3 的审稿是结构化审稿，不是同行评审**。它检查证据等级和数字溯源，不检查科学新颖性或领域重要性。

---

## 时间分配总表

| 段 | 内容 | 建议 |
|---|---|---:|
| 开场 | 三段式定位：想—做—说 | 1 min |
| S1 | 假设生成（AstroInsight 架构） | 6 min |
| S2 | 实验闭环（StarWhisper 架构） | 9 min |
| S3 | 论文工作流（讲师实践） | 7 min |
| 串联 | pipeline 全流程 + 判断卡 | 3 min |
| 收束 | 诚实边界 + 作业指引 | 1 min |

---

## 学生动手点

1. **S1**：改 `conceive()` 的 idea 模板，加一个自己的假设方向，看它能不能过评分门。
2. **S2**：在人工门处拒绝批准，看状态如何停在合同边界（`next_action: stop`）。
3. **S3**：故意在草稿里加一个编造的数字（如 "accuracy 0.95"），看事实审计如何报 orphaned number。
4. **跨域**：换 `--domain dna/smiles` 跑 pipeline，对比三个域的 idea bank 和证据等级分布。
