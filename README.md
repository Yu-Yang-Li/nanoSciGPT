# nanoSciGPT

一个面向教学的"科学领域语言模型"最小整合框架：同一套 GPT 核心代码，通过替换 **tokenizer + 数据准备** 即可从文本迁移到蛋白质、DNA、分子（SMILES）等科学对象。

## 教学定位

这门课不是教你怎么写 Transformer，而是回答三个问题：

1. **预训练到底在做什么？** —— 下一 token 预测为什么能学到有用的表征；
2. **科学对象怎么"语言化"？** —— 氨基酸序列、DNA 碱基、SMILES 字符串本质上都是离散符号序列，可以用同一套方法处理；
3. **什么时候值得做基座模型？** —— 数据规模、任务复用、迁移收益决定你该用专用模型还是预训练基座。

## 架构：共享核心 + 领域专用

```
nanoscigpt/
├── core/                    # 【全部领域共享】
│   ├── gpt.py               # GPT decoder（causal attention）
│   ├── trainer.py           # 训练循环：batch、loss、ckpt、eval
│   ├── sampler.py           # 自回归采样
│   ├── tokenizer.py         # 字符级 tokenizer 基类
│   └── dataset.py           # memmap 数据集 + 变长序列 padding
├── domains/                 # 【每个领域必须自己写】
│   ├── text/                # Shakespeare（教学基线）
│   ├── protein/             # UniProt / PDB 氨基酸序列
│   ├── dna/                 # 基因组 FASTA
│   └── smiles/              # 分子 SMILES
└── tasks/                   # 下游探针（迁移评测）

autoresearch/                # 【B线：与仓库互动的虚拟 AI Scientist】
├── tools.py                 # 工具合同：唯一允许的操作入口
├── evaluator.py             # 形式化评价器：design/ran/evaluated 证据分级
├── state.py                 # 跨轮研究状态：假设、证据、未决问题
└── run.py                   # 主循环：反馈改变下一步 + 人工授权门
```

**核心设计原则**（也是课堂上要讲的）：已核实的三个 nanoGPT 魔改案例（prot-gpt、nanoGPT-DNA、dnaGPT）的全部差异都落在 tokenizer、数据准备、变长处理、评测四处；模型结构、训练循环、采样逻辑完全不变。所以本框架把前者做成每领域必写的"领域插件"，后者做成共享核心。

## 快速开始

```bash
# 1. 安装（只需 torch + numpy，无需 GPU）
pip install -e .

# 2. 准备数据（以 text 为例）
python -m nanoscigpt.domains.text.prepare

# 3. 训练（CPU 几分钟）
python -m nanoscigpt.core.trainer --domain text --max_iters 2000

# 4. 采样
python -m nanoscigpt.core.sampler --domain text
```

## 四个领域的最小可运行链路

| 领域 | 数据 | 模式 | prepare 命令 | train 命令 |
|---|---|---|---|---|
| 文本 | tiny Shakespeare（1MB，自动下载） | 流式 | `python -m nanoscigpt.domains.text.prepare` | `python -m nanoscigpt.core.trainer --domain text` |
| 蛋白 | UniProt reviewed（在线取前 N 条） | 独立序列 | `python -m nanoscigpt.domains.protein.prepare --size 500` | `python -m nanoscigpt.core.trainer --domain protein` |
| DNA | 本地 FASTA（如 chr21.fa 切片） | 流式 | `python -m nanoscigpt.domains.dna.prepare --fasta <path>` | `python -m nanoscigpt.core.trainer --domain dna` |
| SMILES | DeepChem ESOL（1128 条分子） | 独立序列 | `python -m nanoscigpt.domains.smiles.prepare` | `python -m nanoscigpt.core.trainer --domain smiles` |

## 实测结果（本机 CPU，2026-08-28）

四个领域均在本机 Python 3.12 + torch 2.12 CPU 环境完成端到端验证（数据准备 → 100 iter 训练 → 采样）：

| 领域 | 词表 | 数据量 | 100 iter val loss | 采样样例 |
|---|---:|---|---:|---|
| 文本 | 65 | 1.1M chars | 4.20 → **2.66** | 伪莎士比亚字符流 |
| 蛋白 | 22 | 500 条 UniProt | 3.10 → **2.84** | 真实氨基酸序列 |
| DNA | 4 | 350k bases | 1.40 → **1.32** | 碱基序列 |
| SMILES | 33 | 1128 条分子 | 3.40 → **1.68** | 合法 SMILES 字符 |

所有命令均可在普通学生电脑 CPU 上几分钟内完成，无需 GPU。

## 教学叙事：从 nanoGPT 到科学基座

这门课的 A 线（领域基座模型）用这个仓库贯穿：

1. **A-V0 专用模型**：在 10 个 CIF 上训练 CGCNN——没有预训练，因为数据太少；这解释了"专用模型为什么是默认起点"。
2. **A1 科学对象可预训练**：跑 text 域（和 nanoGPT 一样）+ protein/DNA/SMILES 域——同一个架构，换掉 tokenizer 和数据就完成了"科学对象语言化"。
3. **A2 表征可迁移**：冻结预训练模型，取中间层 embedding 训练简单分类器——演示"预训练收益怎么度量"。
4. **A3 基座能力统一**：多个领域共享同一个 core/，只在 domain 层扩展——这就是"基座"的最小结构隐喻：核心复用、领域插拔。

## A线四级进阶实验（2026-08-28 实测）

课程 A 线的完整阶梯，每一级都有可运行命令和实测结果：

| 阶级 | 教学问题 | 命令 | 实测结果 |
|---|---|---|---|
| A1 科学对象语言化 | 换 tokenizer 就能换领域吗？ | 见上方四域表 | 四域全部跑通，loss 均下降 |
| A2a 换预训练目标 | CLM 和 MLM 有什么区别？ | `python -m nanoscigpt.tasks.objective_contrast` | CLM 2.97→2.76；MLM 3.01→2.81（同一蛋白数据） |
| A2b 表征迁移 | 我们的预训练到底带来什么？ | `python -m nanoscigpt.tasks.transfer_probe` | one-hot 100% / 随机编码器 98.3% / 预训练编码器 95%——迁移增益为负（诚实结果） |
| A3a 多任务接口 | 共享编码器能否服务多任务？ | `python -m nanoscigpt.tasks.multihead` | 共享编码器：分类 100% + 回归 MAE 0.27（合成双任务） |
| A3b 路线决策 | 什么时候不该训练基座？ | `python -m nanoscigpt.tasks.route_decision` | 五问决策链：数据不足→正确降级为专用模型 |

**A2b 的核心教学价值**：迁移增益为负——450 条序列的"预训练"不如 one-hot。这不是失败，是课程要证明的事：数据规模不够时基座主张不成立。ESM 的 2.5 亿条序列与我们的 450 条相差六个数量级，"机制相同、规模决定成败"。

## B线：autoresearch——三段式虚拟 AI Scientist

A 线讲“模型怎么建”，B 线讲“科研过程怎么闭环”。`autoresearch/` 是一个**规则驱动**（刻意不用 LLM）的虚拟科学家，它只能通过声明过的工具合同操作本仓库，每一步都被形式化评价器检验，研究状态跨轮持久化。

### 三段式架构（对应 AI Scientist 讲授时间线）

| 段 | 能力 | 代码 | 参考系统 |
|---|---|---|---|
| S1 假设生成 | 想法→评分→专家验证→假设库 | `hypothesis.py` | AstroInsight (EPJ Data Science 2026) |
| S2 实验闭环 | 计划→工具合同执行→评价器分析→反馈决定下一步 | `experiment.py` | StarWhisper Telescope (Comms Eng 2025) |
| S3 论文工作流 | 证据组装→结构化审稿→逐条返修→事实审计 | `paper.py` | 讲师 agentic research 实践 |

三段共用 `ResearchState`（`state.py`）——跨阶段的研究状态是三段成为一个系统的关键。

完整讲义见 [docs/ai-scientist-guide.md](docs/ai-scientist-guide.md)。

### 五个教学概念的落点

| B线概念 | 代码落点 | 课堂观察点 |
|---|---|---|
| 可执行动作 | `tools.py` 的 `CONTRACTS` | 每个动作预先声明命令模板、产物、预算 |
| 工具合同 | `run_tool()` | 不在合同里的操作被直接拒绝 |
| 形式化评价器 | `evaluator.py` | design / ran / evaluated 三级，绝不混层 |
| 反馈改变下一步 | `run.py` 轮次策略 | H1 失败→停止；H2 失败→记录为发现而非报错 |
| 跨轮研究状态 | `state.py` + `research_state_<domain>.json` | 重跑不 `--fresh` 直接恢复终态 |
| 人工授权与结论边界 | `--auto_approve` 门 + 结论轮 | 预算增加必须人工批准；结论写清“能声称什么/不能声称什么” |

### 快速开始

```bash
# 三段式全流程（推荐，约 20 秒）
python -m autoresearch.pipeline --domain protein --fresh --auto_approve

# 或分阶段演示
python -m autoresearch.hypothesis --domain protein --auto_approve
python -m autoresearch.experiment --domain protein --auto_approve
python -m autoresearch.paper --domain protein

# 旧接口（保留兼容）
# 文本域全流程（约 20 秒/轮）
python -m autoresearch.run --domain text --fresh --auto_approve

# 蛋白域（含迁移探针，会出现“迁移增益为负”的诚实结果）
python -m autoresearch.run --domain protein --fresh --auto_approve

# 不带 --auto_approve：人工门会真实等待输入（课堂演示用）
python -m autoresearch.run --domain text --fresh

# 跨轮恢复：重跑同一域，直接到终态
python -m autoresearch.run --domain text
```

### 实测结果（本机 CPU，2026-08-29）

四个域均跑通完整闭环，且失败案例被正确记录为科学发现：

| 域 | H1 预训练 | H2 预算加倍 | H3 迁移 |
|---|---|---|---|
| text | 支持（val 2.66） | 支持（+0.128） | 不适用（开放问题） |
| protein | 支持（val 2.84） | **反驳**（+0.042 未达标） | **反驳**（delta −0.033，规模决定） |
| dna | 支持（val 1.34） | 反驳（收益≈0） | 不适用（开放问题） |
| smiles | 支持（val 1.68） | 支持（+0.214） | 不适用（开放问题） |

蛋白域两个“失败”正是课程核心：**数据规模不够时，预算和迁移都不会产生基座收益**——这不是系统出错，是 AI Scientist 用证据得出的结论。

## 诚实边界（课堂必须讲）

- 所有四个域的数据都是**最小教学样例**，训练出的模型没有科学价值，只用于观察 loss 下降、生成行为和表征结构。
- ESOL 只有 1128 条分子、UniProt 只取 500 条，预训练收益在这个规模**不会显现**；教材如实标注"演示预训练机制，不演示科学收益"。
- DNA 领域跳过了 chr21 开头的 N 区（未测序端粒区），只取 35 万真实碱基作为流式教学样例。
- 想看真实蛋白预训练收益，调用 ESM-2 8M 权重（A2 阶段），不要在这个仓库里追求。

## 课堂讲稿

逐级操作讲稿见 [docs/teaching-guide.md](docs/teaching-guide.md)：每级的讲解要点、现场命令、预期输出、停止边界和学生动手点。

## 一键运行全部领域

```bash
python scripts/run_all.py   # 四个域依次执行 prepare -> train(100 iter) -> sample
```

## 许可与致谢

- 本仓库 MIT 许可。核心架构改编自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)（MIT）。
- 思路参考：[hrzn/prot-gpt](https://github.com/hrzn/prot-gpt)（无 license，仅参考变长序列处理思想）、[diego-taquiri/nanoGPT-DNA](https://github.com/diego-taquiri/nanoGPT-DNA)（无 license，仅参考基因组数据加载思想）。
- 数据：tiny Shakespeare（nanoGPT 自带）、UniProt reviewed、人类 chr21（UCSC）、DeepChem ESOL。
