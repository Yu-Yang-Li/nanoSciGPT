# nanoSciGPT

“科学模型专题实训（二）：如何构建领域基座模型与 AI Scientist 系统”的配套代码、数据和三个教学 Skill。

从科学问题出发，先做分类或回归，再用小模型学习预训练与微调，最后讨论怎样让研究 Agent 继续实验。可以选课程样例，也可以带自己的数据；不用先会写模型代码。

## 现在可以用到哪一步？

截至 **2026-09-05**，基础课堂实验已有真实运行记录，**不代表所有研究流程都已跑通**。

| 课堂内容 | 当前情况 | 实测材料 |
|---|---|---|
| LAMOST 光谱估计有效温度 | 已运行数据检查、RandomForest 训练和评价 | [实际对话](docs/acceptance/cli-after-repair-2026-09-05/baseline/dialogue.md) |
| 文本及九类科学数据的预训练、任务微调 | 十类课程样例均有运行记录；微调实际更新模型参数 | [训练记录](docs/acceptance/training-ten-domains-2026-09-05/)、[微调记录](data/precomputed_results/finetuning-2026-09-05.md) |
| 自己的蛋白质 CSV | 序列＋数值标签可做回归；无标签时只预训练，已有实测 | [自有数据记录](docs/acceptance/student-protein-2026-09-05/) |
| Agent 分析已有结果、讨论下一轮实验 | 有 CLI 对话记录；建议仍需复核，不保证每次回答相同 | [两轮问答](docs/acceptance/provider-boundary-2026-09-05/glm-cli/dialogue.md) |
| 原版 autoresearch、The AI Scientist v1/v2 自动研究 | **试验中，完整流程未验收**；源码准备和部分基线/API已测 | [当前状态](docs/acceptance/training-and-native-status-2026-09-05.md) |

GLM-5.3 的普通 CLI 问答已有成功记录，但本机接入下 **v1 长改码仍未完成**：流式响应最终因长度限制结束，没有代码补丁，见[实测摘要](docs/acceptance/native-stream-2026-09-05/README.md)。课堂可以先做基线、预训练、微调和实验讨论，不把原版 AI Scientist 自动完成论文作为现场必过环节。

## 1. 下载与安装

需要 Git、Python 和可用的编程 Agent。本机实测 Python 3.12；包声明支持 Python 3.10 及以上，但未逐版本验收。**普通课堂实验使用单线程 CPU 小配置，不需要 GPU 或大模型权重。**首次安装依赖需要联网，课程样例数据随仓库提供。

Windows PowerShell：

```powershell
git clone https://github.com/Yu-Yang-Li/nanoSciGPT.git
cd nanoSciGPT
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m nanoscigpt.classroom --list
```

这里直接调用虚拟环境解释器，不必调整 PowerShell 执行策略。Linux 使用 `python3 -m venv .venv`，后续解释器路径换为 `.venv/bin/python`。已有合适的 torch 时无需重复安装；不要覆盖自己研究环境中的依赖。

`--list` 检查课程数据，不启动训练。安装失败时把报错交给 Agent，不用备用分数冒充本次运行。

## 2. 在 Codex CLI 中怎样开始

先完成自己电脑上的 Codex CLI 安装及登录/模型服务配置，再从 **nanoSciGPT 仓库根目录**启动 `codex`。仓库不提供密钥，也不会切换默认模型；不要照抄教师机器的本地代理地址。

下面三段提问任选一段发送即可。明确让 Agent 读取**仓库内**的 Skill，避免读到本机旧同名版本。`pip install -e .` 安装课程代码，不会把这三个 Skill 自动安装到 Codex 全局目录；直接读取文件无需覆盖已有 Skill。

### 第一段：有标签数据，先做基线

```text
请读取 skills/research-baseline-builder/SKILL.md，按它带我做实验。
我想用仓库的 LAMOST 光谱示例预测恒星有效温度。
请使用这个仓库 .venv 中的 Python，实际运行后解释结果。
```

换成自己的数据可以这样说：

```text
请读取 skills/research-baseline-builder/SKILL.md。
我想用[自己的数据地址][实现分类/参数估计或者具体下游任务]。
请先看看数据，缺少什么再问我。
```

方括号替换为你的数据地址和任务。Agent 应沿用已给的信息，只补问当前缺少的内容。课程 LAMOST 子集有 2000 条光谱、128 个流量特征，目标只有 `teff`，不是所有恒星参数。

**输出**：`out/baseline/` 内新实验目录的 `metrics.json`、`baseline_summary.json`、`train_log.txt`。具体位置以命令返回为准。

### 第二段：科学数据预训练，再微调

```text
请读取 skills/nanoscigpt-scientific-language/SKILL.md。
我先选 protein 课程样例，带我实际完成小规模预训练和任务微调。
使用仓库 .venv 中的 Python，并解释两次训练分别学什么。
```

没有选好方向，直接问“你可以带我做哪些数据？”即可。不必都选蛋白质，也不必运行所有领域。

| 名称 | 数据形式 | 课程数据身份 |
|---|---|---|
| `text` | 文本 | 公开文本子集；问答微调用少量人工教学题 |
| `protein` | 氨基酸序列 | 公开序列子集；课程标签不等于真实功能验证 |
| `dna` | 碱基序列 | 公开基因组片段 |
| `smiles` | 分子字符串 | 公开 ESOL 数据 |
| `weather` | 天气网格 | 生成的教学样例 |
| `crystal` | 晶体图 | 生成的教学样例 |
| `structure3d` | 三维结构 | 生成的教学样例 |
| `image` | 图像 | 生成的教学样例 |
| `spectrum` | 光谱 | 生成的教学样例，与第一段真实来源的 LAMOST 子集不同 |
| `field` | 连续物理场 | 生成的教学样例 |

来源见 [data/manifest.json](data/manifest.json)。文本、蛋白质、DNA、SMILES 共享因果 GPT 核心；其余六类保留各自的网格、图、几何或连续场结构，不是只换词表。

**输出**：`out/classroom/<domain>/run_report.json` 和 `model/ckpt.pt`；微调另存到 `finetune/`，其中 `downstream_result.json` 记录实际评价和任务 checkpoint。效果变差也保留，不能保证预训练必然带来提升。

想继续时可以说：“接着刚才微调后的模型再训练一下，数据和评价保持不变，结果另存。”

### 第三段：围绕结果，安排下一轮实验

```text
请读取 skills/ai-scientist-research-loop/SKILL.md。
接着刚才训练的模型，先看已有结果，帮我安排下一轮比较。
这次先讨论，不运行新实验。
```

这一步讨论实际模型、评价方式和预算，不把方案说成已运行。尝试原版研究前可以问：

```text
我想尝试 The AI Scientist v1。先检查这个模型能否接入原版，
告诉我需要哪些环境、模型 API 和运行预算，暂时不要启动研究。
```

完整命令和限制见[原项目接入说明](skills/ai-scientist-research-loop/references/native-projects.md)。`prepare` 只下载固定源码和设置教学配置，不自动安装依赖、调用 API 或完成研究。原版 autoresearch 仍需相应 CUDA 环境；v1 小基线可在 CPU 运行，完整研究另需编程、检索与写作依赖。

实验在本地执行，也可能把代码、日志和数据片段发给模型服务。私有材料先确认允许发送的范围。**不使用历史 `autoresearch/` 规则演示替代原项目后声称已复现 AI Scientist。**

## 3. 不用 Agent，也可以直接运行

从仓库根目录执行。以下使用 Windows 虚拟环境解释器，首次按顺序运行；重做时换输出目录，保留以前的结果。

```powershell
# LAMOST 回归
.\.venv\Scripts\python.exe -m nanoscigpt.baseline --case lamost --out_root out/baseline

# 文本预训练，然后用八组教学问答做微调
.\.venv\Scripts\python.exe -m nanoscigpt.classroom --domain text --profile classroom --out_root out/classroom
.\.venv\Scripts\python.exe -m nanoscigpt.tasks.text_sft --ckpt out/classroom/text/model/ckpt.pt --steps 200 --out_dir out/classroom/text/sft

# 蛋白质预训练与任务微调
.\.venv\Scripts\python.exe -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/classroom
.\.venv\Scripts\python.exe -m nanoscigpt.tasks.downstream_demo --domain protein --ckpt out/classroom/protein/model/ckpt.pt --adaptation finetune --epochs 2 --max_samples 32 --out_dir out/classroom/protein/finetune
```

最后两条命令中的 `protein` 可一并替换为其他领域。`classroom` 会先运行冻结表示的下游探针；后一条显式 `--adaptation finetune` 才继续更新预训练参数。文本问答用 `text_sft`，不能把文本分类误称为聊天微调。小模型可能记住原题，但答不好换个问法的题。

自己的表格用 `nanoscigpt.baseline --csv <绝对路径> --target <目标列> --task regression`；自己的蛋白质见 [CSV 接入说明](skills/nanoscigpt-scientific-language/references/student-protein.md)。其余自有格式需要逐项适配，十类样例能运行不代表任意数据都能直接训练。

## 4. 真实输入、输出与课程资料

上面的提问是**使用示例**，不是伪造的逐字实测记录。原始输入与实际回复另存：

- [LAMOST 实测对话](docs/acceptance/cli-after-repair-2026-09-05/baseline/dialogue.md)
- [蛋白质预训练、微调实测对话](docs/acceptance/cli-after-repair-2026-09-05/protein/dialogue.md)
- [GLM 研究讨论实测对话](docs/acceptance/provider-boundary-2026-09-05/glm-cli/dialogue.md)
- [当前运行范围与未完成项](docs/acceptance/training-and-native-status-2026-09-05.md)
- [课程大纲](docs/course-outline.md) · [讲师导航](docs/instructor/README.md) · [课后证据包](docs/evidence-pack-template.md)

历史稿件和调研材料保留在 `docs/`，旧规则演示保留在 `autoresearch/`，均不替代本页当前用法。课堂代码可执行，不代表三段连续 CLI 带练、所有电脑和所有模型服务均已验收。

## 仓库结构

```text
skills/                         三个教学入口及配套脚本
  research-baseline-builder/     科学问题与监督学习基线
  nanoscigpt-scientific-language/ 预训练、微调与自有数据
  ai-scientist-research-loop/    原版研究流程接入
nanoscigpt/
  core/                         序列 GPT 和训练代码
  domains/                      四类序列的数据准备
  scientific/                   六类结构化数据与模型
  tasks/                        下游探针、任务微调、文本回答微调
data/                           离线数据、来源清单与备用结果
docs/                           使用说明、实测记录与课程资料
tests/                          代码回归测试
out/                            本机结果，不上传 Git
```

## 许可与致谢

代码采用 [MIT 许可](LICENSE)。序列核心改编自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)。科学序列处理思路参考 prot-gpt、nanoGPT-DNA 等项目，没有将缺少许可的代码视为可任意复制。

原版研究项目下载到本机 `out/upstream/`，各自许可仍适用。数据来源与条件见 [data/manifest.json](data/manifest.json)；本仓库许可不替代上游代码和数据许可。
