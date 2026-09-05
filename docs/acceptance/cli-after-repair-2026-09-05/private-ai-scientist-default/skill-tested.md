---
name: ai-scientist-research-loop
description: Use when a student wants to iterate a runnable model with the original karpathy autoresearch, conduct computational research with SakanaAI The AI Scientist v1, or explore research routes with v2 in nanoSciGPT.
metadata:
  short-description: 用原版 autoresearch 和 AI Scientist 带学生做计算研究
---

# AI Scientist研究实践

以三个原项目为主：karpathy/autoresearch、SakanaAI/AI-Scientist、SakanaAI/AI-Scientist-v2。教学适配缩小数据、模型和实验次数，不用预先写好的改动、假设或稿件代替Agent研究。

## 接着学生手里的实验往下做

先沿用学生提供的模型、数据、运行命令或报告，不要求文件一定叫`run_report.json`。课程模型通常在`out/classroom/<domain>/`，但它不一定就是学生正在用的模型。

学生只给了一个分数时，先接住他的问题，再问这个分数怎样得到。例如：“可以先讨论，不用上传数据或代码。这个MAE平时是用什么命令或步骤算出来的？”等他回答，再决定需要了解什么；不把模型结构、数据规模、评价方法和计算资源一次全问完。不了解他的模型时，不先安排转换成课程GPT，也不拿本机环境或示例成绩代表他的条件。

不能共享数据或代码时，只讨论学生允许描述的内容。原项目在本地运行，不代表代码、日志或数据片段不会随模型API请求发出去；实际接入前要确认发送范围。若这些内容不能发送，也没有合适的本地模型，就由学生自行运行实验，在允许的范围内反馈结果，不启动外部研究Agent。

学生说“继续优化”时，根据现有结果提出值得检验的改动，说明依据和大致开销。已经授权在约定预算内运行，就直接推进，不要求每次再念一遍确认口令。若缺少研究目标、可执行实验或必要资源，再问真正影响下一步的问题。

安排比较时，从学生已经说明的设置出发。尚不清楚训练目标，就先问它是什么，不能因为回归任务常用MSE而假定学生也在用；提出备选改动时说清它适用于什么情况。相同设置的重复运行用于了解波动，两次训练之差不能当作稳定的波动范围或显著性门槛。预算只够少量实验时，如实称为初步比较，不强求改善，也不把不确定性写成确定结论。

学生想自己操作时给简短命令；希望你代跑时实际执行，再解释结果。不要为了固定对话节奏停在预检，也不要一口气替学生运行所有领域。

## 按这次想做的研究选择入口

准备实际执行某个原项目时，再读[原项目接入说明](references/native-projects.md)，并用可用的Python运行`python -m nanoscigpt.upstream doctor`。仅讨论方案时，先聊实验本身，不检查本机环境或替学生创建实验文件。源码准备与安装、调用API、训练是不同操作。

### 围绕模型继续优化：autoresearch

读取原版`program.md`、`train.py`、`prepare.py`以及教学设置`TEACHING.md`。由当前编程Agent提出改动、修改训练代码、运行、比较并决定保留或撤回；不是只增加固定训练步数。

```powershell
python -m nanoscigpt.upstream prepare autoresearch --device cuda
```

教学设置从一次基线和最多两轮自主改动开始，每次训练采用相同的时间额度；运行中保持数据准备和评价不变。保存原版`results.tsv`、实际日志和代码改动。缩小设置后重新测基线，不能与上游默认配置的分数混比。预算不足或执行失败时，保留结果并解释原因。

原版使用自己的模型与文本数据，不会自动加载学生的nanoSciGPT checkpoint。若学生要求继续自己的模型，先适配数据、模型和评价接口，并复测基线；不能悄悄换成原版默认模型，却称作“继续优化刚才的模型”。

### 想把研究想法、实验和写作连起来：The AI Scientist v1

```powershell
python -m nanoscigpt.upstream prepare v1 --device cpu
```

使用原版自带的nanoGPT实验模板及其研究流程。课程副本沿用仓库的文本数据，先实际运行小模型基线；依赖和模型API准备好后，再启动原版`launch_scientist.py`，让它生成想法、检索、改代码、做实验、绘图、写作和评阅。v1不只是整理上一轮现成结果，也不需要先有本仓库旧脚本的输出。

默认文本模板从头训练。如果学生已经有本仓库`text/protein/dna/smiles`的因果GPT模型，用下面的适配入口继续同一套权重，再在生成的原版模板目录中运行基线和研究：

```powershell
python -m nanoscigpt.native_v1 --ckpt out/classroom/protein/model/ckpt.pt --name course_protein
```

沿用学生选择替换路径和模板名。先看学生刚才做的是预训练还是下游任务：只有预训练权重时，继续原版的序列预测实验；已有`task_checkpoint`时，把这份微调文件交给适配入口，它会连同任务头、真实标签和评价方式一起导出，不改回“预测下一个字符”。自己的数据同时传入对应的`--data_root`。不要只从默认目录挑一个旧checkpoint。

下游任务使用课程提供的分类/回归实验模板，GPT类和想法、实验、写作流程仍来自原版v1；不能称作未经修改的原版训练模板。优化器重新初始化，各次实验从同一初始模型出发。先比较导出前后的同一指标，确认任务没变，再启动研究Agent。数据或抽样数量变化时应说明并重建基线，不绕过来源检查。掩码编码器、天气、图和连续场不能通过这个序列GPT转换器；先准备对应的原版实验模板。文本不是强制研究方向。

### 想让系统探索和调整实验路线：The AI Scientist v2

```powershell
python -m nanoscigpt.upstream prepare v2 --device cuda
```

使用原版无模板想法生成、实验管理和渐进式研究树搜索。课程设置减少并行数和每阶段实验次数，但保留原有四阶段及其搜索逻辑；不预设两条候选路线，也不要求先拿到v1的`workflow_state.json`。沿用学生的问题和数据，写清输入位置及可用计算资源，再运行原版入口。

## 讲清实际做到哪里

按真实进展说明：源码已准备、实验已运行、Agent研究已运行，或稿件已生成；不要用一个“完成”覆盖所有环节。检查原项目留下的实验日志、指标、代码差异、搜索记录和稿件引用。没有运行证据就不报告实验分数，生成稿件也不等于结论经过验证。

Codex登录不等于原项目拥有模型API。缺少API时，可以完成源码准备和不依赖API的基线实验；不能改用规则脚本后声称原项目已经跑通。v2的视觉反馈、v1的写作与审稿角色要分别核对模型支持，不能只改一个模型名称就宣布兼容。

## 旧的离线演示

只有学生明确想看无API离线演示时才用以下旧实现，并标明它们不是三个原项目的复现：

- [固定预算比较](references/autoresearch-model-iteration.md)：`autoresearch.experiment`。
- [已有结果整理](references/ai-scientist-v1-workflow.md)：`autoresearch.v1`。
- [预设分支比较](references/ai-scientist-v2-tree-search.md)：`autoresearch.v2`。
