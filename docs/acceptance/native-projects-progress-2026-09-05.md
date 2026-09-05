# 原项目教学接入：2026-09-05

> 本文保留第一轮检查结果。后续已补齐十类任务微调、文本回答微调和原版v1课程权重接续，当前状态见[后续实测记录](training-and-native-status-2026-09-05.md)；原项目API全流程和新版连续CLI仍未验收。

## 本轮采用的做法

以karpathy/autoresearch、SakanaAI The AI Scientist v1/v2的源码为主。仅对模型规模、数据范围、实验次数、并行数和运行时间做教学调整。PPT、本轮以前的真实对话导出、共享工作区里的未提交稿件均未改动。

三个原项目已获取固定commit并成功应用教学设置，源码版本和启动说明见[原项目接入说明](../../skills/ai-scientist-research-loop/references/native-projects.md)。本仓库以前的规则迭代、稿件模板和预设分支继续保留，但已经从Skill默认路径中移出。

## 真实运行到了哪里

| 项目 | 本轮证据 | 尚未完成 |
|---|---|---|
| autoresearch | 固定源码及教学配置已准备，原版模型、优化器和评价代码保留 | CUDA运行环境、数据准备、自主改码循环未验收；尚不能自动承接nanoSciGPT权重 |
| AI Scientist v1 | 原nanoGPT模板的小规模CPU实验已运行，原生结果文件已生成 | 独立模型API、Aider及全部角色调用尚未接通，没有执行想法→改码→实验→写稿全流程 |
| AI Scientist v2 | 固定源码已准备；四阶段、实验管理和研究树代码未改，只调整配置 | 模型API、视觉反馈与完整搜索尚未验收 |
| 四类序列数据微调 | 文本、蛋白质、DNA、SMILES的参数更新及原checkpoint保护测试通过；另完成蛋白质真实预训练后微调 | 文本问答微调和其余六类结构化样例的全模型微调仍未补齐 |
| 测试数据复现 | 新脚本可重建旧对话中的ESOL、天气和单分数输入，重复执行不会覆盖学生文件 | 历史`ai-course`对话引用的旧训练输出不随仓库分发，重测应换为本次生成的报告路径 |
| 连续CLI带练 | 第一轮原文和CLI回复已保存；明确报告没有执行本地实验 | Windows沙箱ACL初始化失败；第二轮已停止，第三轮未运行，不能标为通过 |

### v1原版实验记录

运行命令为原模板的`python experiment.py --out_dir run_0`。教学副本使用仓库文本词表和数据：两层、64维、batch 8、64上下文、30轮设置、一个种子、CPU单线程。只改教学参数和数据入口，未用nanoSciGPT训练器替代原实验。

验证损失4.1794→3.4794，命令退出码0。本次捕获总耗时约5.9秒；这只是小模板实验，不是论文中的模型质量或完整研究耗时。原版结果中单种子的标准误差为0，不应解释成没有不确定性。

- [实际命令、源码hash和原生结果](native-v1-baseline-2026-09-05/record.json)
- [完整stdout](native-v1-baseline-2026-09-05/stdout.txt)
- [完整stderr](native-v1-baseline-2026-09-05/stderr.txt)：原版有PyTorch GradScaler弃用提示，实验仍正常结束。

### 蛋白质真实微调记录

```powershell
python -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/finetune-acceptance
python -m nanoscigpt.tasks.downstream_demo --domain protein --ckpt out/finetune-acceptance/protein/model/ckpt.pt --adaptation finetune --epochs 2 --max_samples 32 --out_dir out/finetune-acceptance/protein/finetune
```

预训练验证损失3.0951→2.9163。微调使用32条训练样本、16条验证样本，预训练参数变化L2范数约0.34334，另存`finetuned.pt`，不覆盖原模型。分类准确率0.5625→0.4375，本次没有改善。

这里“微调前”指同一个预训练编码器接随机初始化的任务头，不是与训练充分的冻结表示方案比较。标签来自序列组成，是教学标签，不能解释为蛋白质功能预测结果；这次检查的是预训练后确实继续更新了模型。

小体积结果已保存：[预训练日志](protein-finetune-2026-09-05/train_log.json)、[微调结果](protein-finetune-2026-09-05/downstream_result.json)。原模型和微调模型留在本地`out/finetune-acceptance/`，未放入Git。

### CLI原文及中止原因

- [第一轮输入与实际回复](native-route-dialogue-2026-09-05/dialogue.md)
- [本次状态](native-route-dialogue-2026-09-05/STATUS.md)
- [预定三轮输入](native-route-inputs-2026-09-05.json)

CLI发生`helper_unknown_error: apply deny-read ACLs`，尚未读到本地新版Skill，也没有执行训练。它转读了远程旧版材料，所以本次对话不能作为新版Skill行为的有效验收。没有调整沙箱权限、没有修改全局Codex配置，也没有将旧版本回复冒充成新版本结果。

## 下一步

本轮全套回归测试129项通过（325.40秒）；后续补入防覆盖检查后，相关13项测试再次通过。Skill frontmatter校验通过。测试数有重叠，不应相加；代码测试也不替代上述原项目API运行和CLI连续教学验收。

先解决独立模型服务选择与当前CLI沙箱运行条件，再实测原项目的一次完整研究和连续带练。为学生所选科学数据适配原项目时，明确哪些数据/模型接口发生变化，并建立新的可比较基线；不能只换项目名就称作接上了原模型。
