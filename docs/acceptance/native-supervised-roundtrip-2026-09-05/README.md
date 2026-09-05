# 微调任务与原版 v1 实验接口的往返检查

2026-09-05，Windows、Python 3.12、单线程 CPU。这里记录真实命令运行，不是学生与 CLI 的对话，也没有调用研究 Agent 的模型 API。

## 这次解决的问题

以前微调文件缺少领域标识，而 v1 接入只覆盖序列预测。直接把学生的回归模型交过去，不能算继续原任务。现在微调文件保存领域、任务头、标签标准化、数据文件摘要和抽样数量；分类/回归任务导出相应实验模板，不退回字符预测。

GPT 类来自固定版本 SakanaAI/AI-Scientist 的原模板，源码保持不变。监督训练和绘图是课程适配，原版想法、实验和写作入口不改。不能把课程监督训练代码说成原版训练循环，也不能把这次实验执行说成原版 Agent 研究已经完成。

## 实际结果

| 数据与任务 | 课程微调后 | v1 模板刚加载时 | v1 模板训练后 | 回到课程代码刚加载时 | 再微调后 |
|---|---:|---:|---:|---:|---:|
| 蛋白质 CSV，activity 回归，MAE | 3.7486 | 3.748553 | 3.398048 | 3.3980 | 3.6469 |
| SMILES，ESOL 溶解度回归，MAE | 1.9677 | 1.967671 | 1.671262 | 1.6713 | 1.9417 |

导出、返回时的差异只来自课程报告保留四位小数。两次返回后的再次微调都变差，原样保存；不保证多训练一定更好，不据这次小样本运行判断预训练的科学收益。

- 蛋白质用[八条手工编写的教学 CSV](../student-protein-2026-09-05/input.csv)，6 条训练、2 条验证，不是测量的生物学数据。源文件和任务名贯穿三个步骤。
- ESOL 用仓库列明来源的实测溶解度子集，本次 32 条训练、16 条验证；小规模检查不代表完整数据集 benchmark。
- 课程微调两轮，v1 教学模板训练四步，再回到课程微调两轮。优化器每次新建，未声称恢复优化器状态。
- 两组都实际执行了原版 v1 调用约定的 `experiment.py --out_dir run_0` 和 `plot.py`。原版 GPT 可训练参数严格加载，研究模板可放到独立目录运行。

## 可复核记录

- 蛋白质：[结果与输入模型摘要](protein/acceptance.json)、[完整命令](protein/commands.json)、[原实验 stdout](protein/native/stdout.txt)、[训练图](protein/task_training.png)。
- SMILES：[结果与输入模型摘要](smiles/acceptance.json)、[完整命令](smiles/commands.json)、[原实验 stdout](smiles/native/stdout.txt)、[训练图](smiles/task_training.png)。

每个命令另存 stdout/stderr。完整模型和中间数据留在本机 `out/`，不把模型二进制或学生真实数据加入 Git。所有记录明确标注 `agent_api_run: false`。

## 重跑方法

先在已安装课程依赖的环境中补充绘图依赖、准备固定版本源码：

```powershell
python -m pip install -e ".[native-v1-template]"
python -m nanoscigpt.upstream prepare v1 --device cpu
```

已有课程预训练模型时，可运行下面的教师验收脚本。自有蛋白质数据先按[自有数据入口](../../../skills/nanoscigpt-scientific-language/references/student-protein.md)准备；`--data-root`必须沿用同一目录。模板名和输出目录必须是新的，既有实验不覆盖。

```powershell
python scripts/verify_native_task_lesson.py --domain protein --ckpt <原预训练模型路径> --data-root <该模型的数据目录> --name my_protein_task --output out/my-task-check
```

脚本串起微调、导出、实验、绘图和返回微调，检查相同指标以及源模型未变。它不是新的学生 Skill，也不代替原版 `launch_scientist.py`。

## 同时修正的继续训练行为

四类序列与六类结构化样例再次任务微调时都恢复已保存的任务头。沿用任务数据、抽样数量和标准化数值；变化时拒绝静默续接。六类原始预训练文件没有保存任务头的标签标准化，因此第一次任务微调明确建立新任务头；新微调文件保存完整状态，之后才能连续复用。

## 本轮检查

全量 `pytest -q`：152 项通过，377.22 秒，见[原始测试报告](regression.xml)。覆盖十类任务再次微调、分类/回归导出与返回、独立目录执行和绘图、数据改变时拒绝续接。执行时已有固定版本的原项目源码，因此本轮没有跳过原版 GPT 集成测试。全量测试也包含旧离线演示的检查，不能把总数解读成 152 次原版 Agent 研究。

三个 Skill 的 `quick_validate.py` 检查通过（Windows 使用 `PYTHONUTF8=1`），五份本轮入口/引用文档本地链接无缺失，`git diff --check`通过。修改尚未提交或推送。

原版 autoresearch 自主改码、v1 API 研究、v2 研究树及新版连续 CLI 教学尚未验收。本次记录只补齐同一模型和同一任务的实验衔接，不将整个项目标为正式课堂可用。
