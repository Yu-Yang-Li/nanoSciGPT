# 自有蛋白质CSV接入检查

2026-09-05。本轮是直接命令执行，不是Codex CLI对话验收。`input.csv`是专为接口检查手工构造的八行数据，`activity`也是构造数值，不是生物实验测量结果。不能把本次分数解释为蛋白功能预测能力。

## 实际完成

1. 从CSV列读取序列和数值目标，生成新的独立数据目录，源CSV不变。
2. `classroom --profile smoke`在这份数据上运行预训练、生成和回归。下游报告明确标签来自CSV的`activity`列，不再使用课程组成标签。
3. 接着同一模型执行一轮真实微调。MAE 3.7052 → 3.7293，结果变差；预训练参数确实改变，原模型保留。
4. 对同一文件不指定目标列再次运行，完成预训练与生成，下游记录为`skipped_no_labels`，没有监督分数。文件中存在数字不代表助教可以擅自把它指定为研究目标。
5. 自动化测试检查标签对齐、无标签处理、重复序列跨集合拦截、禁止覆盖旧目录、八行数据自动留出。五项通过。

## 可复跑命令

在仓库根目录，使用新的输出目录：

```powershell
python -m nanoscigpt.student_protein --csv docs/acceptance/student-protein-2026-09-05/input.csv --sequence-column sequence --target-column activity --split-column split --data_root out/student-check/labeled
python -m nanoscigpt.classroom --domain protein --profile smoke --data_root out/student-check/labeled --out_root out/student-check/labeled-run
python -m nanoscigpt.tasks.downstream_demo --domain protein --data_root out/student-check/labeled --ckpt out/student-check/labeled-run/protein/model/ckpt.pt --adaptation finetune --epochs 1 --max_samples 8 --out_dir out/student-check/labeled-finetune

python -m nanoscigpt.student_protein --csv docs/acceptance/student-protein-2026-09-05/input.csv --sequence-column sequence --split-column split --data_root out/student-check/unlabeled
python -m nanoscigpt.classroom --domain protein --profile smoke --data_root out/student-check/unlabeled --out_root out/student-check/unlabeled-run
```

## 保存的证据

- [有标签运行与实际命令](labeled-run.json)、[控制台输出](labeled.stdout.txt)、[回归结果](labeled-result.json)。
- [微调结果](finetune-result.json)：包含参数变化、前后指标、源数据哈希及标签列。
- [无标签运行](unlabeled-run.json)、[控制台输出](unlabeled.stdout.txt)、[明确跳过监督任务](unlabeled-result.json)。
- [五项专项测试](tests.xml)。
- [全套回归149项通过，351.13秒](regression.xml)。回归启动后又增加了小数据自动留出的测试，因此五项专项单独复跑通过；两组有重叠，不相加。这些是代码和命令检查，不是新版Skill的真实CLI对话。

## 范围

当前新增入口支持标准蛋白序列CSV和可选的数值回归标签。分类标签、其余科学对象的自有数据适配，以及三个新版Skill的真实连续CLI对话仍需补齐。自动留出只防止完全相同序列跨集合，不等于完成同源性隔离。第一阶段课程稿由另一会话处理；本轮不改页码、文字稿或PPT。

下一项代码核查：微调checkpoint与原版研究入口的数据和元信息能否连续接续；之后继续完成其他自有数据入口。项目仍不标记为“可用于正式课堂”。
