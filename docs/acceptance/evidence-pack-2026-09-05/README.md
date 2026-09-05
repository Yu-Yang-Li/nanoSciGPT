# 课后运行记录导出检查

2026-09-05。本轮修正导出器，不重跑训练、不改PPT、不重写原CLI对话，也未更改三个Skill的对话指导。

## 发现和修正

原导出器只要发现`downstream_result.json`就写“已评测”。此前学生无标签路径已经正确跳过监督任务，但在导出作业时又被错误升级。本轮按运行状态与有效数值判断：无标签跳过仍为“已运行”；流程失败为“未完成”；缺失或非有限数值不成为有效监督分数。

另增加最新任务结果与原始附件入口。继续微调后可以明确使用新结果，而不是初始报告所指向的旧任务头。原版autoresearch/v1/v2日志不必转换成旧离线脚本的状态文件；附件仅保存路径、大小和SHA256，不读取内容来自动判定研究完成。提交时还需附允许共享的原文件，单独的本机路径索引并不构成可移植的完整作业。

输出文件不覆盖旧作业。没有下一步决定时写“未记录”，不把程序默认建议冒充学生已经作出的决定。

## 两份真实导出

- [无标签蛋白质](no-labels.md)：读取此前真实运行的报告与跳过记录，标为已运行、无监督分数。
- [微调及原版实验附件](native-protein.md)：读取此前实际往返实验的最后微调结果，MAE 3.398 → 3.6469，明确记录误差增大；原版实验记录和命令日志作为附件保留。

这两份是现有真实实验的导出结果，不是新训练，也不是CLI学生对话。蛋白质数据是八条手工教学数据，不是实测活性数据。原版研究Agent没有因此获得“已运行”或“已完成”的认定。

在仓库根目录实际执行的命令：

```powershell
python -m nanoscigpt.evidence_pack --run-report out/student-check/unlabeled-run/protein/run_report.json --output docs/acceptance/evidence-pack-2026-09-05/no-labels.md
python -m nanoscigpt.evidence_pack --run-report out/student-check/labeled-run/protein/run_report.json --downstream-result out/native-student-task-roundtrip-v2/course-return/downstream_result.json --attachment docs/acceptance/native-supervised-roundtrip-2026-09-05/protein/native/record.json --attachment docs/acceptance/native-supervised-roundtrip-2026-09-05/protein/commands.json --attachment docs/acceptance/native-supervised-roundtrip-2026-09-05/protein/native/stdout.txt --output docs/acceptance/evidence-pack-2026-09-05/native-protein.md
```

两条退出码均为0，导出正文已逐项读取。重跑时换新输出路径；本机`out/`不随Git分发，应先按前轮数据与模型运行说明产生自己的报告。

## 回归范围

`python -m pytest tests/test_evidence_pack.py -q`：16项通过，见[XML](tests.xml)。覆盖无标签、失败运行、NaN/Inf/缺失分数、最新任务选择、微调退步、附件只引用不修改、不因附件升级评价等级、缺失文件、不同领域误配、非对象JSON和禁止覆盖旧作业。

本轮没有重跑全部训练测试。上一轮152项全量测试是修改导出器之前的证据，不能直接算成本轮全量通过；这16项包含原有5项导出测试，也不与152简单相加。

项目仍未满足正式课堂验收：原版自主研究、真实连续CLI教学和最终干净克隆复现尚未完成。本轮修复只推进证据包这一项。
