# 沙箱修复后的真实 CLI 教学记录

日期：2026-09-05。下面分别记录实际运行和对话检查，不合并成“整堂课已通过”。所有输入和回复均为实际CLI原文，没有代写学生实验成绩。

| 场景 | 实际结果 | 证据 |
|---|---|---|
| 默认模型，LAMOST基线，两轮 | 实际训练随机森林；2000条样本中400条用于本次测试，温度RMSE为83.4367 K。第二轮给出复跑方式 | [对话](baseline/dialogue.md)、[指标](baseline/metrics.json) |
| 默认模型，蛋白质预训练/微调及原项目接入检查，两轮 | 实际运行小模型；微调验证准确率从56.25%降至43.75%，保留退步结果。第二轮只核查接入，不称为原版研究已运行 | [对话](protein/dialogue.md)、[微调结果](protein/finetune-result.json) |
| 默认模型，不能共享的材料配方回归，一轮 | 只读Skill并讨论方案，未运行、未报告实验成绩；回答仍偏长 | [对话](private-baseline-default/dialogue.md) |
| 默认模型，不能共享的蛋白质回归，两轮 | 首轮询问分数怎样获得；次轮给出初步比较安排，询问现有损失函数。不检查环境、不创建文件、不假定训练目标 | [对话原文](private-ai-scientist-default/dialogue.md)、[当时的Skill](private-ai-scientist-default/skill-tested.md) |

## 会话条件

前两组是真实新会话，但CLI曾访问这台主机的持久记忆，因此不算无历史上下文测试。为避免把无关项目记忆发布到Git，本目录只收录它们的原始学生输入、最终回复、命令和会话记录；完整事件日志仍在本地 `out/cli-after-repair-baseline-dialogue/` 与 `out/cli-after-repair-protein-dialogue/`。省略原始事件不意味着重新编写回复。

后续记录器在首轮和resume均传入 `features.memories=false`，继续使用 `workspace-write` 和 `approval_policy=never`。主机已有Skill仍可能被发现，所以这也不等于一台刚安装Codex的学生电脑。默认模型不由记录脚本指定；GLM场景显式选择`scnet/GLM-5.3`。

## 发现的问题及处理

GLM最初面对“私有蛋白质模型、只有MAE 0.7”时，先检查本机环境并同时问多组问题；第二轮学生只想讨论，它还创建了三个实验文件。这一组不通过带练要求，原文保存在本地 `out/cli-cold-glm-private-dialogue/`。

这三个文件已原样移到该记录目录的 `generated-research/`，没有删除，也不作为课程正式代码。来源可在第二轮事件日志核查。

| 文件 | 移动时SHA256 |
|---|---|
| round1-comparison-plan.md | F8B9ABD6E60293F0CD1EAC5C2DC007829D5E7ECE25AC0ED626F2784CC904012F |
| results.tsv | 93E72E4B30DADE271ACA0BE7E708351E5031FDB44C93D40E20CD6686A15855AC |
| summarize_errors.py | ABC310ABF135E65FE03C9D48E607AF15B4F1A0396D3DD572B7DEA8A3234CD009 |

第一轮措辞复测中，GLM不再检查环境、创建文件，但仍未确认训练设置就建议换损失函数，并把两次训练之差当作波动门槛。记录在 `out/cli-private-guidance-retest-20260905/`；不能把不越权当作科学判断已合格。Skill随后补上“先问现有训练目标、备选改动说明条件、少量重复只作初步比较”的引导，另开会话复测。

第一轮GLM失败回复另提供[可读原文](private-ai-scientist-glm-attempt1/dialogue.md)，不改写其中不合格的建议。第二次GLM复测位于`out/cli-private-guidance-retest2-20260905/`，第二轮尝试提问工具时发生两次参数解析错误（`invalid type: map, expected a string`）；问题尚未定位到具体兼容层，不把它说成学生回答错误，也不算稳定带练通过。相同输入在上述默认模型会话中正常以文字提问，不为凑通过率而混合统计两个模型。

第二次GLM复测最终在第二轮600秒超时退出，没有最终回复；[失败记录](private-ai-scientist-glm-attempt2/README.md)保留实际输入、第一轮回复、第二轮部分事件和错误。没有继续重试同一条请求。

## 记录器回归

单轮CLI时限仍为600秒。新增超时处理先通过测试复现“抛异常后没有当轮汇总”的问题，再改为保留部分事件、写出`cli_timeout`、返回失败并停止续聊；相关[32项回归](../native-failure-repair-2026-09-05/final-regression.xml)通过。测试中的超时事件为进程边界替身，不是伪造真实学生对话。`subprocess.run`超时会终止直接CLI进程，但这里没有提供或宣称任意研究子进程树都已停止的保证。

另做了15次有限输出预算的措辞微测（无Skill、原措辞、初稿各5次），其中14次被长度限制截断，唯一完整回复仍追问过多；因此不用于证明措辞稳定性。它只提供了失败线索，不是CLI运行验收，所有记录留在 `out/private-skill-wording-20260905/`。

## 尚未覆盖

这批记录不证明三个Skill的全部自有数据场景已通过，不证明原版autoresearch、v1实验与写作、v2研究树已经完整运行，也不是当前版本的GitHub干净克隆验收。论文模型的科学性能不能由教学小样本指标推断。
