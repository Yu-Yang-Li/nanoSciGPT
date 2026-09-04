# 三个学生入口的 Codex CLI 对话验收

日期：2026-09-04  
工作目录：仓库根目录  
运行方式：每个场景均为新的 `codex exec --ephemeral` 会话；忽略项目外规则，只读取指定的仓库 Skill。为了允许助教实际检查文件和运行课程命令，使用 `--sandbox danger-full-access`；所有生成结果限制在本仓库的 `out/cli-dialogue/`。

主测试使用本机 Codex CLI 默认模型 `gpt-5.6-sol`，将推理强度临时设为 `low`；兼容性复测显式使用 `scnet/GLM-5.3`。Skill本身没有绑定模型供应商。

## 验收结果

| 学生入口 | 场景 | 真实行为 | 判断 | 完整CLI记录 |
|---|---|---|---|---|
| Research Baseline | 课程LAMOST示例 | 直接运行2000条光谱的RandomForest回归；报告RMSE 83.44 K、R² 0.9981和5折结果；明确只适用于同一课程数据范围 | 通过 | [JSONL](cli-dialogues/baseline-course.jsonl) |
| Research Baseline | 学生自带CSV | 读取1128行分子表格并完成回归；发现输入中已有ESOL预测值，指出可能造成结果过于乐观，只追问该列能否使用 | 通过 | [JSONL](cli-dialogues/baseline-own.jsonl) |
| Research Baseline | 数据不能上传 | 保持在设计状态，不生成分数；只询问配方输入信息的名称 | 通过 | [JSONL](cli-dialogues/baseline-private.jsonl) |
| Scientific Language | 课程protein示例 | 只运行十类数据预检；确认protein训练450条、验证50条，没有提前开始训练 | 通过 | [JSONL](cli-dialogues/science-course.jsonl) |
| Scientific Language | 学生自带weather文件 | 实际读取NPZ形状，发现学生口述的“预测下一时刻完整网格”与文件中的标量标签不一致，只追问目标究竟是哪一种 | 通过 | [JSONL](cli-dialogues/science-own.jsonl) |
| Scientific Language | 无标签且不能上传 | 将4096点光谱解释为保持波长顺序的完整样本，提出连续谱段遮盖任务；不虚构运行，只问是否已统一波长网格并处理坏点 | 通过 | [JSONL](cli-dialogues/science-private.jsonl) |
| AI Scientist | 课程已有模型 | 读取真实`run_report.json`，只生成迭代设置；固定数据、划分、模型结构和评价，仅把训练步数30改为60，等待学生批准 | 通过 | [JSONL](cli-dialogues/ai-course.jsonl) |
| AI Scientist | 学生自带结果 | 发现只有`metrics.json`，明确它不是完整运行记录，不启动自动优化，只询问原始运行命令 | 通过 | [JSONL](cli-dialogues/ai-own.jsonl) |
| AI Scientist | 数据与模型不能上传 | 只把0.42记作学生口述的当前基准，停在单一改动的设计，不声称运行、比较或闭环 | 通过 | [JSONL](cli-dialogues/ai-private.jsonl) |

九个默认模型会话均返回 `turn.completed`，对应CLI进程退出码均为0。课程示例和学生文件场景保留了真实命令及输出；不能上传的场景没有生成实验结果。

## GLM-5.3兼容性复测

使用 `scnet/GLM-5.3` 重跑“学生选择protein”场景。模型读取同一Skill，实际执行 `python -m nanoscigpt.classroom --list`，确认十类数据及protein入口可用，并给出下一步。完整记录见 [glm53-science-course.jsonl](cli-dialogues/glm53-science-course.jsonl)。

CLI提示当前配置的`ultrafast`服务层不适用于GLM-5.3，因此自动省略该服务层；不影响本次请求完成。Skill无需为此增加供应商分支。

## 运行时观察

每个会话首先尝试连接本地 Responses websocket，并记录 `426 Upgrade Required`；随后自动退回可用传输，最终仍产生真实工具调用、助手回复和 `turn.completed`。因此该告警保留在原始JSONL中，但不再构成本轮对话验收阻塞。

CLI还提示已安装Skill的描述超过上下文预算并进行了缩短。三个被测Skill均通过明确仓库路径完整读取，原文没有截断；正式课堂仍建议关闭不使用的插件和Skill，以减少无关上下文。

## 本轮观察到的教学边界

- 信息够时直接检查或运行，不重复盘问；信息不够时只补一个会改变下一步的问题。
- `我理解为`、`我看到`、`已经运行`与实际证据一致。
- 学生文件与课程内置数据分开记录；课程结果没有移植到学生数据。
- 不能上传、缺少标签或缺少完整运行报告时停在设计，不生成分数或研究结论。
- AutoResearch只生成设置，尚未批准时不训练；没有跨过人工确认门。
