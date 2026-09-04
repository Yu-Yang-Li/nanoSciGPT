# 课程大纲：科学模型专题实训（二）

## 如何构建领域基座模型与 AI Scientist 系统

**对象**：全院非计算机专业研究生  
**时长**：90 分钟  
**事实截止日期**：2026-08-15  
**配套仓库**：本仓库（nanoSciGPT）  
**PPT 逐页大纲**：课堂执行版本由讲师单独维护；本文件是仓库内的课程大纲唯一事实源

---

## 课程定位（一句话）

> Agent 时代，人不需要手写 GNN，但必须知道**该让 Agent 建什么系统、什么时候信它、什么时候停**。本课用两条发展史回答这个问题：A 线讲科学数据如何变成模型可消化的语言、什么时候值得做基座；B 线讲工具和评价器如何串成会自我修正的闭环。

## 贯穿设计

每位学生带着自己的课题，沿两条独立时间线逐级升级：

- **A 线（能力复用半径）**：V0 专用模型 → 科学对象语言化 → 换预训练目标 → 表征迁移 → 多任务统一 → 路线决策。实操载体是本仓库 nanoSciGPT。
- **B 线（科研过程长度）**：V0 可执行动作 → 假设生成 → 实验闭环 → 论文与审稿 → 人的位置。实操载体是本仓库 autoresearch 模块。
- **汇流**：A 线产出的模型成为 B 线闭环里被调用的工具；B 线的 evaluator 成为 A 线模型升级与否的裁决者。

课程不承诺 90 分钟训练真实科学基座。以下两种结论同样合格：没有多任务复用条件时"继续用专用模型"；没有真实反馈时"停在工具工作流，暂不称为闭环"。

---

## 90 分钟总表

| 段落 | 时间 | 内容 | 产物 |
|---|---:|---|---|
| 开场 | 0–5 min | Agent 能写代码，但你要决定建什么 | 课题一句话 |
| A 线 | 5–45 min | 领域基座模型发展史 + nanoSciGPT 实操 | 模型路线判断 |
| 过渡 | 45–48 min | 模型成为工具，科研是循环 | —— |
| B 线 | 48–83 min | AI Scientist 发展史 + autoresearch 实操 | 闭环层级判断 |
| 收束 | 83–90 min | 作业证据包、评分标准、结语 | 最小行动 |

## A 线：领域基座模型（40 分钟）

每一阶段四段式：**旧瓶颈 → 技术跃迁 → 论文锚点 → 实操增量**。

| 阶段 | 旧瓶颈 | 技术跃迁 | 论文锚点 | 实操（nanoSciGPT） |
|---|---|---|---|---|
| 前史 | 每个任务从零训练太贵 | 任务可学习，但专用、不复用 | AlphaFold2、Pangu/GraphCast | 无，只铺垫 |
| A1 科学对象语言化 | 我的对象不是文本 | 序列化 + 自监督 | UniRep/TAPE/ProtTrans/ESM-1b | text 跑通 → 换 protein/DNA/SMILES，架构不动 |
| A2a 换预训练目标 | 因果方向未必适合科学序列 | MLM vs CLM | ESM 双向 vs GPT 单向 | objective_contrast 对比 |
| A2b 表征迁移 | 预训练了但下游能用吗 | 冻结探针/微调 | ESM-2、ClimaX、scGPT | transfer_probe：负结果教学 |
| A3a 多任务统一 | 一个模型伺候多任务 | 共享编码器+任务头 | ESM3、Aurora、Evo2 | multihead |
| A3b 路线决策 | 到底要不要基座 | 证据决定路线 | —— | route_decision 五问 |

**A 线一句话**：科学对象可预训练化 → 表征可迁移化 → 基座能力按真实需求统一化。

## B 线：AI Scientist（35 分钟）

同样四段式结构。

| 阶段 | 旧瓶颈 | 技术跃迁 | 论文锚点 | 实操（autoresearch） |
|---|---|---|---|---|
| 前史 | 窄目标自动化早已存在 | Adam/Eve、mobile 机器人化学家 | 无 |
| S1 假设生成 | 人读文献是瓶颈 | 数据统计生成想法→评分→专家验证 | AstroInsight | hypothesis.py |
| S2 实验闭环 | 工具被调用而非被编排 | 反馈改变下一步 | StarWhisper、Coscientist/ChemCrow | experiment.py |
| S3 论文与审稿 | 实验结束到"能声称什么"的治理 | 结构化审稿+事实审计 | The AI Scientist、Robin | paper.py |
| 人的位置 | 责任并未消失 | 高风险授权、结论签署 | 安全边界、verification gap | --auto_approve 门 |

**B 线一句话**：想法经过评分门 → 实验反馈改变下一步 → 稿件可溯源 → 人签署结论。

---

## 案例与实践的分工原则

**案例**（只讲不动手）进门槛：公认代表作、正式来源和独立评测、能说明阶段跃迁、一张图讲清。**实践**（学生当场做）进门槛：普通笔记本 10 分钟内出结果、每步只引入一个新概念、失败也有教学价值。

| 线 | 案例 | 实践 |
|---|---|---|
| A 前史 | AlphaFold2、Pangu/GraphCast | 无 |
| A1 | UniRep/TAPE/ProtTrans/ESM-1b | nanoGPT 文本版 → 四域切换 |
| A2 | ESM、ClimaX、scGPT | objective_contrast、transfer_probe |
| A3 | ESM3、Aurora、Evo2 | multihead、route_decision |
| A 汇流 | GNoME、RFdiffusion、MatterGen、NeuralGCM | route_decision |
| B 前史 | Adam/Eve、mobile | 无 |
| B V0 | FunSearch/AI-Descartes | autoresearch 起步 |
| B1 工具编排 | Coscientist/ChemCrow | tools.py 合同 |
| B2 反馈回流 | SAMPLE/小来/StarWhisper | experiment.py |
| B3 研究状态 | The AI Scientist/Robin | state.py、paper.py |

## 作业：最小闭环证据包

课后提交七项：双轨蓝图、真实或脱敏输入样例、Agent Prompt 与完整轨迹、可执行评价器及结果、至少一条失败/反证、反馈后的下一步、一段"当前能声称什么/不能声称什么"。

评分共 100 分：科学问题与数据边界 15；基座模型与迁移设计 20；工具、反馈与状态闭环 20；评价器、运行证据和可复核性 25；停止条件、责任与结论边界 20。

---

## 文档导航

| 文档 | 身份 |
|---|---|
| [README.md](../README.md) | 仓库入口与快速开始 |
| [course-outline.md](course-outline.md) | 本文件：课程大纲 |
| [course-outline-25p.md](course-outline-25p.md) | **25页逐页大纲**：每页四段式（旧瓶颈→技术跃迁→论文锚点→实操增量） |
| [speaker-script.md](speaker-script.md) | **逐页讲稿**：可直接照讲的口播稿，含舞台指令和关键句 |
| [instructor/teaching-guide.md](instructor/teaching-guide.md) | A 线课堂操作讲稿 |
| [instructor/ai-scientist-guide.md](instructor/ai-scientist-guide.md) | B 线课堂操作讲稿 |
| [evidence-pack-template.md](evidence-pack-template.md) | 课后最小闭环证据包模板 |
| [gpt-like-science-landscape.md](gpt-like-science-landscape.md) | GPT-like 科学模型调研全景 |
| [research-notes.md](research-notes.md) | 调研笔记与来源 |
