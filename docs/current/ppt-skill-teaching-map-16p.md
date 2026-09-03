# 16页PPT—Skill—CLI课堂对照表

这份表供讲师备课。PPT负责讲清问题和发展脉络，Skill负责接住学生的自然表达，CLI只在需要留下真实结果时运行。学生不需要说页码、Skill名称或完整参数。

### P1｜科学问题怎样变成数据问题

- 讲清：观测光谱怎样落到输入、目标和评价。
- 你可以这样说：`我想用LAMOST课程光谱预测恒星有效温度，先帮我跑一个基线。`
- 接续：`nanoscigpt-research-baseline-builder`先核对目标，再运行`nanoscigpt-baseline --case lamost --out_root out/baseline`。
- 留下：数据说明、基线模型、`metrics.json`和`workflow_status.json`。用“已有专家参数时可以直接监督训练”转到P2。

### P2｜大量没有专家订正的数据怎样使用

- 讲清：同一种科学对象，有标签部分和无标签部分承担的作用不同。
- 你可以这样说：`我有一批光谱，只有少量有恒星参数，其余没有专家订正，最后想预测温度。`
- 接续：`nanoscigpt-scientific-language`沿用光谱和目标，只问当前缺少的一项；此时不训练。
- 留下：科学对象、标签情况和最终任务三项口头记录。用“先从原始数据本身构造训练信号”转到P3。

### P3｜从文本理解预训练和微调

- 讲清：无标签文本预测下一个字符；少量标签随后更新预训练模型和任务输出层。
- 你可以这样说：`先用文本例子带我跑一遍预训练和微调。`
- 接续：`nanogpt-pretraining`给出一条命令：`python -m nanoscigpt.classroom --domain text --profile classroom --out_root out/classroom`。
- 留下：`model/ckpt.pt`、`downstream/finetuned_ckpt.pt`、`run_report.json`和微调记录。用“换一种科学对象，首先要换模型读入的单位”转到P4。

### P4｜把科学数据变成模型能学习的“语言”

- 讲清：处理单位、保留关系和预训练目标共同决定科学表示。
- 你可以这样说：`我选蛋白质，先说明模型怎样读这种数据，再跑课程样例。`
- 接续：`nanoscigpt-scientific-language`先读`python -m nanoscigpt.classroom --describe protein`，再给所选领域的一条`python -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/classroom`命令。
- 留下：领域课程卡、`run_report.json`与`downstream_result.json`。科学领域当前是冻结表示接任务头，不称为text路径的完整微调。

### P5｜2023，专用模型登顶

- 讲清：ESMFold、RFdiffusion、Pangu-Weather、GraphCast、GNoME和CHGNet分别把一种科学任务做到很强。
- 你可以这样说：`刚才的蛋白质小模型，与ESMFold和RFdiffusion分别差在哪里？`
- 接续：不新增命令。沿用P4的处理单位、输入输出和任务结果，比较专用模型解决的具体问题。
- 留下：一组“输入—输出—架构—核心创新”的口头对照。转折是“一个任务做强，不等于一套表示可以复用”。

### P6｜2024，从专用走向基座

- 讲清：AlphaFold3、scGPT/scFoundation、NeuralGCM和GenCast开始扩大任务与对象范围。
- 你可以这样说：`如果同一套蛋白质表示还要服务第二个任务，需要增加什么？`
- 接续：不新增命令。只在P4现有模型上讨论第二个任务、适配方式和共同评价。
- 留下：一个可以共享表示的第二任务。转到“模型不仅复用表示，还开始生成候选”。

### P7｜2025，多任务、生成与跨数据域模拟

- 讲清：ESM3、OpenCRISPR-1、Aurora、MatterGen和UMA分别把统一表示推向多任务、生成和模拟。
- 你可以这样说：`如果模型不只预测标签，还要按条件生成候选，输入和评价要增加什么？`
- 接续：不新增命令。用学生选定的科学对象讨论条件输入、候选输出和后续验证。
- 留下：一项生成或模拟需求，以及对应评价。转到P8的更大数据边界。

### P8｜2026，跨模态、跨分布、跨对象

- 讲清：AlphaGenome、MAMMAL、Evo 2、UCE、CAPTAIN、BioMatrix和LOGOS分别扩展测量方式、数据来源或科学对象。
- 你可以这样说：`我的数据来自不同仪器或数据集，哪些差异必须单独检验？`
- 接续：不新增命令。把分布差异写成少样本、跨数据集或外部验证问题，不把前沿预印本当成熟方案。
- 留下：模型适用范围和一项跨分布检查。至此A线结束，P9开始研究迭代。

### P9｜模型跑通以后怎样继续改

- 讲清：研究迭代需要同一V0、唯一改动、同一评价和停止条件。
- 你可以这样说：`读取刚才的run_report，先帮我设计下一轮，只生成计划。`
- 接续：`autoresearch-model-iteration`运行`python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --plan_only`。
- 留下：`iteration_spec.json`。学生看过后再运行一轮，得到`comparison.json`和`research_state.json`。

### P10｜从AutoResearch到The AI Scientist v1

- 讲清：在一次模型比较前后补入研究问题、相关工作、结果整理、图表和写作。
- 你可以这样说：`把刚才的一轮比较接成一条计算研究路线，先只生成计划。`
- 接续：`ai-scientist-v1-workflow`运行`python -m autoresearch.v1 --domain protein --autoresearch-dir out/autoresearch/protein --out-dir out/ai-scientist-v1/protein --plan-only`；学生确认后才使用`--confirm-plan`。
- 留下：`plan.json`、`results.json`、图、证据映射、短稿和`workflow_state.json`。

### P11｜从The AI Scientist v1到v2

- 讲清：从一条研究路线扩展为多条可比较路线，并使用同一个评价器取舍。
- 你可以这样说：`沿用同一个V0和评价指标，先建立两条路线，不要运行。`
- 接续：`ai-scientist-v2-tree-search`运行`python -m autoresearch.v2 init --from-v1 out/ai-scientist-v1/protein/workflow_state.json --out-root out/ai-scientist-v2`。
- 留下：`tree_state.json`。序列示例可批准第二路线；结构化领域没有第二个安全变量时停在`design_only`。

### P12｜2004—2023，结果已经会改变下一步

- 讲清：Robot Scientist、Adam、Eve、ARES、Ada和Mobile Robotic Chemist如何让实验结果改变下一轮。
- 你可以这样说：`刚才comparison.json里的结果，实际改变了下一步什么？`
- 接续：不新增命令。用P9的`comparison.json`类比可读取的实验反馈，同时指出真实实验还需要仪器接口。
- 留下：动作、返回结果、下一步变化三项对应关系。

### P13｜2023—2024，研究目标接入具体工具

- 讲清：Coscientist、AI-Chemist“小来”、ChemCrow和SAMPLE把资料、计算与实验设备接到同一目标下。
- 你可以这样说：`我的课题下一步能调用哪些真实工具，每个工具会返回什么？`
- 接续：不新增命令。列出现有文献库、代码、数据库、模拟器或仪器，不虚构尚未接入的工具。
- 留下：工具名、输入、实际返回和失败记录四列。

### P14｜2025—2026，结果开始修改假设和路线

- 讲清：StarWhisper、Virtual Lab、Co-Scientist和Robin怎样跨轮保存假设、证据和下一步。
- 你可以这样说：`如果这次结果反对原假设，下一轮应该改参数、换路线还是改假设？`
- 接续：不新增命令。读取P9—P11已经留下的状态，判断变化发生在哪一层。
- 留下：原假设、支持或反对证据、修改后的下一步。

### P15｜把模型接进自己的科研过程

- 讲清：模型输出必须进入计算、实验或观测检查，结果再返回研究过程。
- 你可以这样说：`我下一步能做的是模拟，请把模型结果、模拟返回和下一轮调整接起来。`
- 接续：沿用已有六个Skill与状态文件；没有真实模拟器、实验或观测接口时只形成设计，不声称已运行。
- 留下：科学问题、模型输出、检查方式、返回结果、调整位置和人工责任。

### P16｜把今天的结果交给别人复查

- 讲清：提交的不是漂亮结论，而是数据身份、运行、评价、失败和结论边界。
- 你可以这样说：`把今天真实运行的结果整理成一份证据包。`
- 接续：根据已有文件运行`python -m nanoscigpt.evidence_pack --run-report out/classroom/protein/run_report.json --comparison out/autoresearch/protein/comparison.json --state out/autoresearch/protein/research_state.json --output out/evidence-packs/protein.md`。
- 留下：可以复查的Markdown证据包；没有运行报告时不生成分数。
