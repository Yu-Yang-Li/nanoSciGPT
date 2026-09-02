# nanoSciGPT 课堂 Skills

这三个 Skill 对应课程中连续的三段实践：

1. [`research-baseline-builder`](research-baseline-builder/SKILL.md)：把有标签的科学问题变成第一份分类、回归、时序或图像基线；仓库自带 LAMOST `teff` 课程数据和四类代码模板。
2. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：从 nanoGPT 文本预训练类比到蛋白质、DNA、SMILES、天气、晶体、三维结构、图像、光谱和连续物理场。
3. [`ai-scientist-research-loop`](ai-scientist-research-loop/SKILL.md)：从已有 `run_report.json` 生成 autoresearch 计划，完成一次同口径比较，并根据结果决定下一步。

它们都是对话式课堂助教：沿用学生已经提供的信息，每轮推进一项可检查的操作。课程数据与来源见 [`data/manifest.json`](../data/manifest.json)。

当前实现的是普通电脑可运行的基础教学能力。The AI Scientist v1/v2 在第三个 Skill 中用于解释研究路线，尚不等同于原项目的完整复现。
