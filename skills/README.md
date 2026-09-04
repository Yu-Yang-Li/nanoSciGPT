# nanoSciGPT 课堂 Skills

课程对学生只提供三个入口：

1. [`research-baseline-builder`](research-baseline-builder/SKILL.md)：把有标签的科学问题变成第一份分类、回归、时序或图像基线；仓库自带 LAMOST `teff` 课程数据和四类代码模板。
2. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：先用文本理解预训练，再把同一思路迁移到蛋白质、DNA、SMILES、天气、晶体、三维结构、图像、光谱和连续物理场。
3. [`ai-scientist-research-loop`](ai-scientist-research-loop/SKILL.md)：读取已有模型结果，依次完成一次可比较的模型迭代、单路线计算研究和多路线比较。

所有 Skill 都沿用学生已经提供的信息，每轮推进一项可检查的操作。课程数据与来源见 [`data/manifest.json`](../data/manifest.json)。

AutoResearch、The AI Scientist v1 和 v2 的详细说明已并入第三个入口的 `references/`，不再要求学生切换 Skill。当前实现的是普通电脑可运行的基础教学能力；v1/v2 分别用于解释单路线计算研究和多路线比较，均不等同于原项目的完整复现。
