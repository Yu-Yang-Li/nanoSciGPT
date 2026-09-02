# nanoSciGPT 课堂 Skills

课程最终拆成六个职责独立的 Skill。目前已落地的部分如下：

1. [`research-baseline-builder`](research-baseline-builder/SKILL.md)：把有标签的科学问题变成第一份分类、回归、时序或图像基线；仓库自带 LAMOST `teff` 课程数据和四类代码模板。
2. [`nanogpt-pretraining`](nanogpt-pretraining/SKILL.md)：用 tiny Shakespeare 小样例分开看文本预训练和带标签任务头。
3. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：把同一思路迁移到蛋白质、DNA、SMILES、天气、晶体、三维结构、图像、光谱和连续物理场。
4. [`autoresearch-model-iteration`](autoresearch-model-iteration/SKILL.md)：读取已有模型 V0，先生成设置，再运行一个只改变训练预算的候选，并留下完整比较和下一步。
5. [`ai-scientist-v1-workflow`](ai-scientist-v1-workflow/SKILL.md)：读取已完成的 AutoResearch 比较，生成单一路线的相关工作、表、图、证据映射、短稿和规则审查。
6. [`ai-scientist-v2-tree-search`](ai-scientist-v2-tree-search/SKILL.md)：从 v1 的单一路线继续，建立两条可比较路线，逐条批准执行并按同一评价器作出取舍。

所有 Skill 都沿用学生已经提供的信息，每轮推进一项可检查的操作。课程数据与来源见 [`data/manifest.json`](../data/manifest.json)。

当前实现的是普通电脑可运行的基础教学能力。The AI Scientist v1/v2 分别用于解释单路线计算研究和多路线比较，均不等同于原项目的完整复现。
