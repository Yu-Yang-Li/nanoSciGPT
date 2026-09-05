# nanoSciGPT 课堂 Skills

课程对学生只提供三个入口：

1. [`research-baseline-builder`](research-baseline-builder/SKILL.md)：把有标签的科学问题变成第一份分类、回归、时序或图像基线；仓库自带 LAMOST `teff` 课程数据和四类代码模板。
2. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：先用文本理解预训练，再把同一思路迁移到蛋白质、DNA、SMILES、天气、晶体、三维结构、图像、光谱和连续物理场。
3. [`ai-scientist-research-loop`](ai-scientist-research-loop/SKILL.md)：使用固定版本的 `karpathy/autoresearch`、`SakanaAI/AI-Scientist` v1 和 `SakanaAI/AI-Scientist-v2`，在缩小的教学设置下进行模型迭代、研究流程和研究树探索。

所有 Skill 都沿用学生已经提供的信息，每轮推进一项可检查的操作。课程数据与来源见 [`data/manifest.json`](../data/manifest.json)。

原项目的环境、固定源码版本、启动命令和验证范围见第三个入口的 [`native-projects.md`](ai-scientist-research-loop/references/native-projects.md)。三个完整原版研究流程尚未验收，不能把源码准备或小模型基线说成完整研究已经运行。本仓库 `autoresearch/` 下的规则驱动脚本只在学生明确要求无 API 离线演示时使用，并明确标为历史演示，不作为三个原项目的复现。
