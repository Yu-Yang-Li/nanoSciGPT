# nanoSciGPT 课堂 Skills

课程对学生只提供三个入口：

跟着15页PPT操作时，见[当前光变课堂提问](../docs/classroom-lightcurve-prompts.md)。P1、P2使用baseline，P3、P4使用科学语言，P9—P11使用研究Skill。请读取本仓库版本，不依赖教师机器的全局同名Skill。

1. [`research-baseline-builder`](research-baseline-builder/SKILL.md)：用当前数据跑第一份分类或回归，解释结果，也讨论只有少量标签时怎么办。教师使用 StarLightCurves；学生仍可选自己的数据、LAMOST 示例或已有模板。
2. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：先用文本理解预训练和微调，再用光变片段做同一三分类任务。蛋白质、DNA、SMILES、天气、晶体、三维结构、图像、光谱和连续场入口仍保留。
3. [`ai-scientist-research-loop`](ai-scientist-research-loop/SKILL.md)：由当前 Qoder 自带模型主持研究，尽量复用固定版本的 `karpathy/autoresearch`、`SakanaAI/AI-Scientist` v1 和 `SakanaAI/AI-Scientist-v2` 的代码与方法，进行小规模模型迭代、计算研究和研究分支探索。

所有 Skill 都沿用学生已经提供的信息。想先听解释就先讨论，要求动手时再运行；结果回来后接着解释，不把三个提问变成强制流程。光变数据下载、训练与任务边界见[光变课堂说明](../docs/starlight-classroom.md)；其他课程数据来源见 [`data/manifest.json`](../data/manifest.json)。

默认读 [`qoder-research.md`](ai-scientist-research-loop/references/qoder-research.md)，不要求另外购买模型 API。原项目固定源码与可选的独立运行说明见 [`native-projects.md`](ai-scientist-research-loop/references/native-projects.md)。三个完整原版研究流程尚未验收；Qoder 的真实教学研究也要按实际运行证据说明，不能以源码准备或小模型基线代替。本仓库 `autoresearch/` 下的规则驱动脚本只在学生明确要求无 API 离线演示时使用，并明确标为历史演示；Qoder 自带模型主持的研究不属于这类规则演示。
