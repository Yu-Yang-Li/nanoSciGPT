# nanoSciGPT 课堂 Skills

课程对学生只提供三个入口。课堂从 [十一方向 Notebook](../notebooks/eleven_fields.ipynb) 起步，命令行等价入口是 `python -m nanoscigpt.tasks.eleven --field field`。学生可能怎样提问见 [课堂提问](../docs/classroom-prompts.md)。请读取本仓库版本，不依赖教师机器上的全局同名 Skill。

1. [`research-baseline-builder`](research-baseline-builder/SKILL.md)：先得到第一份结果并解释它。十一方向里，“只猜平均值”和“不预训练”两行就是最先要看的基线；学生也可以带自己的表格数据。
2. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：讲清一种科学观测怎样切成 token，再带学生预训练、接下游读出，比较不同无标签数据量。
3. [`ai-scientist-research-loop`](ai-scientist-research-loop/SKILL.md)：接着同一个方向做模型迭代和小研究。课堂实例是扩散场上的自动研究（`autoresearch.eleven_search`），由 DSK 每轮提一个改动，和随机搜索比较；方法上参考 `karpathy/autoresearch`、`SakanaAI/AI-Scientist` v1 和 `SakanaAI/AI-Scientist-v2`。

所有 Skill 都沿用学生已经提供的信息。想先听解释就先讨论，要求动手时再运行；结果回来后接着解释，不把课堂提问变成强制流程。十一方向的数据来源见 [data/eleven/SOURCES.md](../data/eleven/SOURCES.md)，五种子结果见 [docs/pretraining-scale/shared-checkpoints](../docs/pretraining-scale/shared-checkpoints/README.md)。

三个完整原版研究流程尚未验收。原项目固定源码与独立运行说明见 [`native-projects.md`](ai-scientist-research-loop/references/native-projects.md)。2026 年 9 月的光变课堂入口（[光变课堂提问](../docs/classroom-lightcurve-prompts.md)、[光变课堂说明](../docs/starlight-classroom.md)）仍保留，学生明确要继续光变时再用。本仓库 `autoresearch/` 下的旧规则脚本只在学生明确要求无 API 离线演示时使用，并标为历史演示。
