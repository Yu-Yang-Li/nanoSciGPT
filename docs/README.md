# 资料导航

## 当前课堂材料

- [16页课堂大纲](current/course-outline-16p.md)：页码、时间、案例与课堂操作。
- [16页页面文字](current/slide-copy-16p.md)：每页实际出现的文字。
- [16页讲稿](current/speaker-script-16p.md)：教师口播与页间衔接。
- [16页PPT—Skill—CLI课堂对照表](current/ppt-skill-teaching-map-16p.md)：每页学生说法、实际命令、产物与转场。
- [自定义科学数据接入说明](current/custom-domain-guide.md)：接入自己数据前需要补齐的五个接口。
- [六个课堂 Skill](../skills/README.md)：学生从基线走到AI Scientist v2的六段入口。
- [内置数据说明](../data/README.md)与[数据来源清单](../data/manifest.json)：课程数据身份、来源与使用边界。

这五项是当前正式交付。页面文字、大纲和讲稿必须连续覆盖P1—P16，时间从0分钟连续到90分钟。

## 代码与证据

- `python -m nanoscigpt.classroom --list`：检查十类内置数据。
- `python -m nanoscigpt.classroom --describe spectrum`：查看一种数据的课程表示、训练目标和样例身份，不启动训练。
- `python -m nanoscigpt.classroom --domain protein`：运行一个课堂示例。
- `python -m nanoscigpt.evidence_pack --help`：查看结课证据包命令。
- [上游项目与适配边界](upstream-adaptation.md)：说明哪些代码来自或参考外部项目。
- [Scientific Language Skill的GLM-5.3实测](skill-evals/scientific-language-glm53-2026-09-03.md)：无Skill对照、修改前失败与修改后回答。
- [nanoGPT预训练与微调Skill的GLM-5.3实测](skill-evals/nanogpt-pretraining-glm53-2026-09-03.md)：检查课堂所说的“微调”与模型参数是否真的更新。
- [AI Scientist旧研究指南](ai-scientist-guide.md)：保留为研究档案，不作为当前六Skill的学生入口。

## 研究档案

根目录下未放入`current/`的12页、14页、17页、25页版本及早期讲稿，记录了课程形成过程。它们不再提供当前课堂命令，也不要求与16页正式版逐句同步。
