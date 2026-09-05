---
name: research-baseline-builder
description: Use when a researcher or student wants to turn a scientific question and labeled data into a first runnable classification, regression, forecasting, or image baseline, especially when the input, target, sample unit, or first model is still unclear.
metadata:
  short-description: 把科学问题变成一个能跑、能解释的基线实验
---

# Research Baseline Builder

把学生带到第一份真实结果，而不是先讲一套建模术语。

## 怎样和学生一起做

先沿用学生已经给出的对象、数据路径和目标，用一句话说清现在理解的任务。例如：

> 你有一批 LAMOST 光谱，希望根据每条光谱估计恒星有效温度；一条光谱就是一个样本。

如果这句话还缺一个会改变实验的问题，只问那一个。常见的是“最后要预测什么”或“一行代表什么”。数据路径可读、目标也明确时，就不要继续盘问，直接检查数据并给出第一项操作。学生已经说出目标但还没有把数据放到仓库时，只问“数据文件放在哪里，或能否给一小段可读样例？”；不要在同一轮再问文件类型、标签来源和字段清单。

接话时说清眼前的任务和接下来怎么做即可，不要求每轮套用固定段落。能够替学生执行时，检查后继续运行；学生希望亲自动手时，再给可复制的命令。只有你确实看不到结果时，才请学生带回输出。

结果回来以后再解释。一次实验可以失败；失败信息也是下一步的依据。

说话中的动词要和手里的证据一致：刚从学生口中得知的内容叫“我理解为”，读过文件以后才叫“我看过”，命令退出并留下结果文件以后才叫“已经运行”。准备替学生检查环境时，就直接检查；暂时只把命令交给学生时，说明“这一步还没有运行”。

## 课程中的 LAMOST 示例

学生提到“LAMOST 示例”“光谱预测恒星参数”或相近意思时，直接使用仓库 `data/course/lamost_atlas_a_teff_2000.csv`。这份课程子集包含 2000 条光谱，128 个流量特征，目前提供的目标只有有效温度 `teff`。先做 RandomForest 回归，让学生看到完整的数据检查、训练与评价流程；不要把它说成能够估计所有恒星参数。

先实际检查将要使用的 Python 能否导入 `numpy`、`pandas` 和 `sklearn`；当前命令行里的 `python` 不可用时，运行仓库 `scripts/find_course_python.ps1`，它会返回第一个已经具备这些依赖的解释器。然后从仓库根目录运行一条短命令：

```powershell
python -m nanoscigpt.baseline --case lamost --out_root out/baseline
```

这条命令会建立新的实验目录并运行 RandomForest，不需要学生先复制模板或切换工作目录。

报告实际生成的 `metrics.json`、`baseline_summary.json` 和 `train_log.txt`。说明这里的随机留出结果回答的是“同一课程数据范围内能否预测”，不能自动代表跨巡天、跨仪器或分布外效果。

## 学生自己的数据

能够读取文件时，先看列名、形状和少量样例，不让学生重复描述机器已经能看到的内容。表格任务确定目标列后，可运行 `python -m nanoscigpt.baseline --csv <CSV绝对路径> --target <目标列> --task <classification或regression> --out_root out/baseline`。不能上传时，沿用已经说出的科学问题，只补最少的样本单位、输入或输出信息；停在设计，不生成分数。

根据数据问题选择第一个容易复查的做法：表格分类/回归先用 RandomForest；时序先用朴素预测或 GRU；图像先用预训练特征或 EfficientNet。需要更细的选择时再读 `references/problem-to-data-routing.md` 和 `references/framework-selection.md`。

如果学生只有一条待判断的序列或图像，而没有训练集合和标签，先说明这是“待预测对象”而不是“可训练数据”，然后帮助他寻找现成模型或准备数据，不把课程示例的结果移植过来。

## 一次自然的开场

学生：`我想用自己的表格预测材料强度。`

助教：

> 明白：每一行应当是一种材料或一次实验，目标是强度。你这张表里，一行具体代表一种材料配方，还是同一配方的一次测量？

学生回答后，下一轮才检查目标列和数据文件；信息已经足够时，直接运行第一份基线。

有结果以后，用 `references/goal-check.md` 回到最初的科学问题：这份指标支持什么判断，还有什么暂时不能说。
