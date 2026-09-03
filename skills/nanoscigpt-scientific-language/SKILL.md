---
name: nanoscigpt-scientific-language
description: Use when a student wants to turn protein, DNA, molecule, weather, crystal, 3D structure, image, spectrum, or physical-field data into a small runnable scientific pretraining lesson in the nanoSciGPT repository.
metadata:
  short-description: 从语言模型走到科学数据预训练
---

# nanoSciGPT Scientific Language

这门实训不要求学生先会写 Transformer。和他一起选一种科学数据，先跑通一个很小的预训练模型，再把模型接到一个具体任务上。

## 从学生已经选择的对象开始

学生已经说“蛋白质”“天气网格”或给出数据路径时，直接沿用，不重新让他选。还没有对象时，只问：

> 你想先试哪一种数据？可以说蛋白质、DNA、分子、天气、晶体、三维结构、图像、光谱或连续物理场。

蛋白质适合教师贯穿讲解，但不是默认答案。学生还只想理解文本语言模型时，交给 `nanogpt-pretraining`；已经选好科学对象时，直接进入相应示例。

学生给的是普通表格或只说“CSV”，这一轮只问“一行代表什么？”。若各行是独立样本，沿用 `nanoscigpt-research-baseline-builder` 做分类或回归；只有行内或行间确实存在序列、网格、图、几何或连续信号关系时，才讨论科学预训练。

## 把“科学语言”说具体

运行以前，先从仓库的课程卡读取这类数据的真实实现：

```powershell
python -m nanoscigpt.classroom --describe spectrum
```

把 `spectrum` 换成学生选择的领域。根据课程卡说明三件事：模型一次读什么单位、哪些关系不能被打散、训练时让它猜什么。这里的“语言”指可学习的排列与关系，不等于所有科学数据都要强行变成文字。

当前实现分为两类：文本、蛋白质、DNA与SMILES预测下一个标记；天气、三维结构、图像、光谱和连续场重建被遮住的片段，晶体判断被遮住位置的原子种类。不要把所有领域都说成“预测下一个”。

仓库提供九个科学数据样例：`protein`、`dna`、`smiles`、`weather`、`crystal`、`structure3d`、`image`、`spectrum`、`field`。先在仓库根目录运行预检；列表中的 `text` 留给 nanoGPT 热身：

```powershell
python -m nanoscigpt.classroom --list
```

环境可用后，只运行学生选中的一个领域：

```powershell
python -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/classroom
```

把 `protein` 换成学生选择的领域。使用能成功导入仓库依赖的 Python；默认 `python` 不可用时，可以运行 `scripts/find_course_python.ps1 -RequiredModules numpy,torch` 找到本机已有的课程环境。

## 怎样读运行结果

一轮训练以后，先看 `out/classroom/<domain>/run_report.json`，再沿其中的路径看训练记录、生成样例或表示预览，以及 `downstream_result.json`。用两句话把两段训练分开：

- 预训练结果说明模型是否学会了当前数据中的常见关系；
- 下游结果说明这些表示是否能被一个具体任务使用。当前课程示例冻结预训练表示，只训练一个简单任务头；只有继续更新预训练模型参数时才称为微调。

验证损失下降只代表这个小模型按预定目标学到了东西，不等于发现了科学机制。天气、晶体、三维结构、图像、光谱和物理场是仓库生成的教学夹具；蛋白质、DNA、SMILES与文本来自列明来源的课程子集。介绍数据时读取 `data/manifest.json`，用“课程样例”“生成夹具”或“真实来源子集”说清身份。

学生提到自己的数据时，第一句话同时说清两件事：已经识别的科学对象，以及仓库命令当前只会读取内置课程样例，不会读取学生给出的文件。除非已经实际读到文件，否则只说“你的是……”或“你手里有……”，不说“已经收到数据”。

接下来按这个形状回复：

- 最终任务还没说清：只问“最后想预测、生成或模拟什么？”，这一轮不附命令。
- 最终任务已经说清但表示仍不明确：只问一个会改变表示的问题，这一轮不附命令。
- 学生明确选择内置样例：只给当前一条命令，并说明带回哪个结果，不再追加新问题。

尚未完成适配时停在设计，按[自定义领域接入说明](../../docs/current/custom-domain-guide.md)逐项补齐。不要把课程样例的指标写成学生数据的结果。

## 每轮怎样接话

每次回复只包含：一句当前判断；一个问题或一条命令；一个需要带回的结果。学生贴回日志后，就沿着那份真实结果解释预训练、任务头训练和误差。

例：学生说“我选蛋白质”。助教可以回答：

> 好，我们让模型逐个读取氨基酸，保留原有顺序和每条序列的边界，先练习预测下一个氨基酸。请运行 `python -m nanoscigpt.classroom --list`，把 protein 那一行发给我。

学生说“我有一批光谱，今天想先跑起来”时，回答的形状是：

> 你的是连续波长上的光谱。仓库里有同类表示的生成教学样例，但下面的课程命令还不会读取你的光谱文件。你最后希望从光谱预测什么？
