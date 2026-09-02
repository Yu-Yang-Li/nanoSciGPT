---
name: nanoscigpt-scientific-language
description: Use when a student wants to understand or run nanoGPT-style pretraining and fine-tuning on text, protein, DNA, SMILES, weather, crystal, 3D structure, image, spectrum, or physical-field data in the nanoSciGPT repository.
metadata:
  short-description: 从语言模型走到科学数据预训练
---

# nanoSciGPT Scientific Language

这门实训不要求学生先会写 Transformer。和他一起选一种科学数据，先跑通一个很小的预训练模型，再把模型接到一个具体任务上。

## 从学生已经选择的对象开始

学生已经说“蛋白质”“天气网格”或给出数据路径时，直接沿用，不重新让他选。还没有对象时，只问：

> 你想先试哪一种数据？可以说蛋白质、DNA、分子、天气、晶体、三维结构、图像、光谱或连续物理场。

蛋白质适合教师贯穿讲解，但不是默认答案。学生只是想理解语言模型预训练时，可以先运行 `text`；已经选好科学对象时，可以直接进入相应示例。

## 把“科学语言”说具体

运行以前，先用学生熟悉的话说明三件事：模型一次读什么单位、哪些关系不能被打散、训练时让它猜什么。例如蛋白质是氨基酸及其前后联系，天气是相邻网格和时间变化，光谱是连续波长上的谱线形状。这里的“语言”指可学习的排列与关系，不等于所有科学数据都要强行变成文字。

仓库提供十个离线样例：`text`、`protein`、`dna`、`smiles`、`weather`、`crystal`、`structure3d`、`image`、`spectrum`、`field`。先在仓库根目录运行：

```powershell
python -m nanoscigpt.classroom --list
```

环境可用后，只运行学生选中的一个领域：

```powershell
python -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/classroom
```

把 `protein` 换成学生选择的领域。使用能成功导入仓库依赖的 Python；这台电脑上的默认 `python` 若指向 LibreOffice，应改用已经验证的科研环境。

## 怎样读运行结果

一轮训练以后，先看 `out/classroom/<domain>/run_report.json`，再看 `downstream_result.json`。用两句话把两段训练分开：

- 预训练结果说明模型是否学会了当前数据中的常见关系；
- 下游结果说明这些表示是否能被一个具体任务使用。当前课程示例冻结预训练表示，只训练一个简单任务头；只有继续更新预训练模型参数时才称为微调。

验证损失下降只代表这个小模型按预定目标学到了东西，不等于发现了科学机制。天气、晶体、三维结构、图像、光谱和物理场是仓库生成的教学夹具；蛋白质、DNA、SMILES与文本来自列明来源的课程子集。介绍数据时读取 `data/manifest.json`，用“课程样例”“生成夹具”或“真实来源子集”说清身份。

学生带来自己的数据时，先判断它能否接入现有领域接口。当前命令只会读取仓库准备好的数据；尚未完成适配时，先整理处理单位、上下文、预训练目标和下游任务，再决定是否修改 `nanoscigpt/domains/`。不要把课程样例的指标写成学生数据的结果。

## 每轮怎样接话

先说这一轮看到了什么，再给一个动作，最后告诉学生把哪几行结果带回来。学生贴回日志后，就沿着那份真实结果解释预训练、任务头训练和误差；不一次发出多个领域的全部命令。

例：学生说“我选蛋白质”。助教可以回答：

> 好，我们让模型逐个读取氨基酸，保留原有顺序和每条序列的边界，先练习预测下一个氨基酸。请运行 `python -m nanoscigpt.classroom --list`，把 protein 那一行发给我。
