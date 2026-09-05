---
name: nanoscigpt-scientific-language
description: Use when a student wants to turn protein, DNA, molecule, weather, crystal, 3D structure, image, spectrum, or physical-field data into a small runnable scientific pretraining lesson in the nanoSciGPT repository.
metadata:
  short-description: 从语言模型走到科学数据预训练
---

# nanoSciGPT Scientific Language

这门实训不要求学生先会写 Transformer。可以先用文本热身，也可以直接选一种科学数据；先跑通一个很小的预训练模型，再把模型接到一个具体任务上。

## 从学生已经选择的对象开始

学生已经说“蛋白质”“天气网格”或给出数据路径时，直接沿用，不重新让他选。还没有对象时，只问：

> 你想先试哪一种数据？可以说蛋白质、DNA、分子、天气、晶体、三维结构、图像、光谱或连续物理场。

蛋白质适合教师贯穿讲解，但不是默认答案。学生还只想理解文本语言模型时，先运行`text`示例；已经选好科学对象时，直接进入相应示例。

## 把“科学语言”说具体

运行以前，先用学生熟悉的话说明三件事：模型一次读什么单位、哪些关系不能被打散、训练时让它猜什么。例如蛋白质是氨基酸及其前后联系，天气是相邻网格和时间变化，光谱是连续波长上的谱线形状。这里的“语言”指可学习的排列与关系，不等于所有科学数据都要强行变成文字。

仓库提供十个课堂样例：`text`、`protein`、`dna`、`smiles`、`weather`、`crystal`、`structure3d`、`image`、`spectrum`、`field`。先在仓库根目录运行预检：

```powershell
python -m nanoscigpt.classroom --list
```

环境可用后，只运行学生选中的一个领域：

```powershell
python -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/classroom
```

文本热身使用同一入口：

```powershell
python -m nanoscigpt.classroom --domain text --profile classroom --out_root out/classroom
```

把 `protein` 换成学生选择的领域；如果先做文本热身，就使用`--domain text`。使用能成功导入仓库依赖的 Python，默认`python`不可用时再运行`scripts/find_course_python.ps1 -RequiredModules numpy,torch`。

## 怎样读运行结果

一轮训练以后，先看 `out/classroom/<domain>/run_report.json`，再沿其中的路径看训练记录、生成样例或表示预览，以及 `downstream_result.json`。用两句话把两段训练分开：

- 预训练结果说明模型是否学会了当前数据中的常见关系；
- 下游结果说明这些表示是否能被一个具体任务使用。当前课程示例冻结预训练表示，只训练一个简单任务头；只有继续更新预训练模型参数时才称为微调。

学生要求“预训练和微调都跑一遍”时，不止停在冻结表示。十类课程数据都可以接着在原checkpoint上运行真实任务微调，保存到新的目录：

```powershell
python -m nanoscigpt.tasks.downstream_demo --domain protein --ckpt out/classroom/protein/model/ckpt.pt --adaptation finetune --epochs 2 --max_samples 32 --out_dir out/classroom/protein/finetune
```

沿用学生所选领域替换`protein`。检查`downstream_result.json`中的参数变化和前后评价，以及`finetuned.pt`；原预训练文件应保持不变。这里的“微调前”使用刚初始化的任务头，不是充分训练过的冻结表示对照。效果变差时也如实解释，不为了课堂演示筛掉失败结果。

学生接着说“再训练一下”时，用刚产生的`task_checkpoint`继续，沿用数据和抽样数量，另存到新目录。已经微调过的任务会恢复模型和任务头，继续前的评价应与上一轮结束时一致。不要重新拿预训练文件从头接一个任务头，却说是在延续上一轮。旧的六类结构化预训练文件没有保存任务标签的标准化参数，第一次任务微调仍建立新任务头；结果中的`head_initialization`会说明这一点，后续微调才接续保存完整的任务状态。

讲文本从接龙转向问答时，使用下面的回答微调，而不是把标点分类说成问答：

```powershell
python -m nanoscigpt.tasks.text_sft --ckpt out/classroom/text/model/ckpt.pt --steps 200 --out_dir out/classroom/text/sft
```

这个示范用单线程CPU接着原文本模型学习八组短问答，只对回答部分计算误差。`sft_result.json`分别保存原题（`training_samples`）和换个问法（`samples`）的实际回答，以及验证损失和参数变化。先展示原题，再换个问法：记住八道题不等于会回答其他问题。验证题只改写了相同概念的问法；小模型可能仍然不会正确回答，不能把参考答案展示成模型输出，也不能把这个示范说成通用聊天模型。

验证损失下降只代表这个小模型按预定目标学到了东西，不等于发现了科学机制。天气、晶体、三维结构、图像、光谱和物理场是仓库生成的教学夹具；蛋白质、DNA、SMILES与文本来自列明来源的课程子集。介绍数据时读取 `data/manifest.json`，用“课程样例”“生成夹具”或“真实来源子集”说清身份。

如果现场运行中断，读取 `data/precomputed_results/<domain>.json` 继续解释结果格式，并明确说这是仓库预先保存的 `smoke` 备用结果，不能说成本次运行结果。保留本次失败日志，待环境恢复后再运行，不用备用指标覆盖失败记录。

学生带来自己的数据时，先读可用文件和字段。蛋白质CSV已有[自有数据入口](references/student-protein.md)：可直接接入序列列及数值目标列，没有标签就只预训练；不要再套课程的组成标签。其余格式尚未适配时，先整理处理单位、上下文、预训练目标和下游任务，再决定是否修改领域接口。不要把课程样例的指标写成学生数据的结果。

## 每轮怎样接话

按学生希望自己操作还是请助教代跑来接话。你已经能读取数据并执行命令时，完成预检后继续运行他选定的小实验，不要求学生反复粘贴你已经能看到的结果。学生想自己动手时，再分步给命令。只问确实影响下一步的缺失信息，不一次发出所有领域的命令。

例：学生说“我选蛋白质”。助教可以回答：

> 好，我们让模型逐个读取氨基酸，保留原有顺序和每条序列的边界，先练习预测下一个氨基酸。我先检查课程数据和环境，能运行的话就用一个小模型带你做。
