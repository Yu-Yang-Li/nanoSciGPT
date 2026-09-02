---
name: nanogpt-pretraining
description: Use when a student wants a first hands-on language-model lesson, asks how pretraining differs from a labeled downstream task, or wants to run the bundled text example before moving to scientific data.
metadata:
  short-description: 用一个小文本模型看懂预训练和下游任务
---

# nanoGPT 预训练

用仓库里的小文本模型，让学生亲眼看到两段不同的训练：先从无标签文本中练习预测下一个字符，再固定这份预训练表示，训练一个简单任务头。课堂中的第二段叫“任务头训练”，不是微调；微调还会继续更新预训练模型参数，本样例暂不执行。这里借用 nanoGPT 的最小训练思路，不宣称复现 GPT-2 的规模或成绩。

## 从学生当前的问题开始

学生已经问“预训练是什么”时，直接用文本例子说明：一段文本本身就能提供“下一个字符是什么”的训练信号，不需要专家逐句标注。学生已经有运行结果时，先读结果，不让他重跑。

需要运行时，从同时包含 `pyproject.toml`、`nanoscigpt/` 和 `data/` 的仓库根目录开始。默认 `python` 不可用时，运行 `scripts/find_course_python.ps1 -RequiredModules numpy,torch`，再把返回的解释器放到下面命令最前面。

```powershell
python -m nanoscigpt.classroom --domain text --profile classroom --out_root out/classroom
```

第一次只想确认环境时，把 `classroom` 换成 `smoke`。仓库使用 `data/text/` 中的 tiny Shakespeare 课程文本，训练代码位于 `nanoscigpt/core/`；这一步默认离线运行。

## 和学生一起读结果

先打开 `out/classroom/text/run_report.json`，沿着其中的路径查看：

- `model/train_log.json`：这次预训练的迭代数和最佳验证损失摘要；
- `model/ckpt.pt`：这次训练留下的模型参数；
- `downstream/downstream_result.json`：固定预训练表示后，简单任务头得到的结果。

文本下游任务按标点密度构造标签，只用于展示“同一份表示怎样接到一个有标签任务”，不是自然语言理解 benchmark。验证损失下降说明训练过程跑通；任务头有输出说明表示已被接入一个具体任务。这两件事都不能单独证明模型理解了语言机制。

学生问预训练和微调的区别时，按这三个名称回答：预训练从原始文本学习常见关系；任务头训练冻结原模型，只学习怎样从已有表示得到标签；微调则让预训练模型参数也随具体任务继续改变。

## 自然接到科学数据

结果读清以后，只问学生想把哪种科学对象放进同一套思路。接下来需要改变的是模型读入的单位、必须保留的关系和训练时要猜的内容；不必先改 Transformer 的名字。

例如学生说“我先跑文本版”，可以这样接：

> 我们先让小模型读字符并预测下一个字符。运行上面的 `smoke` 命令，把最后几行和 `run_report.json` 的路径发回来；我再和你分开看预训练结果与任务头结果。
