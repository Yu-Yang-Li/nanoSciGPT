---
name: nanogpt-pretraining
description: Use when a student wants a first hands-on language-model lesson, asks how pretraining differs from a labeled downstream task, or wants to run the bundled text example before moving to scientific data.
metadata:
  short-description: 用一个小文本模型看懂预训练和下游任务
---

# nanoGPT 预训练

用 tiny Shakespeare 文本训练约十万参数的小模型，再用人工整理的短问答继续训练。前者练习预测下一个字符，后者只对回答部分计算误差。这里借用 nanoGPT 的最小训练思路，不宣称复现 GPT-2 的规模或成绩。

## 从学生当前的问题开始

学生已经问“预训练是什么”时，直接用文本例子说明：一段文本本身就能提供“下一个字符是什么”的训练信号，不需要专家逐句标注。学生已经有运行结果时，先读结果，不让他重跑。

需要运行时，从同时包含 `pyproject.toml`、`nanoscigpt/` 和 `data/` 的仓库根目录开始。默认 `python` 不可用时，运行 `scripts/find_course_python.ps1 -RequiredModules numpy,torch`，再把返回的解释器放到下面命令最前面。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
python -m nanoscigpt.core.trainer --domain text --out_dir out/text-lesson-01/model --device cpu --max_iters 1000 --eval_interval 15 --eval_iters 5 --batch_size 8 --block_size 64 --n_layer 2 --n_head 2 --n_embd 64 --seed 1337
```

仓库使用 `data/text/` 中的 tiny Shakespeare 课程文本，训练代码位于 `nanoscigpt/core/`；这一步默认离线运行。重新实验使用新的输出目录，保留原文件。已有本轮结果时沿用，不重复训练。

接着学习多种问法：

```powershell
python -m nanoscigpt.tasks.text_sft --ckpt out/text-lesson-01/model/ckpt.pt --pairs data/course/text_qa_phrasings.json --steps 800 --out_dir out/text-lesson-01/sft
```

数据包含32条训练问答和8条验证问答。同一概念在训练中有多种问法，验证保留其他组合。它们是人工教学短句，不是自然问答 benchmark。验证问题不参与本轮梯度更新，但验证集已参与课程方案的开发，不能称为独立盲测。旧8题默认仍保留，课堂使用上面的 `--pairs` 文件。

## 和学生一起读结果

读取本次输出：

- `model/train_log.json`：这次预训练的迭代数和最佳验证损失摘要；
- `model/ckpt.pt`：验证损失最低时的模型参数，不一定是最后一步；
- `sft/sft_result.json`：微调前后实际回答、参考答案、验证损失和参数变化。

学生问“训练前后有什么不同”，看相同问题的前后回答；问“换个问法呢”，看 `samples` 的验证问法。`training_samples` 与 `samples` 分别计数，展示实际正确和错误例子。参考答案与生成文本分开。还想看文本接龙时，查看 `python -m nanoscigpt.core.sampler --help` 并对本轮预训练权重采样。

这些验证问法针对熟悉概念，不是新知识测试。短句正确不意味着长文本写作或陌生知识问答也已学会。错误可以直接指出；几个回答不足以确定错误来自哪个内部机制。旧标点分类不属于本次问答实践，也不拿它的准确率证明语言能力。

学生问预训练和微调的区别时，按这三个名称回答：预训练从原始文本学习常见关系；任务头训练冻结原模型，只学习怎样从已有表示得到标签；微调则让预训练模型参数也随具体任务继续改变。

## 自然接到科学数据

结果读清以后，只问学生想把哪种科学对象放进同一套思路。接下来需要改变的是模型读入的单位、必须保留的关系和训练时要猜的内容；不必先改 Transformer 的名字。

例如学生说“我先跑文本版”，可以这样接：

> 我们先让小模型读文本，练习预测下一个字符，再用短问答接着训练。我先检查数据和环境，跑完一起看它实际说了什么。
