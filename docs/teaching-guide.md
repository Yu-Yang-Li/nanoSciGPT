# nanoSciGPT 课堂操作讲稿

面向：AI4S 实训营"科学模型专题实训（二）"A 线（领域基座模型）  
时长：约 25 分钟（对应 PPT P5–P10 的实操部分）  
前置：学生已安装 `pip install -e .` 和 `pip install -e .[esm]`

---

## 开场（1 分钟）

一句话定调：

> "今天不训练大模型。我们用同一个 GPT 核心，把 tokenizer 和数据从莎士比亚换成蛋白、DNA、分子——亲眼看到'科学对象语言化'只差一层插件。然后回答真正的问题：你的课题到底需不需要基座模型。"

---

## 第一级 A1：科学对象语言化（约 8 分钟）

### 讲什么

nanoGPT 的"下一字符预测"与 ESM 的 masked LM 是同一个思想。区别只在：对象从 Shakespeare 的 65 个字符，换成氨基酸的 20 个字母、DNA 的 4 个碱基、SMILES 的几十个化学符号。

### 操作（现场跑，每域约 1 分钟）

```bash
# 文本基线——和 nanoGPT 一模一样
python -m nanoscigpt.domains.text.prepare
python -m nanoscigpt.core.trainer --domain text --max_iters 100 --eval_interval 50 --eval_iters 10 --block_size 64 --batch_size 8 --n_layer 2 --n_head 2 --n_embd 64
python -m nanoscigpt.core.sampler --domain text --max_new_tokens 80 --num_samples 1
```

采样输出会是伪莎士比亚——像那么回事但没有意义。这就是预训练学到的"统计结构"。

```bash
# 换成蛋白——同一个核心，只换 tokenizer 和数据
python -m nanoscigpt.domains.protein.prepare --size 500
python -m nanoscigpt.core.trainer --domain protein --max_iters 100 --eval_interval 50 --eval_iters 10 --block_size 64 --batch_size 8 --n_layer 2 --n_head 2 --n_embd 64
python -m nanoscigpt.core.sampler --domain protein --max_new_tokens 40 --num_samples 2
```

采样输出是真实的氨基酸字母序列。指着代码问学生：

> "trainer.py 改过一行吗？没有。变了的只有 domains/protein/prepare.py——这 60 行就是'领域插件'的全部。"

### 板书要点

- 流式（text/DNA）：数据拼成长流，随机窗口采样——nanoGPT 原样。
- 独立序列（protein/SMILES）：每条序列独立，padding 必须被 attention 屏蔽——prot-gpt 的核心魔改。

### 停止边界

500 条 UniProt 不会学到蛋白质生物学。这一级只证明"机制可迁移"，不证明"能力可迁移"。

---

## 第二级 A2a：换预训练目标（约 5 分钟）

### 讲什么

CLM（GPT 家族，从左到右预测下一个）vs MLM（BERT/ESM 家族，挖空后双向恢复）。架构完全相同，差异只有两处：attention mask 和 loss 作用的位置。

### 操作

```bash
python -m nanoscigpt.tasks.objective_contrast
```

输出（本机实测）：

```
CLM: loss 2.9716 -> 2.7578 (0.1M params)
MLM: loss 3.0134 -> 2.8093 (0.1M params)
```

指着输出讲：

> "两个 loss 都在降——目标都能学。但 CLM 每步只对一个位置算 loss（下一个 token），MLM 对 15% 被挖空的位置都算 loss。这不是谁更好，而是下游用法不同：要生成就选 CLM，要表征就选 MLM。ESM 选了 MLM，所以它能做结构预测；GPT 选了 CLM，所以它能写文章。"

### 停止边界

300 iter 的小模型不会显出两者的真实差距。这一级展示的是"目标可以切换"，不是"哪个目标更优"。

---

## 第三级 A2b：表征迁移（约 5 分钟）

### 讲什么

"别人预训练好的表征可以直接取用"——不训练 ESM，只冻结它，在它上面训一个线性头。

### 操作

```bash
python -m nanoscigpt.tasks.esm_probe
```

输出（本机实测，size=300）：

```
ESM frozen + linear probe val acc: 1.000
one-hot + linear probe val acc:     0.967
delta = +0.033
```

### 诚实边界（必须讲）

合成任务太简单（疏水头部规则很容易学），+3.3% 低估了真实蛋白任务上 ESM 的优势。课堂上要讲清：

> "探针任务的难度决定迁移收益的可观测性。任务太简单时 everyone wins，看不出差距；真实蛋白定位任务上 ESM 对 one-hot 的优势远大于此——但那需要真实标注数据集。"

### 离线预案

ESM 权重已随仓库提供（`weights/esm2_t6_8M_UR50D.pt`，30MB）。默认从本地加载，无网络依赖。若权重缺失会自动回退到在线下载。

---

## 第四级 A3a：多任务接口（约 4 分钟）

### 讲什么

基座的最小结构隐喻：一个共享编码器 + N 个轻量任务头。任务行为差异全部来自头部，表征复用全部来自编码器。

### 操作

```bash
python -m nanoscigpt.tasks.multihead
```

输出（本机实测）：

```
shared encoder multi-task: cls_acc=1.000 reg_mae=0.2714
```

### 有价值的失败案例（强烈建议讲）

最初版本用 parity（前 3 个 token 的和的奇偶性）做分类任务，共享编码器只能到 45%——均值池化丢失顺序信息，而 parity 是顺序敏感任务。这不是 bug，是"表示瓶颈"的活教材：

> "当你发现多任务学不动时，先问池化方式是否丢掉了任务需要的信息，再问模型够不够大。"

### 停止边界

合成双任务（是否含 token 5 + token 均值）验证的是结构可行性，不是真实科学任务的增益。

---

## 第五级 A3b：证据决定路线（约 2 分钟）

### 讲什么

课程的核心主张：没有证据支持训练新模型时，必须降级路线。五问决策链把它变成可执行检查。

### 操作

```bash
# 交互式：逐题回答 y/n
python -m nanoscigpt.tasks.route_decision
```

输出示例（数据不足时）：

```
路线: use_specialized_model
依据: 第一处失败：无标签数据量是否达到万级以上（或领域公认预训练规模）？
      数据不足，预训练收益不可靠；回到专用模型路线。
```

### 收束

> "五问全过才训基座。任何一问失败都有降级路径——这不是保守，是证据驱动的工程决策。你的课题大概率停在专用模型，这是合格结论，不是失败。"

---

## 时间分配总表

| 级 | 内容 | 建议 |
|---|---|---:|
| 开场 | 定调：换插件不换核心 | 1 min |
| A1 | 四域语言化（重点 text→protein 对照） | 8 min |
| A2a | CLM vs MLM | 5 min |
| A2b | ESM 冻结探针 | 5 min |
| A3a | 共享编码器多任务 | 4 min |
| A3b | 五问路线决策 | 2 min |

## 学生动手点

每级留一个可改参数让学生课后探索：

1. A1：换 `--size 200` 或 `--max_iters 500`，观察 loss 变化；
2. A2a：改 `--mask_prob`（源码中），看 MLM 的 loss 曲线怎么变；
3. A2b：换探针任务难度（源码中的合成规则），看迁移收益怎么变；
4. A3a：换分类规则为顺序敏感任务，复现"表示瓶颈"失败；
5. A3b：用自己的课题诚实回答五问，生成 decision.json 作为作业素材。
