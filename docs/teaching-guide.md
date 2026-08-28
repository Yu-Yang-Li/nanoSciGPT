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

## 第三级 A2b：表征迁移——我们自己的模型（约 5 分钟）

### 讲什么

用自己的 A1 checkpoint 做三方对照：one-hot / 随机初始化编码器 / 自己预训练的编码器。
这不是调用别人的大模型，而是问：**我们刚训出的蛋白 nanoGPT 到底学到了什么可迁移的东西？**

### 操作

```bash
python -m nanoscigpt.tasks.transfer_probe
```

输出（本机实测，size=300）：

```
one-hot:            1.000
random encoder:     0.983
pretrained encoder: 0.950
transfer delta: -0.050
```

### 诚实边界（必须讲）

**这是本仓库最重要的一页。** 迁移增益是负的——我们用 450 条序列“预训练”的模型，在下游任务上不如直接 one-hot。这不是实验失败，这正是课程要证明的事：

> "450 条序列训不出基座。真实蛋白基座（ESM）用了 2.5 亿条序列——比我们多六个数量级。数据规模不够时，'预训练'这个词只是自我安慰，正确路线是回到专用模型。"

这个负结果直接衔接 A3b 的路线决策：五问中"迁移证据"一栏，学生应诚实回答"否"。

### ESM 怎么讲

不运行、不带权重，只在讲稿里讨论：ESM-2 在 2.5 亿序列上用 MLM 目标训练，同样的架构思想在足够数据下确实产生了可迁移的蛋白质表征。我们的小实验展示的是"机制相同、规模决定成败"——这正是从 nanoGPT 到科学基座的核心分界线。

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

---

## B线：autoresearch 虚拟 AI Scientist（约 15 分钟）

### 讲什么

A 线建好模型后，真正的问题是：科研过程怎么闭环。autoresearch 是一个刻意不用 LLM 的规则驱动科学家，它把“自主科研”拆成五件可检验的事，全部在本仓库真实执行：

1. **可执行动作**：每一轮只调用一个工具合同（`tools.py`）。
2. **工具合同**：不在 `CONTRACTS` 里的操作被拒绝——这是边界。
3. **形式化评价器**：`evaluator.py` 只认 design/ran/evaluated 三级，绝不把“跑过”说成“科学结论”。
4. **反馈改变下一步**：H1 失败就停，H2 失败记为“发现”，不是报错。
5. **跨轮研究状态**：假设、证据、未决问题持久化到 `research_state_<domain>.json`，重跑直接恢复。

### 操作（现场跑，每域约 20 秒）

```bash
# 文本域：全流程，模拟人工批准
python -m autoresearch.run --domain text --fresh --auto_approve

# 蛋白域：含迁移探针，会看到“迁移增益为负”的诚实结论
python -m autoresearch.run --domain protein --fresh --auto_approve

# 课堂演示：不带 --auto_approve，人工门真实等待学生输入
python -m autoresearch.run --domain text --fresh

# 跨轮恢复：不带 --fresh，直接到终态
python -m autoresearch.run --domain text
```

### 预期与停止边界

- protein 的 H2（预算加倍）和 H3（迁移）会被标记为 **refuted**——这正是课程要点：450 条序列撑不起基座主张。让学生亲眼看到“系统用证据说‘不’”。
- 所有结论轮都会写清“能声称什么/不能声称什么”，这就是 scientist-in-the-loop 的边界。
- 预算增加这类高成本动作必须过人工门；`--auto_approve` 只是课堂模拟，真实使用要人签字。

### 学生动手点

- 改 `evaluator.py` 里的阈值，观察 H2/H3 结论翻转。
- 给一个新域（如 RNA）加一个工具合同，跑通 prepare→train→sample。
- 阅读 `research_state_protein.json`，把证据链复述一遍。

