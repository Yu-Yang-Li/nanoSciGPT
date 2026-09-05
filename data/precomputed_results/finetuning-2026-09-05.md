# 微调与问答：现场运行中断时的备用结果

这些是2026年9月5日已保存的教师运行结果，不是当前学生的实验。原来的十份 JSON 展示的是 `smoke` 预训练与冻结表示测试；下面补充的是 `classroom` 预训练后的任务微调，以及另做的200步文本回答微调。两种预算的数字不要混作同一次实验。

## 先看微调究竟改变了什么

十类示例都从已经预训练的模型接着训练，下游任务更新了模型参数，结果另存，原预训练文件没有被覆盖。这里的“微调前”是预训练模型接刚初始化的任务头，不是已经充分训练过的冻结表示对照。

| 示例 | 这次预测什么 | 指标 | 微调前 | 微调后 |
|---|---|---|---:|---:|
| text | 文本标点密度类别 | 准确率，越高越好 | 1.0000 | 1.0000 |
| protein | 根据氨基酸组成构造的类别 | 准确率 | 0.5625 | 0.4375 |
| dna | 根据GC含量构造的类别 | 准确率 | 0.6250 | 0.8750 |
| smiles | ESOL实测水溶解度 | 平均绝对误差，越低越好 | 1.6025 | 1.9677 |
| weather | 生成场的平移速度 | 平均绝对误差，网格单位/帧 | 0.02969 | 0.02882 |
| crystal | 晶胞密度代理量 | 平均绝对误差，原子质量代理单位/Å³ | 0.74324 | 0.68357 |
| structure3d | 生成螺旋的螺距 | 平均绝对误差，坐标单位/圈 | 0.35905 | 0.29166 |
| image | 生成图像中的点源数量 | 平均绝对误差，个 | 1.35828 | 1.18324 |
| spectrum | 生成黑体谱的温度 | 平均绝对误差，K | 1384.56 | 1141.03 |
| field | 生成扩散场的扩散系数 | 平均绝对误差，时间单位的倒数 | 0.04026 | 0.03843 |

蛋白质和分子的结果变差了，照样展示。它们说明这次训练完成了参数更新，但没有得到更好的验证结果。文本分类没有变化。其他行只是这一次小样本实验的数值变化，没有据此进行统计显著性或真实科学任务性能判断。

蛋白质组成、DNA的GC含量和文本标点类别都是教学构造的标签；后三者的分类不分别等于蛋白功能、变异效应或问答。天气、晶体、三维结构、图像、光谱、物理场使用生成的教学数据，不是相应观测或实验数据。输入、来源及单位见[内置数据说明](../README.md)与[数据清单](../manifest.json)。

原始数值、参数更新检查及任务来源保存在[十类微调记录](../../docs/acceptance/training-ten-domains-2026-09-05/acceptance.json)的 `domains` 字段。上表只做了小数位显示处理，没有筛掉不改善的结果。

## 再看文本从接龙到回答

同一文本预训练模型学习八组短问答，训练误差只计算回答部分。以下取自200步运行，字符串使用 JSON 写法；`\n` 表示换行，并不是被省略的回答。

| 输入 | 微调前的实际输出 | 微调后的实际输出 | 这是什么题 |
|---|---|---|---|
| `What is DNA?` | `"\n"` | `"A sequence of bases.\n"` | 训练原题 |
| `What is a protein?` | `"\n"` | `"A chain of amino acids.\n"` | 训练原题 |
| `Describe DNA.` | `"\n"` | `" sequed d t for a sam acigheleng"` | 换个问法 |
| `Describe a protein.` | `"\n"` | `"Lig t arnom aata.\n"` | 换个问法 |
| `Describe a spectrum.` | `"\n"` | `"Le t fom amight.\n"` | 换个问法 |

八道原题都能给出训练中的答案，但三道改写题均未正确回答。可以用它说明回答微调怎样训练、为什么记住原题还不等于学会回答新问题，不能把参考答案当作模型生成的内容。

[200步完整输出](../../docs/acceptance/training-ten-domains-2026-09-05/text-sft-200.json)保存八道原题、三道改写题、参考答案和训练记录。不要误用十类汇总文件中的旧20步 `text_sft` 字段代替这次200步结果；旧失败记录保留不改。

## 环境恢复后自己运行

从仓库根目录运行，只选自己的领域；输出目录须未使用过。下面是可复制的复现命令，不是历史命令原文。

```powershell
python -m nanoscigpt.classroom --domain protein --profile classroom --out_root out/my-lesson
python -m nanoscigpt.tasks.downstream_demo --domain protein --ckpt out/my-lesson/protein/model/ckpt.pt --adaptation finetune --epochs 2 --max_samples 32 --out_dir out/my-lesson/protein/finetune
```

把两条命令中的 `protein` 一起换成所选示例即可。文本回答另用：

```powershell
python -m nanoscigpt.classroom --domain text --profile classroom --out_root out/my-text-lesson
python -m nanoscigpt.tasks.text_sft --ckpt out/my-text-lesson/text/model/ckpt.pt --steps 200 --out_dir out/my-text-lesson/text/sft
```

教师整组复现可用 `python scripts/verify_training_lesson.py --output out/my-complete-check`。当前脚本的文本回答预算已是200步；[第一轮命令原文](../../docs/acceptance/training-ten-domains-2026-09-05/commands.json)中仍是20步，两者不是同一运行记录。所有命令都需要先安装仓库依赖；不保证不同电脑上的耗时或数值完全相同。

这份文件只提供离线阅读的结果与复现方法，不携带已训练权重，不能从表格恢复训练，也不证明一次新的 Agent 对话已经执行成功。
