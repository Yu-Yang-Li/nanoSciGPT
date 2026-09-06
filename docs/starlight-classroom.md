# 用一份光变数据贯穿课堂

StarLightCurves 是公开的三分类时间序列数据。每条记录有 1024 个亮度数值和一个类别编号。课堂先直接分类，再用连续片段训练小型 GPT，最后围绕同一任务安排研究。文本 nanoGPT 仍作为理解预训练和问答微调的单独热身。

## 准备数据

从仓库根目录运行。使用已经安装本仓库依赖的 Python；这些实验默认 CPU 两线程，不要求显卡或额外模型 API。

```bash
python -m nanoscigpt.starlight_data prepare --download --output out/starlight/data
```

首次下载约 110 MB。也可以使用提前下载的官方 ZIP：

```bash
python -m nanoscigpt.starlight_data prepare --archive /path/to/StarLightCurves.zip --output out/starlight/data
```

程序检查原始文件校验值，保留官方 1000 条训练记录与 8236 条测试记录，再将训练记录分为 800 条训练、200 条验证。输出目录必须是新目录；已有数据准备完毕就直接复用，不重复下载或覆盖。

- `manifest.json`：来源、样本量、数据身份与校验值。
- `data-and-splits.npz`：完整固定划分，用于最终评价。
- `train-validation-only.npz`：不含测试集，供提出和比较改动时使用。
- `source-description.txt`：数据包自带说明。

来源：[数据说明](https://www.timeseriesclassification.com/description.php?Dataset=StarLightCurves)、[官方下载](https://www.timeseriesclassification.com/aeon-toolkit/StarLightCurves.zip)。仓库只提供下载与处理代码，不重新分发该数据包。

## 先做分类

```bash
python -m nanoscigpt.starlight_lstm --data out/starlight/data/data-and-splits.npz --output out/starlight/lstm-01 --seed 45 --steps 800 --threads 2
```

课堂主示范使用 LSTM：每 32 个连续数值组成一段，按先后顺序读完一条曲线，预测类别 1、2、3。两层 LSTM、隐藏宽度 64，共 62723 个参数，从头监督训练 800 步。使用全部 800 条训练标签，均衡类别采样，验证集宏 F1 选择检查点。输出 `metrics.json`、`history.json`、`predictions.npz`、`lstm.pt` 和实际日志。预测文件中的 `lstm` 是按类别 1、2、3 排列的概率；`y_true` 是原标签。

先解释准确率，再按学生问题查看每类结果；分类正确不能直接证明模型认识恒星的物理机制。LSTM 和后面的 GPT 使用相同划分与片段宽度，但架构、参数量及学习率不同；它们的分数差不能全归于预训练。检验预训练仍看同一个 GPT 的直接训练对照。

随机森林没有删除，补充比较时仍可运行：

```bash
python -m nanoscigpt.starlight_data baseline --data out/starlight/data/data-and-splits.npz --output out/starlight/random-forest-01
```

它读取整条曲线的 1024 个数，输出 `metrics.json`、`predictions.npz` 和 `random_forest.joblib`。更换课堂主示范是为衔接时序模型，不是声称 LSTM 更优，或隐去已有的强基线。[本次发布验证](acceptance/starlight-release-20260907.md)。

## 再让 GPT 学曲线

```bash
python -m nanoscigpt.starlight --data out/starlight/data/data-and-splits.npz --output out/starlight/gpt-01 --seed 45 --head-mode flat --pretrain-steps 300 --finetune-steps 800 --cosine --threads 2
```

模型把连续 32 个亮度数值作为一小段：1024 个点变成 32 段，仍保留先后顺序。小型因果 Transformer 先根据前面的片段预测后一段，再接上三分类任务，更新模型参数。这是借鉴时间序列片段表示的 nanoGPT 教学实现，不是 TimesFM 权重、完整架构或原论文复现。

程序保留数据包原有数值，不重新归一化、寻找周期或相位折叠。对学生自己的原始光变应另行确定处理方式，不能把这份预处理基准的条件直接套过去。

这条命令还运行两份对照，便于学生问起时直接比较：

| 输出 | 实際做了什么 |
| --- | --- |
| `pretrained.pt` | 不使用类别标签，练习预测下一段数值 |
| `finetuned.pt` | 从预训练权重开始，使用标签训练 800 步 |
| `scratch.pt` | 同样的模型不预训练，直接使用标签训练 800 步 |
| `scratch_long.pt` | 不预训练，直接使用标签训练 1100 步，匹配总更新次数 |

`metrics.json` 给出实际比较；`history.json` 保存训练记录；`patch_examples.npz` 保存固定验证样本的片段预测；`predictions.npz` 保存分类概率与原标签。片段预测每一步都利用真实的已观测前文，不是脱离真实数据连续生成整条曲线。匹配更新次数不等于精确匹配计算量。

所有类别标签原本都存在。预训练时不读它们，不意味着我们另外收集了无标签数据。讨论“少量有标签”可以先用这份数据作说明；真要比较少标签条件，需要另定训练标签数量，固定选择并重新运行两种方法。现有 800 标签实验不能改称少标签实验。

## 接着研究

沿用本次 `metrics.json`、`history.json`、数据、模型设置和学生关心的类别。先看观察，再提出解释和改动；解释结果时不自动开启新训练。

讨论训练方法时，复制现有命令，改用 `train-validation-only.npz`，增加 `--validation-only`，并指定新的输出目录。它不会读取测试样本或产生测试指标。比较完再决定是否需要最终测试，不反复根据测试集成绩挑配置或种子。

`nanoscigpt.starlight` 每次从头训练各分支，不是恢复优化器继续训练。若要从 `finetuned.pt` 接着训练，需增加并验证恢复代码；若研究的是训练方法，则明确各组从相同初始化重新开始。`nanoscigpt.native_v1` 的离散序列转换器不能接这个连续片段模型。

autoresearch、v1、v2 的研究思路可以用于这个实验；原版文本模板不会自动变成光变任务。由当前 Qoder 阅读原项目相关代码、安排研究并调用这个训练入口，是教学任务适配。没有运行上游独立调度程序，就不称完整原版复现。研究计划、实际结果和稿件按当前对话逐步产生，不由这里预设赢家或结论。

## 本例可以讲到哪里

标签暂以 1、2、3 称呼。包内字段没有建立可核验的恒星名称、类别天文名称或物理时间单位对应关系；图的横轴用“采样点”，不补写波长、天数或相位。这里是预处理过的分类基准，不是原始巡天数据清洗或新天体发现。

历史试跑中，直接随机森林准确率约 94.07%；预训练后微调的三个种子平均约 92.67%，并非每个种子都优于不预训练的对照。这些是先前运行记录，不是学生本次结果。教学的重点是亲眼看到预训练、分类和比较怎样进行，不保证模型越复杂分数就越高。
