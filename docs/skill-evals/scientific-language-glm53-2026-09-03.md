# Scientific Language Skill｜GLM-5.3 CLI实测

## 测试设置

- Codex CLI：0.152.0
- 模型：`scnet/GLM-5.3`
- 模式：`read-only`、`ephemeral`、`minimal`
- 要求：只回复当前会对学生说的话，不执行命令

## 无Skill对照

同类请求在没有读取课程Skill时出现了三种不稳定行为：

1. 光谱请求一次追问数据量、仪器、波长范围、分辨率、信噪比、标签、任务、GPU和截止时间，并直接推荐CNN、自编码器等模型。
2. 天气请求直接说“先跑一个最小预训练切片，跑通后立刻放大到全量”，没有说明课程代码尚不能读取学生的数据。
3. CSV请求同时询问行列数、列名、目标和脱敏样本，并说“把你的CSV换进同一模板本地跑，步骤完全一样”。

这些回答方向不一定错误，但不适合作为90分钟课堂中的低负担带练。

## 修改前Skill

读取修改前Skill后，回答已经会区分内置样例和学生数据，但仍出现两项事实错误：

> 好，就用光谱。模型按波长顺序逐点读谱，保留连续谱线形状与相邻关系，训练时猜下一段谱形。

> 好，就做天气：模型逐格读取，保留相邻网格和时间先后关系，先学预测下一时刻。

仓库中的这两个结构化样例实际执行的是遮盖重建，不是下一段或下一时刻预测。

## 修改后复测

### 学生有自己的光谱，目标未知

> 你的是连续波长上的光谱。仓库里有同类表示的生成教学样例，但课程命令还不会读取你的光谱文件。你最后希望从光谱预测什么？

### 学生有自己的天气网格，目标未知

> 你的是天气网格数据，模型会按网格读取并保留空间关系，训练时重建被遮住的片段。但课程命令目前只读内置课程样例，不会读取你的文件。先说清：你最后想预测、生成或模拟什么？

### 学生只有不能上传的CSV

> 你的 CSV 不用上传；课程命令只读仓库内置样例，不会读取你的文件。想判断它能不能跟着课程做，先告诉我：你的 CSV 里一行代表什么？

### 学生没有数据

> 不用先准备数据，仓库自带课程样例，可以直接跑通一个小模型。你想先试哪一种数据？可以说蛋白质、DNA、分子、天气、晶体、三维结构、图像、光谱或连续物理场。告诉我你的选择，我再给你对应的运行命令。

### 学生已有蛋白质目标（加入自备FASTA入口以前）

> 你的对象是蛋白质序列，最终预测酶功能：模型逐个读氨基酸、保留顺序与序列边界；课堂命令只读内置样例，不会读你的FASTA。下一步只补一个缺口：你大约有多少条序列、典型长度多少？把这两个数发我。

### 学生有自己的蛋白质FASTA（新增入口以后）

> 能。你手里的是一万条蛋白质序列的FASTA，仓库可以直接接入自备蛋白质FASTA做小预训练，不必换回内置样例。但要注意：当前示例的下游任务头是按序列组成构造的教学分类，不是酶功能预测；要预测功能，需要另接功能标签。你这些序列带EC号或功能注释吗，还是只有序列本身？先回我这一点，我们再定预训练和功能任务头怎么做。

学生回答“只有序列”后，GLM-5.3只推进数据准备这一项：

> 你手里是约一万条蛋白质序列，FASTA加载器可直接接入；没有EC或功能标签，这轮先做预训练、跳过下游，酶功能预测要等有标签另接。请运行：`python -m nanoscigpt.domains.protein.prepare --fasta D:\data\enzymes.fasta --out_dir out/student-data/protein`。把输出发我。

### 学生有自己的SMILES表，只想先预训练

第一次复测仍追问“最后想预测、生成或模拟什么”，尽管学生已经明确只做预训练。收紧交互规则后，GLM-5.3直接给出当前一步：

> 你手里是SMILES分子串，仓库已支持，没标签就先只做预训练。请运行：`python -m nanoscigpt.domains.smiles.prepare --csv D:\data\molecules.csv --smiles-column molecule --out_dir out/student-data/smiles`，跑完把输出贴给我。

### 学生有整理好的光谱NPZ和温度标签

加入结构化NPZ入口后，用真实GLM-5.3 CLI再次测试。学生已经说明文件路径、四个数组、输入形状和回归目标，回答不再重复提问：

> 你的光谱符合NPZ合同，温度是一维回归目标。运行：`python -m nanoscigpt.prepare_structured --domain spectrum --npz D:\data\spectra.npz --out-dir out/student-data/spectrum --patch-size 8 --task-name “恒星温度回归”`。完成后发我输出。

这次模型给出的命令漏掉了命令行要求的`--sample-unit`和`--target-unit`，因此不能作为可复制的最终教学记录。Skill随后补入了完整命令示例，并要求路径、数组和目标已知时直接给出全部必需参数。

修改后再次用同一提示复测，得到：

> 你的光谱符合NPZ合同，温度是一维回归目标。运行：`python -m nanoscigpt.prepare_structured --domain spectrum --npz D:\data\spectra.npz --out-dir out/student-data/spectrum --patch-size 8 --task-name "恒星温度回归" --sample-unit "一条归一化光谱" --target-unit kelvin`。完成后发我输出。

这一命令包含解析器要求的全部参数，并且没有把未执行的操作说成已经完成。

### 学生有晶体周期图数组，但没有说明形成能单位

晶体NPZ入口加入后，第一次GLM-5.3复测直接生成了命令，并自行把单位写成`eV`：

> 你的晶体数据符合周期图NPZ合同，形成能是一维回归目标。在仓库根目录运行：`python -m nanoscigpt.prepare_structured --domain crystal --npz D:\data\crystals.npz --out-dir out/student-data/crystal --task-name "形成能回归" --target-unit eV`。完成后把输出发我。

这条命令既缺少必填的`--sample-unit`，又猜测了学生未提供的形成能单位。Skill随后明确：目标单位属于运行证据；`eV/atom`、`eV/cell`等不能混用，缺失时只追问单位，单位已知后才给出完整命令。

同一缺单位提示复测后，GLM-5.3停止生成命令，只追问当前缺口：

> 你手里的 `D:\data\crystals.npz` 已按晶体周期图合同备齐原子序数、分数坐标、掩码、晶格矩阵和 train/val_y，可以直接接入形成能回归。先确认一件事：形成能标签的单位是什么？是 eV/atom、eV/cell 还是其他？

补充“标签单位是eV/atom”后再次复测，得到可直接复制的完整准备命令：

> 你的晶体符合周期图NPZ合同，形成能按eV/atom回归。运行：`python -m nanoscigpt.prepare_structured --domain crystal --npz D:\data\crystals.npz --out-dir out/student-data/crystal --task-name “形成能回归” --sample-unit “一个周期晶胞” --target-unit “eV/atom”`。完成后发我输出。

## 当前判断

Skill现在能稳定做到：已有信息不重复问；自有数据与内置样例分开；目标未知时只追问当前缺口；结构化样例的预训练说法与实际代码一致。蛋白质与DNA的自备FASTA、学生自己的SMILES表已经可以完成数据准备和小型预训练；天气、晶体、图像、光谱、连续场和三维点集可通过明确的NPZ合同进入预训练与一维回归任务。FITS和原始仪器格式仍需先整理，不能直接运行。
