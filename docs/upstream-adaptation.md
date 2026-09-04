# 上游项目与nanoSciGPT适配边界

更新时间：2026-08-30

这份文件只回答代码库层面的问题：哪些开源项目影响了当前实现，哪些内容被实际采用，哪些只作为技术参照。

| 上游项目 | 当前状态与许可 | nanoSciGPT实际采用 | 没有采用 |
|---|---|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT；上游已在2025年说明项目较旧，并推荐新项目nanochat | 小型decoder-only Transformer、下一token训练、`prepare/train/sample`教学分工 | GPU集群配置、GPT-2权重加载、完整训练栈 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | MIT；当前更活跃 | 只参考“清楚的一条命令和可复查训练产物”这一工程方向 | 不复制其面向GPU的大规模训练流程 |
| [hrzn/prot-gpt](https://github.com/hrzn/prot-gpt) | 仓库未见明确许可证 | 独立蛋白序列、变长padding、padding mask的设计思想 | 不复制源码；其GPU训练预算不作为课堂默认值 |
| [diego-taquiri/nanoGPT-DNA](https://github.com/diego-taquiri/nanoGPT-DNA) | WIP；仓库未见明确许可证 | 单碱基词表、FASTA/BED式数据边界的设计思想 | 不复制源码，不复现其双RTX 4090训练预算 |
| [DeepChem ESOL](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/delaney_datasets.py) | DeepChem代码为MIT；数据源为Delaney ESOL | 使用仓库已分发的1128条SMILES与实测水溶解度列，演示预训练后接回归任务 | 不引入DeepChem、RDKit或其完整模型栈；课堂结果不作为分子性质基准 |
| [EarthPT](https://github.com/aspiaspace/EarthPT) | MIT；连续时序nanoGPT改造 | 用来确定连续观测需要patch投影与数值任务头 | 不复制其源码、卫星数据或大模型训练流程 |
| [AstroPT](https://github.com/Smith42/astroPT) | 天文图像patch自回归项目 | 用来确定图像不能只换字符词表，需要patch投影和连续输出 | 不复制源码、权重或观测数据 |
| [GPTCast](https://github.com/DSIP-FBK/GPTCast) | 天气雷达离散化与GPT式预测项目 | 用来说明天气网格必须保留空间结构 | 不引入VQGAN与业务天气数据 |
| [CGCNN](https://github.com/txie-93/cgcnn) | 晶体图卷积项目 | 用来确定周期晶胞需要节点、邻接关系和周期距离 | 当前教学图网络不是CGCNN复现 |
| [SpectralGPT](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) | 光谱patch掩码建模项目 | 用来确定波长轴与局部光谱片段需要显式保留 | 不复制模型或遥感数据 |

## 代码许可处理

- 当前核心实现沿用nanoGPT的最小GPT组织方式；仓库保持MIT许可，并在`THIRD_PARTY_NOTICES.md`保留上游版权与许可文本。
- 对没有明确许可证的prot-gpt、nanoGPT-DNA，只采用公开README中描述的技术思想，代码在本仓库重新实现。
- 连续场、图像、图结构和三维几何不冒充成“只换tokenizer即可支持”。当前分别使用数值patch、周期图消息传递和距离不变量，并有独立CPU样例与测试。

## 课堂选择规则

`python -m nanoscigpt.classroom --list`只列出已经随仓库提供数据、通过CPU实跑的领域。当前是：

- `text`：语言模型热身；
- `protein`：蛋白质序列预训练＋组成属性教学分类；
- `dna`：DNA序列预训练＋GC含量教学分类；
- `smiles`：SMILES预训练＋ESOL水溶解度教学回归。
- `weather`：时空patch重建＋移动速度回归；
- `crystal`：周期图原子恢复＋密度代理回归；
- `structure3d`：三维距离重建＋螺距回归；
- `image`：图像patch重建＋源计数；
- `spectrum`：波长patch重建＋温度回归；
- `field`：时空patch重建＋扩散系数回归。

新增六域全部使用确定性教学数据；跑通表示与任务流程不等于复现对应前沿项目。
