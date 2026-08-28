# 全学科 GPT-like 科学模型调研与 nanoSciGPT 架构启示

研究快照：2026-08-28  
用途：回答两个问题：哪些科学模型确实属于 GPT-like；nanoSciGPT 应怎样用一套最小框架讲清不同科学对象的“语言化”。

## 一、先固定口径

本调研不按项目名字里是否带 `GPT` 判断，而按训练机制分四级：

| 等级 | 可操作定义 | 对 nanoSciGPT 的意义 |
|---|---|---|
| G0：nanoGPT 直接改造 | 仓库明确源自、重度参考或直接复用 nanoGPT | 可以做代码血缘案例 |
| G1：GPT-2 同构 | decoder-only、causal attention、next-token prediction；实现可来自 GPT-2、GPT-Neo 或自写 decoder | 可以共用离散语言模型教学骨架 |
| G2：宽口径 GPT-like | 保留因果自回归，但输入是连续 patch、VQ token、图序列或混合离散—连续对象，损失不一定是词表交叉熵 | 需要增加领域 tokenizer、输入投影和输出头 |
| N：相邻但非 GPT-like | masked modeling、encoder-only、encoder-decoder、diffusion、contrastive，或只是用通用 LLM 处理学科文本 | 用作路线对照，不能写成 GPT 魔改 |

同时区分“GPT-like 模型”和“科学基座模型”：前者只要求架构与训练范式相符；后者还必须证明预训练能力能跨任务、数据集或条件迁移。能生成一个对象，不自动等于基座模型。

## 二、结论先行

1. **全学科覆盖不能按院系名称穷举，应按科学数据形态闭合。** 当前已经覆盖离散序列、排序集合、图、结构化科学文件、点云、图像、光谱、时空场、连续波形、临床事件和数学表达式。新增学科大多只是这些表示的组合。
2. **nanoGPT 的直接科学改造不止 DNA 和蛋白质。** 已核实 `prot-gpt`、`nanoGPT-DNA`、`ar0it/dnaGPT`、`EarthPT`、`AstroPT` 五个直接或明确源自 nanoGPT 的项目；后两者证明 nanoGPT 可以从离散字符扩展到连续地球观测时间序列和天文图像 patch。
3. **“只改 domain”只对离散科学字符串成立。** DNA、RNA、蛋白质、SMILES 可以共用 token embedding + categorical LM head；图像、天气、波形和点云还需要 patch/VQ/MLP tokenizer、连续回归头或解码器。
4. **GPT-like 科学模型的真正技术史不是换数据集，而是表示方式升级：**

   `科学字符串 → 条件与任务 token → 图/结构序列化 → 连续观测离散化或 patch 化 → 多对象统一科学语法`

5. **调研已在“表示类型”层面达到饱和。** 后续高价值工作不是继续加项目名，而是核查核心仓库的可运行接口、数据许可和最小复现成本。

## 三、全学科证据矩阵

| 学科/对象 | 怎样变成模型输入 | GPT-like训练目标 | 公认或高价值代表 | 证据与成熟度 | 归类 |
|---|---|---|---|---|---|
| 文本基线 | 字符或BPE token | 下一token | [nanoGPT](https://github.com/karpathy/nanoGPT) | MIT；教学母体 | G0基线 |
| 蛋白质序列 | 氨基酸或BPE；序列边界与padding mask | 下一残基/token | [prot-gpt](https://github.com/hrzn/prot-gpt)、[ProtGPT2](https://www.nature.com/articles/s41467-022-32007-7)、[ProGen](https://www.nature.com/articles/s41587-022-01618-2) | ProtGPT2为Nature Communications正式论文；ProGen含功能实验；prot-gpt仅小型代码先例且无明确许可证 | G0/G1 |
| DNA与基因组 | 单碱基、k-mer或BPE；正反链与长上下文 | 下一碱基/token；可加入类别和数值token | [nanoGPT-DNA](https://github.com/diego-taquiri/nanoGPT-DNA)、[DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT)、[Evo](https://www.nature.com/articles/s41586-025-09749-7)、[regLM](https://genentech.github.io/regLM/readme.html) | nanoGPT-DNA为WIP；DNAGPT为预印本+代码；Evo为自回归Hyena/Transformer混合体而非GPT-2复刻 | G0/G1/G2 |
| RNA设计 | RNA碱基序列；条件中加入dot-bracket二级结构 | 给定目标结构生成下一碱基 | [RNA-Design-LM](https://github.com/KuNyaa/RNA-Design-LM) | 2026预印本+代码；加入热力学评价和约束解码，尚非长期公认基座 | G1前沿 |
| 单细胞/转录组 | 将基因按表达量排序，或将基因与表达值组成token序列 | 下一基因排名或迭代恢复表达值 | [tGPT](https://doi.org/10.1016/j.isci.2023.106536)、[scGPT](https://www.nature.com/articles/s41592-024-02201-0) | tGPT用GPT2LMHeadModel并在2230万细胞上做自回归；scGPT为GPT启发的生成式masked解码，非标准CLM | G1；scGPT为G2 |
| 分子字符串 | SMILES、SELFIES及BPE；可加入性质、骨架、靶点token | 下一化学token；条件生成 | [SMILES-GPT](https://github.com/sanjaradylov/smiles-gpt)、[MolGPT](https://doi.org/10.1021/acs.jcim.1c00600)、[ChemGPT](https://www.cambridge.org/engage/chemrxiv/article-details/627bddd544bdd532395fb4b5) | MolGPT为JCIM正式论文+MIT代码；ChemGPT基于GPT-Neo且研究已发表于Nature Machine Intelligence | G1强锚点 |
| 化学反应与三维分子 | 反应物/产物字符串，或直接序列化XYZ/PDB文件 | 下一token或条件序列生成 | [XYZ/CIF/PDB语言模型](https://arxiv.org/abs/2305.05708)、[ProtTeX](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00585) | 证明无需改主干即可生成3D科学文件；几何有效性仍须外部评价器 | G1/G2 |
| 一般图、分子图、蛋白相互作用图 | 欧拉路径、节点—边序列或可逆图token | 下一图token；或scheduled masked token | [GraphGPT](https://proceedings.mlr.press/v267/zhao25r.html)、[G2PT](https://icml.cc/virtual/2025/poster/45870) | GraphGPT为ICML 2025正式论文+代码，覆盖节点、边、图级迁移；序列化顺序和图同构是关键边界 | G2强锚点 |
| 晶体与材料 | CIF/POSCAR文本；元素、晶格、坐标、空间群和性质token | 下一CIF/token；条件结构生成 | [CrystaLLM](https://www.nature.com/articles/s41467-024-54639-7)、[AtomGPT](https://github.com/atomgptlab/atomgpt) | CrystaLLM为Nature Communications正式论文、MIT代码与数据；结构生成仍需对称性、稳定性和DFT筛选 | G1/G2强锚点 |
| 粒子与高能物理 | VQ-VAE把变长粒子点云压成离散jet token | 下一jet token；预训练后迁移到jet tagging | [OmniJet-α](https://doi.org/10.1088/2632-2153/ad66ad) | MLST 2024正式论文+公开代码；提供生成→分类的真实跨任务迁移 | G2强锚点 |
| 天文观测 | 星系图像切成连续patch；可串联图像与SED | 下一patch回归；也支持MAE对照 | [AstroPT](https://github.com/Smith42/astroPT) | 明确从nanoGPT改造；ICML 2024 AI4Science workshop；当前仓库为AGPL-3.0，早期论文所述MIT不能替代当前许可核查 | G0/G2 |
| 地球观测时间序列 | 多光谱反射率按时序切成连续token/patch | 下一观测patch回归 | [EarthPT](https://github.com/aspiaspace/EarthPT) | 明确从nanoGPT改造，MIT；NeurIPS 2023 Climate Change AI workshop；700M模型为论文级而非课堂复现规模 | G0/G2 |
| 天气雷达与时空场 | VQGAN将降水雷达图压成离散codebook token | GPT-2式时空下一token预测 | [GPTCast](https://gmd.copernicus.org/articles/18/5351/2025/) | GMD 2025同行评审；代码、数据、权重齐全且MIT；这是“连续场先离散化再语言建模”的最佳锚点 | G2强锚点 |
| 遥感高光谱 | 3D空间—光谱patch | 多目标重建/MAE | [SpectralGPT](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) | TPAMI 2024正式论文，但实质是MAE式重建，不是causal next-token；名称容易误导 | N，对照 |
| 地震与连续波形 | Z/N/E三分量波形切成时间patch并连续投影 | 下一波形patch回归 | [SeismoGPT](https://github.com/wesmail/seismogpt)、[CBS-GPT](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.084017) | SeismoGPT为2026预印本+MIT代码；CBS-GPT为PRD 2024正式论文；均更接近连续自回归而非离散token LM | G2 |
| 临床纵向记录 | 诊断、用药、检查、时间间隔等事件token | 下一临床事件/token；患者轨迹生成 | [CEHR-GPT](https://arxiv.org/abs/2509.03643)、[TransformEHR](https://www.nature.com/articles/s41467-023-43715-z) | CEHR-GPT仍以预印本为主；TransformEHR为正式encoder-decoder生成模型，作为相邻路线 | G1前沿/N |
| 神经科学/脑电 | 以电极为单位，将EEG切成时序信号token | 下一信号预测 | [EEGPT](https://arxiv.org/abs/2410.19779) | 自回归预训练、跨设备和多任务迁移，但当前主要是预印本；[BrainLM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/029ce70401321de3808b3ac39e1ab167-Abstract-Conference.html)是masked模型，不归入严格GPT-like | G2前沿；BrainLM为N |
| 数学与符号方程 | 运算符、变量、常数、前缀/中缀表达式token | 条件生成下一符号 | [SymbolicGPT](https://github.com/mojivalipour/symbolicgpt)、[EqGPT](https://pmc.ncbi.nlm.nih.gov/articles/PMC12638968/) | SymbolicGPT有MIT代码；方程必须由符号代入或数值残差验证，字符串完全匹配不是充分评价 | G1/G2 |
| 通用科学时间序列 | 滑动窗口或patch映射到embedding；可量化也可连续投影 | 下一patch/未来窗口 | [GPT4TS](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)、[TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) | GPT4TS复用冻结GPT-2；TimesFM是decoder-only但并非nanoGPT；代表跨天气、能源、交通、传感器的共享表示层 | G2 |
| 跨自然科学统一语法 | 把多类科学对象及接触/约束关系编码进共享词表 | 同一语法空间内统一下一token预测 | [LOGOS](https://arxiv.org/abs/2606.16905) | 2026预印本与开放资源声明；可作为前沿收束，不能当作长期公认结论 | G2前沿观察 |

## 四、已核实的 nanoGPT 直接科学改造

| 项目 | 代码关系 | 科学输入 | 相对nanoGPT的关键变化 | 许可/证据边界 |
|---|---|---|---|---|
| [prot-gpt](https://github.com/hrzn/prot-gpt) | README明确受nanoGPT启发 | 独立蛋白序列 | 变长padding、padding mask、不跨序列采样、Lightning训练 | 未见明确license；只参考思想 |
| [nanoGPT-DNA](https://github.com/diego-taquiri/nanoGPT-DNA) | README称模型与训练循环 heavily based on nanoGPT | hg38碱基流/BED区间 | A/T/C/G词表、基因组区间加载、DART-Eval | WIP且未见明确license；不直接复制代码 |
| [ar0it/dnaGPT](https://github.com/ar0it/dnaGPT) | 明确为nanoGPT应用于DNA | DNA序列 | 主要替换数据和配置 | MIT；改动较薄 |
| [EarthPT](https://github.com/aspiaspace/EarthPT) | README明确“began its life as nanoGPT” | 地球观测多光谱时间序列 | MLP tokenizer、连续值输入、regressive loss | MIT；NeurIPS workshop，不等于成熟全球地球基座 |
| [AstroPT](https://github.com/Smith42/astroPT) | README明确“began its life as nanoGPT” | 星系图像patch、后续扩展至图像+SED | MLP tokenizer、regressive loss、AR/MAE可切换、linear probe/LoRA | 当前仓库AGPL-3.0；ICML workshop；许可随版本发生过变化 |

排除项：`nanogpt-seis`训练的是地震学论文和网页文本，而不是地震波形本身；它是领域文本LLM，不属于“科学对象语言化”的主证据。真正对应波形对象的是SeismoGPT。

## 五、从全学科案例抽出的四种“科学语言化”机制

### 1. 固定离散词表

适用：DNA、RNA、蛋白质、SMILES、数学符号。

```text
raw object -> vocabulary tokenizer -> integer token IDs
           -> causal GPT -> categorical LM head -> cross entropy
```

优点是最接近nanoGPT；主要问题是边界、长上下文、语法合法性和科学评价不能只靠perplexity。

### 2. 结构化文件或图序列化

适用：CIF、PDB、XYZ、反应字符串、一般图。

```text
graph / 3D structure -> reversible serialization
                     -> node/edge/coordinate tokens
                     -> next-token prediction
                     -> parser + scientific validator
```

关键不只是tokenizer，还要处理排列不变性、数值精度、语法约束和物理有效性。

### 3. 学习型离散tokenizer

适用：天气雷达、粒子点云、复杂图像或连续场。

```text
continuous field / point cloud -> VQ-VAE or VQGAN codebook
                               -> discrete tokens
                               -> GPT next-token prediction
                               -> domain decoder
```

GPTCast和OmniJet-α说明，tokenizer重构误差会成为整个基座模型能力的上限；必须单独评价token fidelity。

### 4. 连续patch自回归

适用：天文图像、地球观测、波形、一般时间序列。

```text
continuous observations -> patch + MLP projection
                        -> causal decoder
                        -> continuous regression head
```

EarthPT和AstroPT虽然源自nanoGPT，但不再是“词表分类”。因此它们不能只通过新增一个`domains/<name>/prepare.py`并入当前离散token实现。

## 六、对 nanoSciGPT 仓库的直接架构结论

当前仓库的文本、DNA、蛋白质和SMILES四域属于第一类。若要真正承载全学科教学，不应把所有差异继续塞进`domains/`，而应拆成以下组合关系：

```text
raw scientific object
        |
        v
domain loader
        |
        v
representation adapter
  - discrete_vocab
  - structured_serializer
  - learned_codebook
  - continuous_patch
        |
        v
causal backbone
        |
        v
prediction head
  - categorical_next_token
  - continuous_next_patch
  - conditional_generation
        |
        v
scientific evaluator / constraint checker
```

因此后续代码设计应满足：

1. `domain`只负责科学数据读取和元数据边界；
2. `representation`负责对象怎样变成序列；
3. `objective/head`决定交叉熵还是连续回归；
4. `evaluator`负责合法性、结构、物理量或外部任务评价；
5. 主干可以共享，但不能假定所有输入都是整数token、所有输出都是词表概率。

这比“所有领域只改domain”更准确，也与实际的EarthPT、AstroPT、GPTCast、GraphGPT和OmniJet-α一致。

## 七、最适合课程的代表链，而不是项目点名墙

90分钟课程不应讲完全部项目。全学科调研用于证明技术链闭合，主讲只需要以下六个台阶：

1. **nanoGPT**：文本字符如何成为token，理解因果预训练；
2. **ProtGPT2 / MolGPT**：不改GPT基本主干，把科学字符串当语言；
3. **tGPT**：没有天然顺序的转录组怎样通过排序被构造成语言；
4. **GraphGPT / CrystaLLM**：图和三维结构怎样通过可逆序列化进入next-token训练；
5. **EarthPT / AstroPT**：直接从nanoGPT出发，但连续科学观测需要MLP tokenizer和回归头；
6. **GPTCast / OmniJet-α**：先学习科学tokenizer，再把时空场和粒子点云变成离散语言，并检验迁移。

最后用LOGOS作前沿观察：研究正在从“每个学科一套tokenizer”尝试走向共享科学语法，但目前证据不足以宣布“一个GPT统一所有自然科学”已经成立。

## 八、容易误收的项目与停止边界

以下情形不纳入严格GPT-like主表：

- 名称带GPT但只做领域文本问答，如OceanGPT、各类GeoGPT、BioGPT；它们没有对原始海洋、地质或生物数据做科学token预训练。
- Transformer模型但使用masked/encoder目标，如DNABERT、ESM、BrainLM、SpectralGPT、ClimaX、AstroCLIP。
- 自回归滚动预测但没有大规模预训练与跨任务迁移证据的专用预测器；它可以是causal model，但未必是foundation model。
- diffusion或flow模型，如MatterGen、GenCast；它们是重要生成路线，但不属于GPT-like。
- 只在名称或摘要中类比ChatGPT，却没有公开模型结构、预训练目标和科学对象表示的项目。

## 九、研究饱和度与剩余工作

按科学数据形态审计，当前不存在会改变总体架构判断的明显空白：

- 离散生物序列：已覆盖；
- 化学字符串与反应：已覆盖；
- 排序集合与单细胞：已覆盖；
- 图与三维结构：已覆盖；
- 图像、光谱和连续观测：已覆盖；
- 时空场与一般时间序列：已覆盖；
- 粒子点云、临床事件、脑电和数学符号：已覆盖。

仍需后续实施而非继续广搜的事项：

1. 对拟进入代码库的上游项目逐个检查当前license、可下载数据和最小运行成本；
2. 用同一小数据预算复现四种representation adapter，而不是复现全部大模型；
3. 为每类对象建立科学评价器，避免把训练loss下降等同于科学能力；
4. 将“正式同行评审、workshop、预印本、代码WIP”在PPT中使用不同标签。

达到这些边界后，新增同类型项目只记录为替换位，不再扩大一般候选池。
