# data/eleven 的来源

这里是 `nanoscigpt.tasks.eleven` 和 `notebooks/eleven_fields.ipynb` 用到的六个真实数据方向。每个文件包含：12,800 个无标签对象、监督训练池、留出组 A 的输入，以及所用下游任务的标签。token 以 uint8 编号存储，`symbols` 给出编号对应的符号。光谱、天气场、扩散场、天文图像、三维结构五个方向是模拟数据，运行时按固定随机流生成，不存文件。

| 文件 | 来源 | 处理 |
|---|---|---|
| protein.npz | Bouwmeester 等，DeepLC 酵母 60 分钟梯度色谱数据。[数据，CC BY 4.0](https://doi.org/10.5281/zenodo.4542884) · [论文](https://doi.org/10.1038/s41592-021-01301-5) | 7 到 32 个残基的未修饰肽段，I 并入 L，重复记录取保留时间中位数 |
| dna.npz | 预训练和留出窗口：酿酒酵母参考基因组 [sacCer3](https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/)，按染色体分开。下游发夹：Ke 等 Array Melt。[数据，CC BY 4.0](https://doi.org/10.5281/zenodo.15299579) · [论文](https://doi.org/10.1038/s41467-025-60455-4) | 基因组切成 64 nt 窗口；发夹取 Watson–Crick 两态发夹的实测熔解温度，重复序列取中位数 |
| rna.npz | [Rfam 15.1](https://rfam.org/downloads) 种子比对中 RF01960、RF02543 两个核糖体 RNA 家族，CC0 | 去掉比对空位后切成 32 nt 片段，按 accession 分开；标签为片段内重叠三核苷酸的经验熵 |
| smiles.npz | Ramakrishnan 等 QM9。[原始记录](https://doi.org/10.6084/m9.figshare.978904) | SMILES 按词法单元切分，按重原子组成分开；标签由 RDKit 2025.09.6 计算 sp³ 杂化碳比例 |
| crystal.npz | Castelli 等的 ABX₃ 钙钛矿计算数据，取 Matbench 收录版本。[Matbench](https://matbench.materialsproject.org/Leaderboards/00matbench_v0.1_matbench_perovskites/) | 保留晶格、元素和分数坐标；标签为晶胞体积的立方根 |
| text.npz | Tiny Shakespeare 原文 | 小写、只留字母和空格、去重，取 24 到 64 个字符的行；下游输入去掉空格，标签为词数 |
