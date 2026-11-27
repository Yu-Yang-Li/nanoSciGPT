# 酵母肽段的色谱保留时间

Bouwmeester et al., DeepLC (2021)。[数据与许可](https://doi.org/10.5281/zenodo.4542884) · [研究论文](https://doi.org/10.1038/s41592-021-01301-5) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

保留 7–32 残基的未修饰肽段，合并 I/L 等价表示并对重复记录取中位数。按编辑相似度 80% 去近邻。精确标签由秒换算为分钟，粗分档边界为 10、20、30、40、50 分钟。数据来自同一 60 分钟梯度实验汇编；切分单位是肽段。

本包对原数据做了筛选、切分、分档及序列特征提取。panel.csv 中辅助对象只提供档位。准确复跑使用 fixture.npz，特征可由 featurize.py 从表中重算。paired-control.json 包含按种子的打乱对照；随机置换辅助对象的档位，其余训练设置相同。
