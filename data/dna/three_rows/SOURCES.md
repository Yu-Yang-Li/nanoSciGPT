# 短 Watson–Crick DNA 发夹的熔解温度

Ke et al., Array Melt (2025)。[数据与许可](https://doi.org/10.5281/zenodo.15299579) · [研究论文](https://doi.org/10.1038/s41467-025-60455-4) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

选用原始拟合表中 WatsonCrick 系列的两态发夹，标签为实测荧光曲线拟合的 Tm。保留四核苷酸环及完整互补干，合并重复序列，按去除外层碱基对的干序列及反向互补对称性划分家族。粗分档边界为 35、45、55、65、75 °C。

本包对原数据做了筛选、切分、分档及序列特征提取。panel.csv 中辅助对象只提供档位。准确复跑使用 fixture.npz，特征可由 featurize.py 从表中重算。paired-control.json 包含按种子的打乱对照；随机置换辅助对象的档位，其余训练设置相同。
