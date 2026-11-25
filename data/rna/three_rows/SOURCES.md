# 短 RNA 双链形成自由能

任务是在 37 °C、1 M NaCl 条件下预测短双链的实测形成自由能 ΔG°₃₇（kcal/mol）。训练源是 Zuber 等（2022）的补充工作簿 `Zuber_et_al_Stacking.Regressions.xlsx` 中的 `Experiment List`。按 ΔG°₃₇ = ΔH° − 310.15 K × ΔS° 计算，同一双链的多次测量取中位数。焓用 kcal/mol，熵先从 cal/(mol·K) 换算。取自开放获取文章，按 CC BY 4.0 使用。[训练来源](https://doi.org/10.1093/nar/gkac261) · [许可](https://creativecommons.org/licenses/by/4.0/)

确认源是 Sieg 等（2023）为 Table 1 公开的工作簿中 1 M NaCl 的实测列。正文表列 24 条，作者工作簿有 25 条。本次按预先指定的工作簿评分。单独剔除只见于工作簿的那一条后，第二组剩 10 条，五颗种子仍全部三行递进。作者用荧光标记的短双链测量结合等温线，与训练源的光学熔解法不同。作者代码库随附 MIT 许可，许可全文保存在同目录的 `Eco80-LICENSE.txt`。论文采用 CC BY-NC 4.0，课程材料同时注明这一条件。[实验论文](https://doi.org/10.1093/nar/gkad807) · [作者工作簿](https://github.com/JPSieg/Eco80_NNP/tree/master/Figures_Tables/Table_1_Helix_association_energies) · [论文许可](https://creativecommons.org/licenses/by-nc/4.0/)

外部原表有 25 条双链，去掉与旧表完全重复的 3 条。训练源中与外部面板同家族的 45 条也已移出。共享整条链、只差一个配对位点，或去掉两端后核心相同的双链通过连通分量归为一家。剩下的 166 条训练源双链按家族分配为监督池 38、辅助池 96、开发集 32。外部两组各 11 条，分别来自 9 和 10 个家族。两个确认组来自同一项实验。

演示使用种子 53，从监督池抽 32 个精确标签。辅助池只提供六档。同一对齐双链在三种方法中使用相同的堆叠与末端特征。`panel.csv`、`featurize.py` 与 `fixture.npz` 分别提供原始双链及分组、特征计算和可训练数组。`paired-control.json` 记录只打乱辅助档位后该种子的结果。
