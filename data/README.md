# 内置课堂数据

仓库直接携带十份nanoSciGPT小数据和两份监督学习开场数据，默认运行不访问网络。来源、用途和所需文件见[manifest.json](manifest.json)，使用与引用说明见[第三方告知](../THIRD_PARTY_NOTICES.md)。

十类数据的真实 smoke 运行摘要保存在[预生成课堂结果](precomputed_results/README.md)。现场运行失败时可以用它继续讲结果格式，但必须说明这是备用结果，不能当作当前会话已经运行的证据。

| 目录 | 内置内容 | 课堂用途 | 边界 |
|---|---|---|---|
| `text/` | tiny Shakespeare字符流 | 先看一次最小语言预训练 | 只作语言模型热身 |
| `protein/` | 500条UniProtKB reviewed FASTA及处理结果 | 蛋白质序列预训练；组成属性教学分类 | 分类标签由序列组成生成，不是生物学benchmark |
| `dna/` | hg38 chr21教学切片及处理结果 | DNA序列预训练；GC含量教学分类 | 只用A/C/G/T，不处理变异和调控注释 |
| `smiles/` | 1128条Delaney ESOL记录及处理结果 | SMILES预训练；实测水溶解度回归 | 不检查生成分子的化学有效性，不作性质SOTA比较 |
| `weather/` | 移动Gaussian标量场 | 时空patch重建；速度回归 | 不是再分析或数值天气预报数据 |
| `crystal/` | 周期晶胞几何原型 | 原子掩码恢复；密度代理回归 | 不是DFT数据或材料稳定性评价 |
| `structure3d/` | 刚体变换后的三维螺旋 | 距离重建；螺距回归 | 不是PDB结构或折叠任务 |
| `image/` | Gaussian天文点源小图 | 图像patch重建；源计数 | 不是望远镜观测或测光流程 |
| `spectrum/` | 黑体连续谱与合成吸收线 | 波长patch重建；温度回归 | 不含仪器响应和真实恒星标注 |
| `field/` | Fourier扩散场 | 时空patch重建；扩散系数回归 | 不是高保真PDE模拟benchmark |
| `course/lamost_atlas_a_teff_2000.csv` | 2000条ATLAS-A直接观测光谱整理后的128波段特征 | 首页演示恒星有效温度回归 | 随机留出只说明同一数据来源内的插值 |
| `course/palmer_penguins_morphology.csv` | Palmer Penguins的342条完整形态记录 | 首页演示科学问题如何转成表格分类 | 随机划分不证明跨岛屿、跨年份泛化 |

若要重做离散序列处理，可运行各领域的`prepare.py`。仓库内原始文件存在时不会下载；只有原始文件缺失时，text、protein和SMILES脚本才会访问各自的公开地址。DNA默认要求本地FASTA，不会静默生成合成数据。

六个结构化样例可离线重建：

```bash
python scripts/build_structured_fixtures.py
```

固定随机种子、生成方程和任务参数都写在脚本与各域`meta.json`中。
