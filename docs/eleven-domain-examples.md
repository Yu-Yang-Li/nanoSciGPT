# 11 个方向的 32 标签实验

每项都有一份已划分的数据、可阅读的 Nano 架构、随机森林与两种 Nano 的训练代码。三个模型用相同的 32 个监督对象训练，依次为随机森林、Nano 直接微调、Nano 预训练后微调。v8 学生演示全程固定种子 53，每项只训练一组模型，再分别预测两个确认组。五种子研究计数保存在下表最后一列。

## 在本地运行

仓库和 v8 学生包均默认运行单种子三行实验。`python -m nanoscigpt.classroom --domain rna` 单跑短 RNA 双链，`--domain all` 依次运行全部十一项。在完整仓库中可另用 `--profile classroom` 或 `--profile smoke` 运行原教学任务。[旧任务与数据](../data/README.md)

旧的 RNA 二分类仍可单独运行：`python -m nanoscigpt.classroom --domain rna --data_root data/comparisons/v7 --out_root out/rna-v7`。需要重跑旧版五种子实验时，将 `--data_root` 指向已存档的 v7 学生包中的 `data` 目录。

在项目或本地训练包根目录使用 Python 3.12。本次 Windows CPU 环境为 Python 3.12.10、PyTorch 2.12.0+cpu、NumPy 2.4.2、SciPy 1.17.0、scikit-learn 1.7.2。从空虚拟环境安装相同版本可运行以下命令。已安装这些库的环境可以直接运行，训练时全程离线。

```text
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.2 scipy==1.17.0 scikit-learn==1.7.2
python -m nanoscigpt.classroom --profile three-rows --list
python -m nanoscigpt.classroom --profile three-rows --domain protein --out_root out/my-run
```

把 `protein` 换成下表中的领域名即可单独训练。要一次运行全部十一项，使用新输出目录：

```text
python -m nanoscigpt.classroom --profile three-rows --domain all --out_root out/all-eleven
```

每个领域输出 `three_rows.csv` 和 `three_rows_receipt.json`。前者只有三行，列是种子 53 在两个确认组的成绩；后者记录输入文件 SHA-256、三种方法的单种子成绩和用时。再次运行时可选择新输出目录，或者主动加 `--overwrite`。训练函数只收到监督训练标签、辅助数据和待预测输入，确认对象的标签与参考成绩由外层评分程序读取。

## 数据集、架构与训练代码

各数据链接指向对应的 `three_rows` 目录。其中 `fixture.npz` 是可直接训练的数组，`settings.json` 保存独立开发阶段选定的步数、超参数、阈值及封存参考分数。表中数量为辅助对象数；最后一列为两个确认组的对象数。科学模拟和真实观测来源分别列明。

| 领域与数据 | 监督预测目标 | 辅助数据与预训练信号 | Nano 架构与训练实现 | 两组大小 |
|---|---|---|---|---:|
| [光谱 spectrum](../data/spectrum/three_rows) | 从模拟双温热源可见光谱预测 250/700 nm 对数通量比 | 2,400 条可见光谱配对独立的 320/700 nm 颜色测量 | [16 维 patch Transformer，拼接可见光拟合的参考量，再做回归](../nanoscigpt/tasks/three_rows/spectrum.py) | 480 / 480 |
| [蛋白质 protein](../data/protein/three_rows) | 酵母肽段的色谱保留时间 | 2,200 个其他序列只提供同类实测量的六档；[来源与加工](../data/protein/three_rows/SOURCES.md) | [共用 Nano PatchEncoder 和监督读出](../nanoscigpt/tasks/three_rows/measured_sequence.py) | 400 / 400 |
| [RNA rna](../data/rna/three_rows) | 预测短双链在 37 °C 的实测形成自由能 | 96 条其他双链提供同类实测量的六档。确认来自另一项实验，按双链家族隔离。[两份原表、许可与划分](../data/rna/three_rows/SOURCES.md) | [相邻堆叠及末端配对计数、16 维非线性编码和岭回归读出](../nanoscigpt/tasks/three_rows/rna_duplex.py) | 11 / 11 |
| [DNA dna](../data/dna/three_rows) | 短 Watson–Crick DNA 发夹的熔解温度 | 973 个其他序列只提供同类实测量的六档；[来源与加工](../data/dna/three_rows/SOURCES.md) | [共用 Nano PatchEncoder 和监督读出](../nanoscigpt/tasks/three_rows/measured_sequence.py) | 243 / 243 |
| [小分子 smiles](../data/smiles/three_rows) | 从结构预测中性或近中性化合物的辛醇/水亲脂性 | SangsterLogP 原始实测值，2,484 个其他分子只提供五档弱标签，每个种子另用 32 个精确标签；五个分区的骨架族相互独立 | [原子环境计数、Morgan 计数指纹与 Nano PatchEncoder](../nanoscigpt/tasks/three_rows/paired_numeric.py) | 480 / 480 |
| [晶体 crystal](../data/crystal/three_rows) | 从模拟晶胞组成、原子位置及晶格预测质量密度 | 640 个晶胞，随机遮蔽原子后重建晶胞描述量。沿用已确认的旧版 | [16 维周期图消息传递，节点池化，拼接晶胞描述量做回归](../nanoscigpt/tasks/three_rows/crystal.py) | 192 / 480 |
| [三维结构 structure3d](../data/structure3d/three_rows) | 从位置噪声不均匀的模拟三维轨迹估计螺距 | 3,072 对重复位置测量，估计每个位置的测量精度 | [16 维 patch Transformer 估计精度，加权几何拟合读出螺距](../nanoscigpt/tasks/three_rows/structure3d.py) | 480 / 480 |
| [天气场 weather](../data/weather/three_rows) | 从两层重叠运动云的四帧模拟图像估计紧致云的平流速率 | 同一批目标云叠加独立运动的宽云层。640 个训练场配有目标层独立光学示踪的自相关读数。学生包仅提供混合图像和训练用读数 | [多尺度光流与 16 维 patch Transformer 速度回归头](../nanoscigpt/tasks/three_rows/weather_flow.py) | 480 / 480 |
| [扩散场 field](../data/field/three_rows) | 从 298 K 的四帧浓度场、孔隙度与交联指数估计指定工作温度下的扩散系数 | 模拟单组分材料；2,400 个额外材料在两个温度上各测一批示踪粒子位移，每种子另取 32 个精确标签 | [Fourier 模态拟合、Nano PatchEncoder 和温度换算](../nanoscigpt/tasks/three_rows/thermal_diffusion.py) | 480 / 480 |
| [天文图像 image](../data/image/three_rows) | 估计 16×16 模拟拥挤星场中 3 至 22 颗注入星源的数目 | 3,072 张普通曝光配有独立模拟的高分辨率曝光总峰数，作为弱监督。每种子另用 32 个精确计数 | [多尺度峰形、天空背景扣除、16 维 patch Transformer 与计数回归头](../nanoscigpt/tasks/three_rows/crowded_image.py) | 480 / 480 |
| [文本 text](../data/text/three_rows) | 根据去掉空格与标点的小写台词行恢复英文词数 | Tiny Shakespeare 不同原文区段，2,048 行提供 12,891 处稀疏词边界标注；每种子另用 32 行精确词数。两个确认区段各 480 行 | [一层 Nano GPT、局部字符卷积、相邻字母查表与词数读出](../nanoscigpt/tasks/three_rows/boundary_count.py) | 480 / 480 |

每项森林均有 200 棵树。Nano 的直接微调从随机初始化的编码器出发；预训练后微调先使用本行列出的辅助信号，再读取与直接微调相同的 32 个监督标签。各领域的读出、训练步数和开发阈值保存在对应的配置文件中。序列任务按已计算的数值描述训练，数据包中包括训练需要的特征、辅助目标和留出标签。无需在课堂机器上重新计算 Umap 轨道、比对 RNA 锚点或安装 RDKit。

## 种子 53 演示成绩与五种子研究结果

两列演示分数均为 Windows 实际训练对应的参考值，顺序为随机森林、直接微调、预训练后微调。所有任务用平均绝对误差（MAE），越低越好。最右列独立列出开发后封存的五种子研究计数，分子为同一种子两步均严格改善的次数。

| 领域 | 指标 | 种子 53 第一组 | 种子 53 第二组 | 五种子研究完整递进（两组） |
|---|---|---|---|---|
| 光谱 | MAE，dex | 0.10364 → 0.04745 → 0.03940 | 0.09995 → 0.04682 → 0.03705 | 4/5、4/5 |
| 蛋白质 | MAE，分钟 | 7.49521 → 5.86003 → 3.28848 | 7.65339 → 5.37139 → 3.25855 | 5/5、5/5 |
| RNA | MAE，kcal/mol | 3.55624 → 3.32641 → 3.15464 | 3.85476 → 3.52879 → 3.08571 | 5/5、5/5 |
| DNA | MAE，°C | 3.06061 → 2.09456 → 1.91425 | 2.97537 → 2.31464 → 2.00111 | 4/5、4/5 |
| 小分子 | MAE，logP | 1.16833 → 1.15762 → 0.82720 | 1.07817 → 1.05805 → 0.72559 | 4/5、4/5 |
| 晶体 | MAE，质量/体积 | 0.29374 → 0.05485 → 0.05014 | 0.29582 → 0.05779 → 0.04942 | 4/5、4/5 |
| 三维结构 | MAE，长度/圈 | 0.01947 → 0.00569 → 0.00531 | 0.01911 → 0.00561 → 0.00540 | 5/5、5/5 |
| 天气场 | MAE，坐标单位/帧 | 0.02012 → 0.01485 → 0.00919 | 0.01952 → 0.01495 → 0.00907 | 5/5、5/5 |
| 扩散场 | MAE，模拟长度²/模拟时间 | 0.00675 → 0.00338 → 0.00238 | 0.00799 → 0.00356 → 0.00269 | 5/5、5/5 |
| 天文图像 | MAE，个/图 | 1.66667 → 1.57917 → 1.30625 | 1.69375 → 1.58542 → 1.34375 | 5/5、5/5 |
| 文本 | MAE，词/行 | 1.59583 → 1.11458 → 0.95417 | 1.29792 → 1.04792 → 0.93542 | 5/5、5/5 |

这里复跑既有留出组，验证数据与代码能否重现各自锁定版本。蛋白质与 DNA 的两种 Nano 在每个方向使用相同的监督步数、核回归头和正则系数。数据包中没有预训练或微调后的权重。

## 读取数据

```python
import json
from pathlib import Path
import numpy as np

folder = Path("data/rna/three_rows")
settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
with np.load(folder / "fixture.npz", allow_pickle=False) as data:
    print(data.files)
    print(data["53_x"].shape, data["53_y"].shape)
    print(data["aux_x"].shape, data["group1_y"].shape)
print(settings["seeds"]["53"])
```

数组名以种子、`aux`、`group1` 或 `group2` 开头。种子前缀对应 32 个有标签训练对象，`aux` 保存归一化输入、无标签观测或配对辅助目标，两个 `group` 前缀保存确认对象。螺距的成对辅助观测用 `aux_a` 和 `aux_b`；小分子辅助池的 `aux_y` 是 0 至 4 的溶质亲脂性档位。拥挤星场的混合像素保存在 `*_obs`，高分辨率峰数保存在 `aux_A`。参考分数只在训练和预测完成后参与校验。

文本任务的输入用 1–26 表示英文字母，0 表示行尾补齐。训练用的 `aux_marked` 以 −1 表示未标注字母间隙，0/1 分别表示不分词/分词。模型将各间隙分词概率求和加一，再用监督行校准。学生包只提供抽样位置的边界信息。

扩散场的 `obs` 是四帧 64 点标量场，`material` 为模拟孔隙度与交联指数，`temp` 为工作温度（K）。辅助池 `msd` 两列是 298 K 与 318 K 各自的二维粒子均方位移。森林和两种 Nano 均可使用由室温场拟合的系数及同一温度关系；只有预训练阶段使用配对位移。生成器见 [simulate.py](../data/field/three_rows/simulate.py)。两个确认组各有 480 份同分布的新模拟材料，数据没有经过真实材料标定。

CPU 参考分数保留 2×10⁻⁶ 的绝对误差校验。crystal、image、spectrum、structure3d、text 在 Linux x86-64 使用单列的逐种子参考分数，其余使用共同参考。两套参考均对应相同训练数据、权重初始化种子、模型与超参数，另在干净解压目录从头训练验收。表中列出 Windows 参考值。

小分子三行使用同一数值矩阵，包含常规结构描述量、96 维原子局部环境计数和 128 维 Morgan 计数指纹。精确 logP、数据表的计算值、作者调整值与 RDKit MolLogP 都未作为输入。两个 Nano 的监督阶段在开发集共同选择了冻结编码器、只拟合核回归头。打乱对照和外部模型成绩见研究记录。 [数据出处与许可](../data/smiles/three_rows/SOURCES.md)

RNA 的预训练在 96 条其他双链的实测量粗分档上进行。五种子下，真实配对优于打乱配对的次数为第一组 4/5、第二组 5/5。两组共 22 条，都来自同一外部实验；这些结果的适用范围是这批短双链及其测量条件。v7 的 5S rRNA 与 tRNA 分类在两个确认组均为 1/5，作为旧任务保留。
