# nanoSciGPT

“科学模型专题实训（二）：如何构建领域基座模型与 AI Scientist 系统”的配套代码、数据和教学 Skill。

**科学语言**是这个项目的出发点。语言模型读的是一串 token，学的是根据前面的 token 预测下一个。光谱、氨基酸序列、晶胞、天气图、星场图像也可以写成一串 token：光谱每 4 个相邻波长的通量算一个 token，蛋白质每个氨基酸算一个，晶体先放晶格、再放每个原子。写成 token 之后，十一种科学观测都交给同一个两层的 nanoGPT，用同一个目标预训练：预测下一个 token。方向不同，只换把观测切成 token 的编码器，以及最后读出一个数的线性头。

| 方向 | 一个 token 是什么 | 一条样本 | 下游任务 |
|---|---|---|---|
| 光谱 | 连续 4 个波长采样点的通量 | 双温模拟谱 64 个采样点，切成 16 个 token | 紫外/可见通量比 |
| 蛋白质 | 1 个氨基酸 | 酵母肽段，最多 32 个 token，词表 19 项 | 色谱保留时间（实测） |
| DNA | 1 个碱基 | 酿酒酵母基因组 64 nt 窗口，64 个 token | 发夹熔解温度（实测） |
| RNA | 1 个碱基 | Rfam 核糖体 RNA 片段，32 个 token | 三核苷酸熵 |
| 小分子 | 1 个 SMILES 词法单元，如 `Cl`、`[C@@H]` | QM9 分子，最多 22 个 token，词表 27 项 | sp³ 杂化碳比例 |
| 晶体 | 晶格矩阵占 1 个，之后每个原子 1 个（元素 + 分数坐标） | ABX₃ 钙钛矿晶胞，6 个 token | 赝立方晶格常数 |
| 三维结构 | 原子间距离矩阵中的 1 个 4×4 块 | 16 个点的模拟链，16 块中遮去含首尾距的 2 块，剩 14 个 token | 首尾距 |
| 天气场 | 同一 4×4 区域的连续四帧 | 4 帧 16×16 的模拟云图，16 个 token | 平均云团亮度 |
| 扩散场 | 4 个相邻位置的连续四帧 | 4 帧、每帧 64 个位置的模拟浓度场，16 个 token | 扩散系数 |
| 天文图像 | 4×4 像素块 | 16×16 的模拟星场，16 个 token | 星像半高全宽 |
| 文本 | 1 个字母或空格 | 一行英文，最多 62 个 token | 去空格后的词数 |

符号类 token 查词表得到向量，数值类 token 经过一个线性层投影，晶体的原子 token 由元素向量加坐标投影组成。[编码器代码](nanoscigpt/scientific/encoders.py) · [主干](nanoscigpt/scientific/scigpt.py)

## 预训练数据越多，下游越好

每个方向用 200、800、3,200、12,800 个无标签对象各预训练一次，五个随机种子。然后从这批检查点出发，只用 32 个标签训练下游读出，在两组未参与训练的留出样本上测误差。11 个方向的下游误差都随无标签数据量三段下降（实线），预训练损失也一起下降（虚线）。

![十一个方向的预训练损失与下游误差](docs/pretraining-scale/shared-checkpoints/curves.png)

6 个方向的口径完全统一：冻结主干，只训练线性头，每档固定 1,600 次预训练更新，所以曲线只反映数据量的变化。另外 5 个方向保留了其他做法。光谱和扩散场每个对象平均训练 32 遍，数据越多训练步数越多；晶体、文本和 DNA 用全参数微调。DNA 换成冻结读出后也能逐档下降，但 12,800 档比不预训练差约 5%，所以保留全参数微调，这时比不预训练好 21%。比较弱的几处：

- **文本**：下游误差和只猜均值差不多。
- **RNA**：只比只猜均值好约 10%。
- **DNA**：预训练损失在最后一档基本持平。
- **晶体**：下游任务晶格常数几乎就写在第一个 token（晶格矩阵）里，测的是模型能不能把这个数读准。

[逐方向数字、口径说明和全部留出结果](docs/pretraining-scale/shared-checkpoints/README.md)

## 怎么装、怎么跑

**Notebook（推荐）。** 打开 [notebooks/eleven_fields.ipynb](notebooks/eleven_fields.ipynb)，在第一格把 `FIELD` 改成想看的方向（默认 `field`，即扩散场），然后从上到下运行。第一格会自己安装缺少的依赖，在魔搭 Notebook、本机 Jupyter 都能用，只需要 CPU。后面四格依次是：打印一条样本被切成的 token，用 200、800、3,200 个对象预训练，用 32 个标签训练下游读出，把预训练损失和下游误差画在一张图上。在仓库外单独打开 Notebook 时，第一格会先克隆仓库。

第一格的用时取决于网络。在一台 24 核 Windows 电脑上，从只装了 Jupyter 内核的 Python 3.11 环境开始，第一格从 PyTorch 官方源装 CPU 版 torch 2.15.0、NumPy、scikit-learn 和 Matplotlib，约用 10.5 分钟，几乎都花在下载 torch 上。魔搭 Notebook 的镜像通常自带 torch，这一步会跳过。依赖装好后，11 个方向各跑一遍都没有报错，整本 Notebook 的用时如下（三个方向同时运行，每个用 2 个线程）。默认方向是扩散场。

| 方向 | 整本用时 | 种子 53 的下游误差（200 / 800 / 3,200） | 只猜均值 | 三档逐档下降 |
|---|---:|---|---:|:---:|
| 扩散场（默认） | 25 秒 | 0.130 / 0.124 / 0.094 | 0.170 | 是 |
| 光谱 | 17 秒 | 0.112 / 0.094 / 0.062 | 0.330 | 是 |
| 蛋白质 | 57 秒 | 13.20 / 13.95 / 8.02 | 10.84 | 否 |
| DNA | 189 秒 | 5.26 / 5.05 / 4.15 | 7.03 | 是 |
| RNA | 56 秒 | 0.310 / 0.254 / 0.169 | 0.174 | 是 |
| 小分子 | 46 秒 | 0.200 / 0.147 / 0.092 | 0.236 | 是 |
| 晶体 | 31 秒 | 0.067 / 0.078 / 0.048 | 0.224 | 否 |
| 三维结构 | 39 秒 | 0.9997 / 0.8652 / 0.8647 | 2.950 | 是 |
| 天气场 | 36 秒 | 0.150 / 0.168 / 0.161 | 0.166 | 否 |
| 天文图像 | 38 秒 | 0.708 / 0.567 / 0.431 | 0.625 | 是 |
| 文本 | 103 秒 | 1.295 / 1.392 / 1.272 | 1.243 | 否 |

单个种子、单组留出、只到 3,200 档时，曲线会有起伏，上图的五种子平均才是结论依据。在 Windows 上，种子 53 的每个数都和五种子实验里同一种子的记录一致；Linux 上浮点计算顺序不同，预训练损失差 2% 以内，单种子下游误差可能差到 20% 左右。

**命令行。** 需要 Python 3.10 及以上和 Git。

```bash
git clone https://github.com/Yu-Yang-Li/nanoSciGPT.git
cd nanoSciGPT
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
.venv/bin/python -m nanoscigpt.tasks.eleven --field field
```

Windows 把 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。在本机打开 Notebook 需要先装 Jupyter：`.venv/bin/python -m pip install notebook`，再运行 `.venv/bin/python -m jupyter notebook`。不打开界面、直接整本执行可以用 `.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/eleven_fields.ipynb --output-dir out`，执行后的 Notebook 存在 `out/eleven_fields.ipynb`。最后一条命令和 Notebook 做的事一样：先打印 token，再输出每档的预训练损失和下游误差。用 `--sizes 200,800,3200,12800` 可以加到最大档，`--seed` 可以换种子。Windows 上种子 53 的结果和五种子实验里的同一种子逐项一致。照这一节在空目录首次安装的耗时和两个平台的结果，见 [首次使用检查](docs/first-run-2026-11-26.md)；从 PyTorch 官方源下载 torch 在当时的网络下要 27 到 36 分钟。

## 更多

- [五种子完整实验与留出结果](docs/pretraining-scale/shared-checkpoints/README.md)，含读出方式、预算的选择规则和没有通过的任务
- [只看预训练损失的四档总表与离线复现包](docs/pretraining-scale/data-scaling/README.md)
- [扩散场上的自动研究](docs/autoresearch-field.md)：DSK 每轮提一个改动，和同预算随机搜索比较，各跑 3 次；第 2 版合并等价配置、开发集用 5 个种子
- [课堂提问](docs/classroom-prompts.md)
- [三个教学 Skill](skills/README.md)：配合编程 Agent 做基线、预训练和研究迭代
- [早期课堂与验收记录](docs/history.md)：2026 年 9 月的光变课堂、三行对照和旧验收

## 仓库结构

```text
notebooks/eleven_fields.ipynb   十一个方向的主入口
nanoscigpt/
  core/                         nanoGPT 主干和训练代码
  scientific/                   编码器（符号、数值块、晶胞）和 SciGPT
  tasks/eleven.py               十一个方向的数据、预训练和下游读出
data/eleven/                    六个真实数据方向的 token 小表；模拟方向运行时生成
docs/                           实验结果、课程资料和历史记录
skills/                         三个教学 Skill
tests/                          回归测试
```

## 许可与致谢

代码采用 [MIT 许可](LICENSE)。主干改编自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

真实数据的来源如下，各自许可仍然适用：

- 酵母肽段保留时间：DeepLC 数据集，[CC BY 4.0](https://doi.org/10.5281/zenodo.4542884)
- 核糖体 RNA 片段：[Rfam 15.1](https://rfam.org/downloads)，CC0
- DNA 窗口：酿酒酵母参考基因组 [sacCer3](https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/)
- 小分子：[QM9](https://doi.org/10.6084/m9.figshare.978904)
- 晶体：[Matbench 钙钛矿](https://matbench.materialsproject.org/Leaderboards/00matbench_v0.1_matbench_perovskites/)
- 文本：Tiny Shakespeare

DNA 发夹熔解温度的出处见 [data/eleven/SOURCES.md](data/eleven/SOURCES.md)。
