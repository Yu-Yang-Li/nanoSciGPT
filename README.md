# nanoSciGPT

“科学模型专题实训（二）”的CPU课堂仓库。学生从一条命令开始，依次看到：

```text
内置数据 → 科学对象表示 → 小型预训练 → 具体任务 → 模型V0 → V0到V1迭代记录
```

当前正式课堂材料是 [16页大纲](docs/current/course-outline-16p.md)、[16页页面文字](docs/current/slide-copy-16p.md) 和 [16页讲稿](docs/current/speaker-script-16p.md)。课程先用LAMOST光谱参数估计走一遍有标签科学数据的标准流程，再进入nanoSciGPT与AI Scientist。换成自己的数据前，请先看[自定义科学数据接入说明](docs/current/custom-domain-guide.md)。其他材料的当前/档案身份见[资料导航](docs/README.md)。

## 六个课堂 Skill

学生按照同一份结果逐步向后推进：

1. `nanoscigpt-research-baseline-builder`：有标签科学数据的第一份基线；
2. `nanogpt-pretraining`：文本预训练与微调；
3. `nanoscigpt-scientific-language`：把同一思路迁移到科学数据；
4. `autoresearch-model-iteration`：完成一次V0到V1的同口径比较；
5. `ai-scientist-v1-workflow`：把单路线实验整理成可复查材料；
6. `ai-scientist-v2-tree-search`：比较多条研究路线并保留取舍依据。

各Skill的使用边界和入口见[六个课堂Skill索引](skills/README.md)。

## 先跑起来

需要Python 3.10或更高版本。默认只用CPU，仓库已经带好课堂数据，不要求学生临时下载。

```bash
python -m pip install -e .
nanoscigpt-doctor

# 有标签课程基线：LAMOST、普通表格或单条数值时序
nanoscigpt-baseline --case lamost --out_root out/baseline
nanoscigpt-baseline --csv <数据.csv> --target <目标列> --task regression --out_root out/baseline
nanoscigpt-baseline --series-csv <时序.csv> --value-column <数值列> --time-column <时间列> --out_root out/baseline

# 已整理成训练/验证数组的天气、晶体、图像、光谱、连续场和三维点集
nanoscigpt-prepare-structured --help

# 查看当前真正能运行的十个选择
python -m nanoscigpt.classroom --list

# 先看一种数据在课程代码里实际怎样表示和训练；这一步不训练模型
python -m nanoscigpt.classroom --describe spectrum

# 任选一个；默认classroom配置约几十秒
python -m nanoscigpt.classroom --domain protein
python -m nanoscigpt.classroom --domain weather
python -m nanoscigpt.classroom --domain crystal
python -m nanoscigpt.classroom --domain spectrum

# 只检查环境和代码链路时用更短配置
python -m nanoscigpt.classroom --domain protein --profile smoke
```

安装后也可以使用：

```bash
nanoscigpt-classroom --domain protein
```

`nanoscigpt-doctor`只检查当前Python、四个依赖、十类数据和六个Skill，不修改环境。检查未通过时，先解决它列出的缺项，不直接开始训练。

`--describe`返回模型处理单位、保留关系、实际预训练目标、下游任务和样例身份。它同时明确标记`student_data_loaded=false`，避免把内置教学结果写到学生数据上。

### 把六个Skill安装到Codex

Windows PowerShell：

```powershell
pwsh -NoProfile -File scripts/install_skills.ps1 -Destination (Join-Path $env:USERPROFILE ".codex\skills")
```

Linux或macOS：

```bash
bash scripts/install_skills.sh "$HOME/.codex/skills"
```

安装脚本只复制[课程索引](skills/README.md)列出的六个Skill。目标位置已有同名Skill时会停止，不会静默覆盖；安装完成后在新的Codex会话中使用。模型和数据仍从本仓库根目录运行。

## 学生可以选择什么

`--list`只展示已经随仓库提供数据、通过CPU端到端测试的领域。

| 选择 | 内置数据 | 课堂运行内容 | 具体任务 |
|---|---|---|---|
| `text` | tiny Shakespeare | 字符预训练与采样 | 标点密度教学分类，并更新预训练参数 |
| `protein` | 500条UniProtKB reviewed记录 | 氨基酸序列预训练 | 蛋白质组成属性教学分类 |
| `dna` | hg38 chr21教学切片 | 单碱基序列预训练 | GC含量教学分类 |
| `smiles` | 1128条Delaney ESOL记录 | SMILES字符预训练 | 实测水溶解度教学回归 |
| `weather` | 96组移动标量场 | 时空patch掩码重建 | 移动速度教学回归 |
| `crystal` | 96个周期晶胞 | 周期图上的原子掩码恢复 | 晶胞质量密度代理回归 |
| `structure3d` | 96组三维螺旋坐标 | 距离行掩码重建 | 螺距教学回归 |
| `image` | 96张天文点源小图 | 图像patch掩码重建 | 源数量教学回归 |
| `spectrum` | 96条黑体连续谱 | 波长patch掩码重建 | 温度教学回归 |
| `field` | 96组一维扩散场 | 时空patch掩码重建 | 扩散系数教学回归 |

protein和DNA的标签是透明的课堂标签，只用来演示“预训练后怎样接任务头”；SMILES使用ESOL中的实测水溶解度列。六个结构化样例的标签来自确定性生成器中记录的参数。它们都不是论文benchmark，也不用于比较模型优劣。

十个入口并不共用同一种表示：离散序列使用小型因果Transformer；天气、图像、光谱和连续场使用数值patch；晶体使用周期图消息传递；三维结构使用刚体变换不变的距离表示。

## 默认CPU配置

| 配置 | 训练步数 | 模型 | 用途 |
|---|---:|---|---|
| `smoke` | 2 | 1层、16维 | 安装后快速检查 |
| `classroom`（默认） | 30 | 2层、64维 | 课堂演示与学生跟练 |

表中训练步数对应四个离散序列领域。六类其他科学数据的`smoke`配置运行2步预训练和2步具体任务，默认`classroom`配置各运行20步。两类配置都固定使用CPU。

每次运行单独写到：

```text
out/classroom/<domain>/
├── model/
│   ├── ckpt.pt
│   └── train_log.json
├── downstream/
│   └── downstream_result.json
└── run_report.json
```

`run_report.json`记录实际命令、设备、数据预检、耗时和产物路径，便于学生把运行过程交回来检查。

## 数据和离线运行

十类课程数据都在 [data/](data/) 中并纳入版本管理。来源、用途和必要文件见 [data/manifest.json](data/manifest.json) 与 [data/README.md](data/README.md)。

默认课堂命令只读取这些文件，不访问网络。只有教师主动运行各领域的`prepare.py`且原始文件不存在时，text、protein和SMILES才会访问公开来源；DNA不会静默生成合成数据。

## 代码结构

```text
nanoscigpt/
├── classroom.py             # 学生统一入口、数据预检、CPU配置、运行报告
├── evidence_pack.py         # 把已有运行、评价和停止记录整理成Markdown
├── core/
│   ├── gpt.py               # 小型因果Transformer
│   ├── trainer.py           # 四个离散序列域共享训练循环
│   ├── sampler.py           # 采样
│   ├── tokenizer.py         # 字符级tokenizer
│   └── dataset.py           # 流式数据与独立变长序列
├── domains/
│   ├── text/
│   ├── protein/
│   ├── dna/
│   └── smiles/
├── scientific/
│   ├── adapters.py         # 数值patch、周期距离、三维距离表示
│   └── models.py           # patch Transformer与周期图网络
└── tasks/
    ├── downstream_demo.py   # 四种离散序列的CPU具体任务
    └── structured_demo.py   # 六种结构化对象的预训练与任务

autoresearch/                # 从一次结果继续安排下一轮实验
```

## 与开源项目的关系

- 小型GPT核心和`prepare/train/sample`教学分工来自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的MIT代码传统。nanoGPT上游已说明项目较旧；本仓库仍采用它，是因为课堂要看清最小结构，而不是复现当前大规模训练栈。
- protein的独立序列和padding思路参考 [prot-gpt](https://github.com/hrzn/prot-gpt)；DNA数据边界参考 [nanoGPT-DNA](https://github.com/diego-taquiri/nanoGPT-DNA)。这两个仓库未见明确许可证，因此只参考公开技术描述，没有复制其源码。
- SMILES标签来自DeepChem分发的Delaney ESOL数据；本仓库不依赖DeepChem或RDKit。
- EarthPT、AstroPT、GPTCast、CGCNN等项目用于确定数值patch、图结构和连续观测的适配边界；本仓库没有复制其源码或训练数据，只实现独立的小型CPU教学版。

逐项说明见 [上游项目与适配边界](docs/upstream-adaptation.md)。

## 手动运行底层步骤

课堂外需要拆开看时：

```bash
# 预训练
python -m nanoscigpt.core.trainer --domain protein --out_dir out/manual/protein --max_iters 30 --eval_interval 15 --eval_iters 5 --block_size 64 --batch_size 8 --n_layer 2 --n_head 2 --n_embd 64

# 接一个具体任务
python -m nanoscigpt.tasks.downstream_demo --domain protein --ckpt out/manual/protein/ckpt.pt --out_dir out/manual/protein-task

# 查看结构化对象怎样完成表示、预训练和具体任务
python -m nanoscigpt.tasks.structured_demo --domain weather --out_dir out/manual/weather

# 先读取A线的模型V0，只生成迭代设置
python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --plan_only

# 确认设置后，运行V1并与V0比较
python -m autoresearch.experiment --domain protein --baseline_run out/classroom/protein/run_report.json --out_root out/autoresearch --auto_approve
```

第一条命令不会启动训练，只写`iteration_spec.json`。第二条命令读取同一份V0，跳过重复训练V0，写出`comparison.json`和同目录下的研究状态。已有候选结果时应使用新的`--out_root`，不覆盖旧实验。

## 把已有结果整理成证据包

完成一次比较后运行：

```bash
python -m nanoscigpt.evidence_pack --run-report out/classroom/protein/run_report.json --comparison out/autoresearch/protein/comparison.json --state out/autoresearch/protein/research_state.json --output out/evidence-packs/protein.md
```

这个入口只读取已有JSON，不重新训练。只有运行报告时可以省略比较结果和研究状态，输出会标为“已运行”；找不到运行报告时不会创建证据包。

## 测试

```bash
python -m pytest -q
python -m pytest tests/test_classroom.py -q
```

`tests/test_classroom.py`会真实启动十个CPU smoke run，并用课堂配置检查每个领域的训练趋势、模型文件、表示或生成样例、运行报告和具体任务结果。

## 课堂边界

- loss下降只说明训练过程执行完成。
- 课堂生成的蛋白质、DNA和SMILES样例没有经过结构、功能或化学有效性检查。
- 教学分类标签不代替真实生物学标签；ESOL小回归也不代替正式分子性质评测。
- 六个结构化样例是确定性方程或几何原型，不是观测数据、实验晶体库或PDB结构库。
- 任何新增领域仍必须同时具备表示层、初始数据、CPU命令和端到端测试，才进入学生可选列表。

## 许可

本仓库采用MIT许可。nanoGPT原始版权告知、第三方技术参考和数据使用说明见[第三方告知](THIRD_PARTY_NOTICES.md)；逐项适配边界见[上游项目说明](docs/upstream-adaptation.md)与[数据清单](data/manifest.json)。
