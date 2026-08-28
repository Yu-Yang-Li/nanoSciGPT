# nanoSciGPT 调研笔记

日期：2026-08-28  
用途：记录整合框架的设计依据——为什么这样分层、每个领域的先例是什么。

## 一、nanoGPT 基线（karpathy/nanoGPT，MIT，62.5k stars）

master 结构已实时核验（2026-08-28）：

- `model.py`：GPT 定义（~300 行）；`train.py`：训练循环；`sample.py`：采样；`configurator.py`：config 覆盖。
- `data/shakespeare_char/prepare.py`：字符到整数直映射（stoi/itos），vocab=65，90/10 切分，uint16 存 train.bin/val.bin。
- `config/train_shakespeare_char.py`：block_size=256、n_layer=6、n_head=6、n_embd=384。

**关键发现**：nanoGPT 天然的"领域插件"接口就是 `data/<domain>/prepare.py` + `config/<domain>.py`。本框架直接继承此模式。

## 二、科学领域先例（三个已核实案例）

### prot-gpt（hrzn/prot-gpt，20 stars，无 license）

对 nanoGPT 的三处关键魔改（README 核实）：

1. 独立序列训练：context 不跨序列（蛋白序列之间独立）；
2. 变长 padding + masking：序列 pad 到 block_size，attention 屏蔽 pad；
3. 训练循环改用 PyTorch Lightning。

完整链路：PDB pdb_seqres.txt → 过滤 protein → ~10M 参数训练 → 生成 → AlphaFold Colab 折叠可视化。

### nanoGPT-DNA（diego-taquiri/nanoGPT-DNA，0 stars，无 license）

- 单文件自包含（train_gpt_dna.py 21.5KB 内含全套 GPT + DataLoader）。
- 核苷酸级 tokenization（A/T/C/G，vocab_size=5）。
- 数据：hg38 BED 区间 + FASTA（pyfaidx 按区间读取），BED 第 4 列区分 train/valid。
- 实验：85M 参数、13B tokens、2×RTX 4090 DDP；评测 DART-Eval Task 1。

### dnaGPT（ar0it/dnaGPT，0 stars，MIT）

老版 nanoGPT fork，DNA 改动集中在 dna_gpt/ 子目录。MIT 许可是亮点，但改动薄、fork 痕迹重，仅作 license 安全的参考。

## 三、规模参考（课堂讲演进，不复现）

| 项目 | 规模 | 课堂角色 |
|---|---|---|
| Evo 1/1.5（evo-design/evo，Apache-2.0） | 7B 参数，OpenGenome 300B tokens | 从核苷酸 GPT 到长上下文架构演进 |
| GenSLM（ramanathanlab/genslm，MIT） | 密码子级 MLM，Science 2023 | MLM vs CLM 目标对照 |
| HyenaDNA（HazyResearch/hyena-dna） | 长上下文 SSM | 架构演进参考 |

## 四、架构决策依据

三个先例的全部差异都落在四处：**tokenizer、数据准备、变长处理、评测**；模型结构、训练循环、采样逻辑完全不变。

所以 nanoSciGPT 的分层是：

- **core/（共享）**：GPT 模型、trainer、sampler、tokenizer 基类、dataset 基类；
- **domains/（领域专用）**：每领域必写 prepare.py（数据获取 + tokenizer 定义 + 模式声明）。

## 五、诚实边界

- prot-gpt 和 nanoGPT-DNA 都**无 license**：本仓库只参考其思想（变长 padding、BED+FASTA 加载），代码从零实现，不存在 license 污染。
- 四个领域的教学数据都是最小 fixture：UniProt 只取 500 条、ESOL 1128 条、chr21 切 35 万碱基、Shakespeare 1MB。预训练收益在这个规模不会显现，课堂上如实标注。
