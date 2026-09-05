# 上游项目与nanoSciGPT适配边界

更新时间：2026-08-30

这份文件只回答代码库层面的问题：哪些开源项目影响了当前实现，哪些内容被实际采用，哪些只作为技术参照。

## B线补充：2026-09-05

B线现以原版karpathy/autoresearch、SakanaAI/AI-Scientist、SakanaAI/AI-Scientist-v2为入口。`nanoscigpt/upstream.py`获取固定commit，保留原仓库及许可文件，并生成可复核的教学改动。固定版本、原版启动方式和当前验证范围以[原项目接入说明](../skills/ai-scientist-research-loop/references/native-projects.md)为准。

本仓库`autoresearch/`里的规则迭代、离线稿件整理、预设分支比较继续保留，但不再充当原版复现。v1的原nanoGPT模板CPU基线已运行；模型API驱动的完整研究流程尚待验收。

### v1 的兼容模型接口：显式教学适配

固定版本的 v1 会拒绝 `scnet/GLM-5.3` 这样的模型名，且两个评阅入口另有写死的模型。现在可以在已准备的、无相关源码改动的 v1 副本上显式配置：

```powershell
python -m nanoscigpt.upstream configure-api v1 --root out/upstream --model scnet/GLM-5.3 --review-model gpt-6-astra
```

模型名是配置示例，不绑定 Skill；必须选择自己接口实际提供的模型。命令只修改该副本的 `ai_scientist/llm.py` 和 `launch_scientist.py`，生成 `teaching_api_changes.diff` 与 `teaching_api_setup.json`。不会安装依赖、启动研究、保存密钥或改变系统配置。它只接受固定版本；相关文件已有修改时拒绝覆盖。重复相同配置会核对源码摘要，更换配置请使用另一份副本。

- 想法与反思调用：将所选模型名原样传给 OpenAI-compatible 接口。
- 实验改码与写作：按 Aider 的 `openai/<实际模型名>` 路由，辅助模型显式采用同一模型。
- 初次评阅与修改后评阅：都使用指定的 `--review-model`，不再暗中使用原来写死的模型。
- 非 GPT 模型的多回答请求保留原版逐次调用方式，不要求服务支持一次返回多个回答。

启动研究的进程还需自行配置 `OPENAI_BASE_URL` 与 `OPENAI_API_BASE` 为同一经过核查的服务地址，以及相应 `OPENAI_API_KEY`。CLI 登录不代表每台学生电脑都有兼容服务；本机测试用的 loopback 地址和占位字符串不是可公开分发的通用 API 配置。不要把真实密钥写入命令示例、研究笔记或 Git。

配置后的 `launch_scientist.py` 仍是原项目启动入口，研究提示词、想法生成、实验调度与写作流程没有被替换。**配置成功仅表示路由已写入，不等于完整研究通过。** 请求重试、总调用预算、实际改码及写作仍需分别验收；当前不建议把整个原版启动命令直接作为已保证低负载的学生入口。

### v1 的失败记录与完成状态

固定版本会删除非零退出或超时实验的 `run_N/`，且绘图重试耗尽后仍可能返回成功。教师可在独立原版副本中应用小范围修补：

```powershell
python -m nanoscigpt.upstream configure-failures v1 --root out/upstream
```

仅修改 `ai_scientist/perform_experiments.py`，另存 `teaching_failure_changes.diff` 和 `teaching_failure_setup.json`。它与 `configure-api` 操作不同文件，可以分别采用；均不启动研究或安装依赖，也不覆盖学生已有源码修改。

- 失败或超时时，将部分结果移入当次实验目录内的 `failed_runs/run_N_<唯一后缀>/artifacts/`，连同此次 `experiment.py` 和完整 stderr 记录。不是删除；重复失败产生不同目录。
- 成功实验仍使用原来的 `run_N/final_info.json` 结果接口。
- 没有完成任何新实验，或绘图重试耗尽后仍失败，实验函数返回 False，由原版启动程序停止当前 idea；不会把模型说的 `ALL_COMPLETED` 单独当成运行证据。

真实子进程的非零退出、超时、重复失败、正常结果以及绘图失败分支已测试；远程 coder 在这些自动测试中用受控回复替代，因此这不是整场 Agent 研究验收。补丁不限制外部 API 重试和整场预算，也不提供 v2 恢复机制。配置收据的状态仍为 `configured_not_run`。

## A线原有适配

| 上游项目 | 当前状态与许可 | nanoSciGPT实际采用 | 没有采用 |
|---|---|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT；上游已在2025年说明项目较旧，并推荐新项目nanochat | 小型decoder-only Transformer、下一token训练、`prepare/train/sample`教学分工 | GPU集群配置、GPT-2权重加载、完整训练栈 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | MIT；当前更活跃 | 只参考“清楚的一条命令和可复查训练产物”这一工程方向 | 不复制其面向GPU的大规模训练流程 |
| [hrzn/prot-gpt](https://github.com/hrzn/prot-gpt) | 仓库未见明确许可证 | 独立蛋白序列、变长padding、padding mask的设计思想 | 不复制源码；其GPU训练预算不作为课堂默认值 |
| [diego-taquiri/nanoGPT-DNA](https://github.com/diego-taquiri/nanoGPT-DNA) | WIP；仓库未见明确许可证 | 单碱基词表、FASTA/BED式数据边界的设计思想 | 不复制源码，不复现其双RTX 4090训练预算 |
| [DeepChem ESOL](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/delaney_datasets.py) | DeepChem代码为MIT；数据源为Delaney ESOL | 使用仓库已分发的1128条SMILES与实测水溶解度列，演示预训练后接回归任务 | 不引入DeepChem、RDKit或其完整模型栈；课堂结果不作为分子性质基准 |
| [EarthPT](https://github.com/aspiaspace/EarthPT) | MIT；连续时序nanoGPT改造 | 用来确定连续观测需要patch投影与数值任务头 | 不复制其源码、卫星数据或大模型训练流程 |
| [AstroPT](https://github.com/Smith42/astroPT) | 天文图像patch自回归项目 | 用来确定图像不能只换字符词表，需要patch投影和连续输出 | 不复制源码、权重或观测数据 |
| [GPTCast](https://github.com/DSIP-FBK/GPTCast) | 天气雷达离散化与GPT式预测项目 | 用来说明天气网格必须保留空间结构 | 不引入VQGAN与业务天气数据 |
| [CGCNN](https://github.com/txie-93/cgcnn) | 晶体图卷积项目 | 用来确定周期晶胞需要节点、邻接关系和周期距离 | 当前教学图网络不是CGCNN复现 |
| [SpectralGPT](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) | 光谱patch掩码建模项目 | 用来确定波长轴与局部光谱片段需要显式保留 | 不复制模型或遥感数据 |

## 代码许可处理

- 当前核心实现沿用nanoGPT的最小GPT组织方式；仓库保持MIT许可，并在`THIRD_PARTY_NOTICES.md`保留上游版权与许可文本。
- 对没有明确许可证的prot-gpt、nanoGPT-DNA，只采用公开README中描述的技术思想，代码在本仓库重新实现。
- 连续场、图像、图结构和三维几何不冒充成“只换tokenizer即可支持”。当前分别使用数值patch、周期图消息传递和距离不变量，并有独立CPU样例与测试。

## 课堂选择规则

`python -m nanoscigpt.classroom --list`只列出已经随仓库提供数据、通过CPU实跑的领域。当前是：

- `text`：语言模型热身；
- `protein`：蛋白质序列预训练＋组成属性教学分类；
- `dna`：DNA序列预训练＋GC含量教学分类；
- `smiles`：SMILES预训练＋ESOL水溶解度教学回归。
- `weather`：时空patch重建＋移动速度回归；
- `crystal`：周期图原子恢复＋密度代理回归；
- `structure3d`：三维距离重建＋螺距回归；
- `image`：图像patch重建＋源计数；
- `spectrum`：波长patch重建＋温度回归；
- `field`：时空patch重建＋扩散系数回归。

新增六域全部使用确定性教学数据；跑通表示与任务流程不等于复现对应前沿项目。
