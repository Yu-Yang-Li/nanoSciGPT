# 原项目接入说明

## 源码与教学设置

在nanoSciGPT仓库根目录运行`python -m nanoscigpt.upstream prepare <project>`。它获取固定版本源码到`out/upstream/<project>/`，保存`teaching_setup.json`和`teaching_changes.diff`，不安装依赖、不调用模型API、不训练。已有实验不重置；要另做一次准备，换`--root`。

| project | 原项目 | 固定commit | 允许的教学改动 |
|---|---|---|---|
| autoresearch | https://github.com/karpathy/autoresearch | 228791fb499afffb54b46200aca536f79142f117 | 四层模型、较小batch、256上下文、每次60秒训练、较小固定评测集；基线后最多两轮自主改动 |
| v1 | https://github.com/SakanaAI/AI-Scientist | 1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb | 复制原版nanoGPT模板；两层64维、batch 8、64上下文、30轮设定、一个随机种子、单文本数据；保留想法、代码实验和写作流程 |
| v2 | https://github.com/SakanaAI/AI-Scientist-v2 | 96bd51617cfdbb494a9fc283af00fe090edfae48 | 单worker、单种子、每阶段最多3次、单次执行180秒；保留原版四阶段和研究树 |

这些是待研究的低负载起点，不保证任何学生电脑都适用，也不与原论文预算等价。单次执行时间不是整个研究的时间/API费用上限；v1多次改码、v2四阶段、文献检索、写作与评阅还会产生额外调用。开始完整研究前，和学生约定整场时间、API费用及停止方式。

## 环境

准备实际运行时，先运行`python -m nanoscigpt.upstream doctor`。读所选上游README，在独立虚拟环境中安装它的依赖；不要直接覆盖nanoSciGPT教学环境。仅讨论学生的研究方案，不需要检查教师这台电脑。原项目完整流程主要面向Linux，Windows优先使用WSL或准备好的Linux机器；v1的小模板基线已单独验证可以在Windows CPU运行。

- autoresearch使用CUDA和专用计算依赖。执行`uv sync`前核对上游Python版本及当前GPU支持。`prepare.py --num-shards 1 --download-workers 1`仍会联网下载数据，不是离线课程子集；先了解下载体积。
- v1实验模板训练需torch/numpy，绘图还需matplotlib。只测模板时，在教学环境安装`python -m pip install -e ".[native-v1-template]"`即可补上绘图依赖；这不等于安装了完整研究环境。完整研究还需Aider、检索API、模型API及LaTeX工具。选`--model`并不自动改变全部审稿调用；固定版本的部分评阅仍使用`gpt-4o-2024-05-13`。
- v2的编程、运行反馈、视觉反馈、总结和写作分工有不同模型设置。检查`bfts_config.yaml`和源码中的实际路由；文本模型不一定支持图像反馈。GLM-5.3不能未经实测直接填进去宣称全面兼容。

只核对密钥是否存在和服务是否可用，不在对话、日志或Git中保存密钥。CLI登录不能替代原项目要求的API配置。实验在本地执行，仍可能把代码、日志、数据片段和稿件发给模型服务；私有材料先核对允许发送的范围，不以“本地运行”替代这项确认。

接入参数以实际服务的文档和返回为准，不从模型名字推断兼容性。例如，当前SCNet接口未为GLM列出关闭思考支持，实测额外传入`thinking`报400；经本机代理时这个字段还可能被丢弃。短回答成功不代表Aider长改码可用。遇到`finish_reason=length`、空正文或网关超时，先保存记录并说明未完成，不以继续提高额度作为默认处理。文档与实际对照见[接口核查](../../../docs/acceptance/provider-boundary-2026-09-05/README.md)。

## autoresearch：由编程Agent执行原版研究循环

进入生成的`out/upstream/autoresearch/`，读原版`program.md`与`TEACHING.md`。准备好数据并运行一次`uv run train.py`作为新基线；其后由Agent根据实际结果修改代码和执行实验。模型结构、优化器和训练策略都可成为研究内容；数据与评价保持约定不变。

教学停止条件覆盖原版无限循环要求。只在这个独立checkout中实验，每次撤回前保存补丁或提交，避免丢失失败证据。不要在nanoSciGPT主仓库照搬原版重置分支的指令。

## v1：实际执行，而不是汇总模板

准备完整研究之前，课程提供两项可选、显式记录的修补。在nanoSciGPT根目录运行：

```powershell
python -m nanoscigpt.upstream configure-failures v1 --root out/upstream
python -m nanoscigpt.upstream configure-api v1 --root out/upstream --model <接口实际提供的研究模型名> --review-model <接口实际提供的评阅模型名>
```

第一项保留失败实验及报错，并修正无新实验或绘图失败仍返回成功的情况；第二项供OpenAI-compatible服务使用，将研究与两个评阅入口显式路由到所选模型。各自保存补丁与配置收据，拒绝覆盖学生已有源码修改。均不安装依赖或启动研究，也不控制整场研究的调用费用。模型接口还需正确配置进程环境；详细变量、验证范围和限制见[适配说明](../../../docs/upstream-adaptation.md)。使用上游本来支持的模型时，不必为了教学统一接口而改路由。

准备后，在原版checkout中：

```powershell
cd templates/nanoSciGPT_teaching
python experiment.py --out_dir run_0
cd ../..
python launch_scientist.py --experiment nanoSciGPT_teaching --num-ideas 1 --parallel 0 --model <已核实可用的原版模型名>
```

基线输出在模板的`run_0/`，完整研究在原版`results/`。原模板会从`../../data/shakespeare_char/`读取数据，因此基线命令应从模板目录执行。课程准备器写入与当前文本词表一致的元数据；没有做跨架构权重转换。

### 继续A线的同一个序列模型

`nanoscigpt.native_v1`可以把本仓库四种因果序列GPT接到原版v1。它不是通用模型转换器，模型仍采用相同的GPT结构；复制权重时仅去掉原版高效注意力实现不保存的固定掩码缓冲区，保留所有可训练参数。已用原版GPT检查过相同输入的预测一致性。

```powershell
# 在nanoSciGPT仓库根目录
python -m nanoscigpt.native_v1 --ckpt out/classroom/protein/model/ckpt.pt --name course_protein
# 在生成的原版模板目录
cd out/upstream/v1/templates/course_protein
python experiment.py --out_dir run_0
cd ../..
python launch_scientist.py --experiment course_protein --num-ideas 1 --parallel 0 --model <已核实可用的原版模型名>
```

保存的`course_bridge.json`记录输入权重hash、数据路径和预算。文本/DNA使用流式数据；蛋白质/SMILES独立采样，短序列的填充位置不参与损失，不跨样本拼接。每次实验加载相同模型权重，重新建立优化器，不声称恢复了A线优化器状态。若改变模型结构导致权重无法严格加载，应另建并说明新的基线，不能忽略不匹配参数。

`course_protein`可换成新的模板名；已有目录不会被覆盖。这个入口支持CPU教学实验。它不负责将天气、晶体、图像或掩码模型改成GPT，也未对这些对象完成原版v1模板验收。

### 继续分类或回归，而不是换回序列预测

学生刚做过微调时，读取其`downstream_result.json`中的`task_checkpoint`，用这份文件替换上面的`--ckpt`；自有数据加上`--data_root <已准备的数据目录>`。导出器会识别任务头，并将相同样本、标签、标准化数值和初始模型复制到模板目录。不要把`model/ckpt.pt`误当成最新的微调模型。

这种情况下，`native_gpt.py`是原项目的完整、未改动GPT源码；`experiment.py`是课程分类/回归模板，`plot.py`绘制实际训练记录。原版研究Agent仍按其正常接口调用这两个文件。模板独立运行，不需要从研究目录反向导入nanoSciGPT。`task_setup.json`、`task_data.npz`和`initial_model.pt`在同一组比较中保持固定；研究改动写在`experiment.py`，失败结果也保留。

先执行`python experiment.py --out_dir run_0`及`python plot.py`。`run_0/final_info.json`里的初始指标应与微调报告一致；`run_0/checkpoint.pt`可回到课程微调入口继续训练。继续时沿用原报告的抽样上限，避免无意改变验证样本。旧文件缺少领域、任务或数据来源信息时，回到原预训练模型另做一次微调，不猜测这些信息。

这项教学适配只保留原版GPT类和研究流程，不把自编的监督训练循环称为原版训练循环。没有执行`launch_scientist.py`就不算完成了Agent研究。

核对生成的新想法及检索记录、实际运行过的实验、代码改动、图表与稿件。若API或论文编译失败，报告已完成的环节，保留报错，不用本仓库的离线`draft.md`冒充原版生成稿件。

## v2：实际搜索，而不是预先规定的两条路线

读并按学生的问题填写`teaching_topic.md`，写清数据绝对路径、目标和计算范围。该文件是研究主题起点，不包含预写的实验结论。在原版checkout执行：

```powershell
python ai_scientist/perform_ideation_temp_free.py --workshop-file teaching_topic.md --max-num-generations 1 --num-reflections 2 --model <已核实可用的原版模型名>
python launch_scientist_bfts.py --load_ideas teaching_topic.json --idea_idx 0 --writeup-retries 1 --num_cite_rounds 3
```

第二条命令前确认第一条实际生成的JSON路径和想法内容。由原版实验管理Agent决定实验节点和改进方式，保留`logs/`和`workspaces/`的真实记录。少量搜索不能代表完整预算下的表现；v2也不保证一定比v1好。

## 当前验证范围

2026-09-05：三个固定版本已获取并成功应用教学设置；v1原nanoGPT模板的CPU基线已运行。原版GPT与课程GPT的相同输入预测一致性检查通过，另接续实际训练的蛋白质权重完成了原版实验。记录见仓库`docs/acceptance/native-protein-continuation-2026-09-05/`。这不是Agent研究全流程。

autoresearch自主改码循环、v1的API研究全流程、v2研究树全流程尚未验收。不要把本仓库旧的十次单轮CLI记录当作这些原项目的验收证据。
