# 课堂交付复核：入口已统一，完整研究尚未验收

2026-09-05。本轮不改PPT和正式15页文字，不重写历史对话。检查对象为独立打磨仓库的未提交版本，基于`e0028b127d62fec2f854ac0886e68e3e4df37778`；不能将本地结果当作GitHub已发布结果。

后续进展见[原版 API 与研究阶段实测](../native-api-2026-09-05/README.md)：已找到并实际使用本机模型接口，v1 想法生成和反思通过，v2 工具与默认模型读图通过；Aider 在独立短路径环境导入通过，但自主改码仍因超时或截断失败。下文环境清单保留原检查时点，不再据其中“未配置环境变量”推断不存在可用服务。

## 本轮改动

- README的当前快速开始改为离线LAMOST、文本预训练与回答微调、所选科学数据预训练与任务微调；旧四域教程和历史结果保留并注明身份。
- Skill总索引与讲师导航统一指向三个当前Skill。两份旧讲义保留，但不再作为原版B线的操作入口；不存在的`esm`安装extra不再推荐。
- 安装时不再暴露`nanoscigpt-autoresearch`、`nanoscigpt-ai-scientist-v1`、`nanoscigpt-ai-scientist-v2`，避免这些名称静默启动本地规则演示。旧`autoresearch/`模块仍在，显式请求历史离线演示时可使用`python -m autoresearch.*`对应模块。

这是一次明确的兼容性调整：过去使用上述三个命令名的人需要改用历史模块命令或当前原项目Skill。没有将这些名字改接`prepare`并冒称原版研究已完成。

独立子智能体复核了本轮七文件修改前后差异、README命令参数和实际数据路径，规格与质量均通过；没有发现需要修正的问题。该复核为只读，不是额外一次训练或Agent教学。

## 已实际执行的质量检查

| 检查 | 当前证据 | 验证范围 |
|---|---|---|
| 完整pytest | **165 passed in 429.58s**；[XML](regression.xml) | 本轮入口修正之前收集的完整测试，含真实CPU训练、微调、数据接入、原版v1实验接口及旧演示测试 |
| 入口修正后的回归 | **7 passed in 0.20s**；[XML](entrypoint-tests.xml) | `test_public_skill_surface.py`和`test_instructor_materials.py`；含新增2项，不与165相加称作全量结果 |
| 修改前失败验证 | 新增测试得到`2 failed, 4 passed` | 旧console scripts与旧导航确实触发失败，改动后通过 |
| 三个Skill格式 | 均输出`Skill is valid!` | 使用Python `-X utf8`；初次默认GBK解码失败，不算通过，也未改系统编码 |
| 十域预检 | `classroom --list`十类均ready | 是数据文件及形状检查，不以此替代训练；训练由pytest及已有逐域真实记录证明 |
| 数据清单 | 12项、35个必需文件，缺失0；数据50文件、11,416,445字节 | 50文件均已由Git跟踪；LAMOST与Penguins两份CSV的SHA256与清单相符 |
| 本地链接 | 入口修正后，50份Markdown中130个本地链接，断链0、空链接0 | README、docs、skills、data；不验证网络链接或历史论文事实，也不覆盖随后新增的本报告 |
| 安装验证 | 新隔离venv中离线editable安装成功 | `--system-site-packages`复用本机依赖，不是独立依赖环境或远端干净克隆 |
| 安装后的命令 | 元数据仅暴露`nanoscigpt-baseline`和`nanoscigpt-classroom`；后者`--list`十域ready | 来自隔离环境的dist-info，不是只读取pyproject配置 |
| 修改格式 | `git diff --check`无内容错误 | Git仍提示下次操作可能转换LF/CRLF；没有为此全量格式化 |

完整测试命令：

```powershell
python -m pytest -q --junitxml=out/release-gate-20260905/regression.xml
python -X utf8 -m pytest tests/test_public_skill_surface.py tests/test_instructor_materials.py -q --junitxml=out/release-gate-20260905/entrypoint-tests.xml
```

本机Python为3.12。测试启动有既存`RequestsDependencyWarning`，涉及urllib3/chardet/charset_normalizer组合；本轮没有升级全局依赖消除警告。真正冷环境依赖安装仍需最终克隆验收。

安装检查使用`out/release-gate-20260905/install-probe`，执行`pip install --no-deps --no-build-isolation --no-index -e .`。全局已有nanoSciGPT未被卸载，安装位于该隔离环境。

## 不能因此宣布完成的部分

1. **三个Skill新版连续CLI教学**：当前受限沙箱的[损坏状态文件](../cli-sandbox-diagnosis-2026-09-05/README.md)尚未修复，授权仍待用户回复。旧对话和单轮成功不代替新版连续教学。
2. **原版autoresearch、v1、v2完整研究**：本轮`upstream doctor`仍显示Aider不可用、当前torch为CPU版、CUDA不可用，相关API环境变量未配置。它只是就绪信息，不排除其他尚未接入的模型服务；完整Agent、写作、视觉反馈与v2停止恢复尚未验收。
3. **完整Protein贯通路径**：原版v1的实验模板已能接续课程模型，但autoresearch尚未承接该模型，不能换一个模型后声称延续原实验。
4. **任意自有数据**：已有表格基线和蛋白质CSV的实际接入；其他类型不能凭教学夹具跑通就宣布学生任意原始文件可直接用。不能共享数据的新版多轮引导也待真实CLI测试。
5. **版本发布**：本轮没有提交或推送；旧版本干净克隆记录不是当前未提交版本的复现证明。正式课件一致性待另一会话冻结后对接。

`tests/`不依赖固定`.tmp/`目录，测试使用pytest临时目录；历史原始对话中的旧路径保持不改。辅助脚本`prepare_dialogue_fixtures.py`的默认输出仍为`.tmp/dialogue-fixtures`，新验收应显式传`--output out/...`，不能假设旧临时数据随Git分发。

当前判断：**尚不能标记“可用于正式课堂”**。本轮补齐了发布入口与质量检查证据，没有将原项目接口实验、旧规则演示或脚本测试替代真实Agent研究。
