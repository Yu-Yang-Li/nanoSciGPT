# 原版研究系统：模型调用与一次真实想法反思

日期：2026-09-05。这里记录直接调用原项目函数的实测，不是 Codex CLI 学生对话，也不是完整研究验收。

## 实际完成了什么

| 原项目入口 | 模型 | 实际结果 | 记录 |
|---|---|---|---|
| v1 `create_client` / `get_response_from_llm` | gpt-6-astra | 返回 OK | [原始记录](v1-default.json) |
| v1 同一入口 | scnet/GLM-5.3 | 原版路由拒绝模型名，尚未发起网络请求 | [失败记录](v1-glm.json) |
| v2 `backend_openai.query` 工具调用 | scnet/GLM-5.3 | 强制调用 `record_probe`，返回 `value=7` | [原始记录](v2-glm-tool.json) |
| v2 同一工具调用 | gpt-6-astra | 返回 `value=7` | [原始记录](v2-default-tool.json) |
| v2 图像输入 | gpt-6-astra | 读取已有教学训练曲线，正确回答峰值位于第 3 步 | [原始记录](v2-default-image.json) |

上游固定版本、被调用文件的 SHA256 均在记录中。v1 `llm.py`、`generate_ideas.py` 和 v2 `backend_openai.py` 的 Git 差异检查为空。GLM 的文本工具调用通过不代表其图像能力已经通过。

本机 `http://127.0.0.1:10100/v1` 实际提供了兼容接口。这纠正了此前“没有独立环境变量中的 API key，因此无法继续”的过强判断。`local-no-auth-probe` 仅是该已核查本机接口的 SDK 占位字符串，不是云服务密钥，不能当作学生通用配置。

## v1 实际提出并修改了一条方案

输入是八条手写教学蛋白质序列及其示意数值标签对应的监督微调模板；不是实验测得的蛋白质活性数据。

调用原版 `generate_ideas`，设置一次新想法、两轮反思计数（首次生成和一次反馈修改）。没有改写原版研究提示词或替换生成逻辑。

- 首次方案提出保护预训练表示的相对关系，并比较多个约束强度。
- 反思后收束为三种设置：原有微调、相似性轮廓约束、直接特征约束；要求只使用训练数据确定系数，保留原数据、模型及评价方式。
- 实际新增 **1 条**想法；`ideas.json` 中另外 2 条是上游种子，不能计作本次成果。
- 2 次真实 API 请求，耗时 94.25 秒；响应合计 6,319 tokens。
- **尚未验证新颖性，尚未执行该方案的实验，也没有论文和评阅。** 模型自己给出的 Novelty 分数不是独立证据。

可复核原文：[完整请求](ideation/requests.json)、[完整响应](ideation/responses.json)、[最终想法与原有种子](ideation/ideas.json)、[运行摘要](ideation/record.json)。请求与响应只保存正文，不保存认证头。

`probe.py` 与 `run_ideation.py` 是当时在 `out/native-api-20260905/` 执行的诊断脚本原文归档，依赖该原工作目录布局；不是学生安装入口，不能在本归档目录直接运行。

## 尚未完成

1. v1 的 Aider 改码、实验反馈、写作与评阅；GLM 模型名需要显式兼容，不能暗中换成另一模型。
2. v2 多阶段研究树、失败保留、中断恢复及写作；不能用工具参数探针代替。
3. 原版 autoresearch 的完整运行及受限 Codex CLI 连续课堂对话。

依赖安装也发现了真实 Windows 问题：深层 venv 路径触发 LiteLLM 包的长路径错误，原尝试退出码为 1；随后改用独立短路径环境，不修改 Windows 全局长路径设置，不修改原教学 Python。

## Aider 实验改码：失败也保留

短路径 `C:\Users\16571\.venvs\nsv1-0905` 已实际安装 Aider 0.86.2，核心导入通过，课程监督模板重新运行的基线 MAE 为 3.398047924。该环境复用系统包，安装器报告了与其他已装软件的依赖冲突；不是通过了冷环境安装验收。

当前已定位三种不能算改码成功的情况：

- 初始化时将辅助模型设为空，Aider 内部需要其 `token_count`，因此报错；随后显式使用同一模型担任辅助角色。[初始化失败](coder-failures/initialization.json)
- 默认模型请求超过 90 秒。底层 `num_retries=0` 没有阻止 Aider 自己继续重试，已只终止本次测试进程。[停止记录](coder-failures/timeout.json)、[实际对话](coder-failures/timeout-chat.txt)
- GLM 通过 `openai/scnet/GLM-5.3` 接入 Aider，但 5,000 输出额度下没有完成代码，实际回答只有“我将”。原实验文件没有变化，结果标为失败而不是“改码已完成”。[失败摘要](coder-failures/token-limit.json)、[回答原文](coder-failures/token-limit-answer.txt)、[完整日志](coder-failures/token-limit-chat.txt)

这说明 **v1 自己的模型名路由** 与 **Aider 的兼容接口路由** 是两处不同配置：后者能调用 GLM，不代表前者已修好。Aider 在截断时记录的 token 计数为 0，属于未取得可靠计数，不代表未调用或免费。

后续诊断关闭 Aider 的外层传输重试、关闭自动执行建议命令，限定单次请求，保留原实验提示词与 diff 编辑方式。这些是明确记录的教学运行限制，不是对原项目无限循环或全部默认行为的复现。

随后两次有界请求也没有产生代码：GLM 的 12,000 token / 180 秒设置耗时 185.578 秒后结束；默认模型的 6,000 token、low reasoning / 90 秒设置耗时 95.547 秒后结束（总耗时包含初始化）。它们均没有进入外层自动重试，也没有修改实验文件、数据、任务配置或初始模型；退出码均为 1，不误报通过。这里确认的是当前本机兼容服务下的失败，不据此断言模型本身不具备编程能力。

- GLM：[请求参数与消息](coder-failures/glm-bounded-request.json)、[失败摘要](coder-failures/bounded-timeout.json)、[日志](coder-failures/bounded-timeout-chat.txt)。
- 默认模型：[请求参数与消息](coder-failures/default-low-request.json)、[失败摘要](coder-failures/default-low-timeout.json)、[日志](coder-failures/default-low-timeout-chat.txt)。
- [最终诊断脚本原文](coder-probe-final.py)保持其原 `out/` 工作目录假设，不是学生运行入口。

**当前结论：想法生成及反思已实测；自主实验改码未通过。** 停止重复提高输出额度，下一步需要针对实际编程请求的接入兼容性和整轮超时处理做修复；不能用更小的“回复 OK”测试代替这一关，也不能直接把诊断脚本发布为成熟教学工具。
