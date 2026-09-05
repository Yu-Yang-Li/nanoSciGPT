# GLM课堂问答与v1接口：分层核查

后续进展见[流式对照](../native-stream-2026-09-05/README.md)：相同原研究消息改为流式后撑过200秒，最终仍因`length`结束且没有代码补丁。下文非流式失败保留为历史记录，不当作当前唯一故障。

2026-09-05。课程工作区为独立的`nanoscigpt-classroom-hardening`分支。本轮未改PPT、全局代理、模型配置或凭据，未把规则演示替换进原版研究流程。

## 1. 提问为何卡住

读取上一轮原始CLI会话后，发现GLM调用`request_user_input_async`时，把`options`写成了含`label`和`description`的对象数组；实际工具要求字符串数组。前两次解析失败，第三次改成字符串后才成功。之后模型又调用`sleep`等待回复，最终第二轮600秒超时。不能把这次失败泛化成“GLM所有工具都不能调用”，也不能把成功提交异步问题算成已交付最终答复。

源码证据是本机会话`01a07085-d8a5-72a2-88c7-99cea2116b39`的`function_call`与对应工具结果；可读失败材料见[上一轮记录](../cli-after-repair-2026-09-05/private-ai-scientist-glm-attempt2/README.md)。

课程Skill只补一条针对命令行的衔接说明：普通文字提问后结束本轮，收到下一条学生消息再继续；实际文件与实验操作仍用工具。没有更改工具schema或限制整个Codex的能力。

使用**完全相同的两条学生输入、相同GLM模型、相同受限沙箱**另开会话复测：两轮均退出0并产生最终回复，没有再次调用异步提问或睡眠等待，没有执行训练或创建实验文件。原始事件仍在本地`out/cli-text-guidance-glm-20260905/`。

- [两轮学生输入与实际回复](glm-cli/dialogue.md)
- [会话条件](glm-cli/session.json)
- [实测Skill快照](glm-cli/skill-tested.md)

这次只说明卡住的交互路径得到改善。回答仍偏长，第一轮做了未经任务背景支持的分数好坏类比；第二轮也尚不足以作为严谨的实验方案。未验收全部私有数据场景、多次稳定性或完整课堂，不把`completed_turns=2`改写成全面教学通过。

## 2. 为什么关闭推理没有生效

本机10100端口由opencodex 2.42.0提供。在其已安装的`buildOpenAIChatPassthroughRequest`中，用不含凭据、完全不联网的输入实测：`thinking`字段被白名单丢弃，`max_tokens`和`reasoning_effort`则保留。见[实际构造器检查](proxy-field-record.json)与[检查脚本](proxy-field-probe.ts)。源码摘要一并记录，未修改安装文件。

这证明此构造器会丢字段，但不证明只修改代理就能解决GLM推理开销。对同一个已配置SCNet服务另做直连对照：

| 检查 | 实际结果 | 记录 |
|---|---|---|
| 首次直接请求 | 401；诊断脚本误把`${SCNET_API_KEY}`引用当作值，属于本次探针错误 | [记录](direct-record.json) |
| 按已有用户环境解析凭据，带`thinking: {type: disabled}` | 400 Format Error；第一次只留状态，随后一次保留脱敏错误正文 | [状态](direct-resolved-record.json)、[错误正文](direct-diagnostic-record.json) |
| 仅去掉`thinking`，其余请求保持一致 | 200，回答`2`；63输出token中61为推理token | [记录](direct-no-thinking-record.json) |

直连使用的是同一个已配置服务与已有凭据，凭据仅进入请求头，没有写入日志或Git，也没有修改用户或系统环境。请求内容只有人工构造的算术问题。它不是原版改码验收。

[SCNet接口文档](https://www1.scnet.cn/ac/openapi/doc/2.0/moduleapi/api/chat.html)的Chat Completions部分只为Qwen3和DeepSeek-V4列出`enable_thinking`，未列出GLM关闭思考的支持；不能直接把其他服务商的参数套过来。综合文档与上述实际400/200对照，本课程不再将“关闭GLM思考”作为已验证配置。此结论针对本机当前服务接入，不断言模型在所有部署中都不能关闭思考。

## 3. 继续原版研究的条件

原版研究提示词、数据、任务、初始模型和Aider编辑方式不变。由于推理和代码共同占用输出额度，有界检查明确把**单次总输出**上限设为20000 token、API等待上限240秒，最多一次请求；不启用自动重试，不自动运行返回代码。

实际结果仍失败：203.937秒返回`litellm.BadGatewayError: ... Timeout elapsed`，回答为空、没有代码补丁，所有固定输入与实验代码摘要均未变化。Aider吞掉了底层错误后返回空正文，外层探针据此正确判为`failed_no_implemented_change`而不是通过。不能把Aider记录的0 token解释成没有调用或没有费用；本次未收到可靠usage。

[运行摘要](native-coder/record.json)、[原请求](native-coder/requests.json)、[错误日志](native-coder/process-log.txt)与[脚本快照](native-coder/probe.py)已保存。脚本保留原`out/`路径假设，仅为诊断记录，不是学生入口。基线真实运行，新实验没有运行。

这次说明提高单次预算没有解决当前非流式长请求。客户端240秒尚未到就收到网关超时，不能直接断言上游服务器宕机；需要进一步区分代理、上游首响应期限与非流式等待。后续应检查传输方式或使用经验证的研究服务，不继续盲增token，不改写假设冒充原项目成功。没有修改全局代理来强行延长所有会话的超时。

所有诊断仅服务于接入与教学复现。没有训练学生私有数据，没有取得新的科学结论；原版autoresearch、v1完整实验写作与v2研究树仍须各自验收。

## 4. 本轮回归

本仓库全量`tests/`重新执行：**186项通过，380.64秒，0失败、0跳过**，见[完整报告](full-tests.xml)。命令为`python -X utf8 -m pytest tests -q --tb=short --junitxml=out/provider-boundary-20260905/full-tests.xml`，仅运行进程设置OMP/MKL线程数为1。三个Skill格式检查通过。接口失败与真实对话单独按以上记录判定，不因代码测试通过而升级为完整研究通过。
