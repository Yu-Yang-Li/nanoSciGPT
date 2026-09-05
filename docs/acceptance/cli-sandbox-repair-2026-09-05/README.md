# 受限 CLI 沙箱修复与实际读文件检查

用户在前一轮明确列出的“失败处理修补、备份沙箱状态并受限重建”确认问题后回复“继续”。本轮据此执行，不将此前自动目标续跑消息视作同意。

## 实际改动

只将原来的损坏文件重命名为 `C:\Users\16571\.codex\.sandbox\deny_read_acl_state.json.corrupt-20260905-150856-663`，没有递归删除、修改全局配置或放宽执行权限。重命名前后 SHA256 均为 `6A4875DDACEAA91FB3369F0F6D962F77442DAF1B1D97733457D12BCABDF79441`，原22个零字节完整保留；需要回查时仍可读取该备份。

随后由新的 Codex CLI 会话自动创建合法 JSON：`{"principals": {}}`。没有手工编造该状态，也没有单独改写 ACL。此操作修复的是状态解析故障，未查明原文件为什么损坏。

## 实际执行证据

- 新会话：`01a07066-25be-71c1-92e1-cf3a91136a7a`，CLI `0.153.3`。
- 命令未覆盖默认模型；该会话实际 `turn_context.model` 为 `gpt-6-astra`。
- 已读取会话的 `turn_context`：`sandbox_policy.type=workspace-write`、`network_access=false`、`approval_policy=never`。
- 模型实际执行 `Get-Content -LiteralPath .\pyproject.toml -TotalCount 8`，命令事件 `exit_code=0`，返回内容与仓库文件相符。
- [输入](turn-1-input.txt)、[完整 CLI 命令](turn-1-command.json)、[原始事件](turn-1.jsonl)、[最终回复](turn-1-reply.md)、[stderr](turn-1-stderr.txt)分别保留。

stderr 仍有本机 Responses WebSocket 426 和另一 MCP 服务连接报错。这次 CLI 最终能够完成模型响应和本地命令，不代表这些旁路服务也已修复。

这是沙箱读文件验收，不是三个 Skill 或整堂课验收。记录中的 `classroom_execution_passed=false` 保留原值。后续教师实测分别保存，新会话不继承本线程；记录脚本又增加了每轮 `features.memories=false` 的进程级覆盖，以避免旧项目记忆影响无历史上下文验收。这不改变用户全局记忆设置。
