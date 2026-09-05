# CLI教学验收的沙箱故障：已定位，尚未修复全局状态

2026-09-05。CLI版本`0.153.3`。本轮没有放宽权限，没有修改全局Codex配置或沙箱状态文件；仅做诊断，并修正课程对话记录脚本。

后续已获用户确认并完成[受限沙箱修复](../cli-sandbox-repair-2026-09-05/README.md)。本页保留当时未修复的诊断和失败原文，不再作为当前故障仍存在的证据。

## 实际诊断

使用CLI自身默认模型（当前用户配置为`gpt-6-astra`，调用未传`--model`），新建两个会话，均显式要求`workspace-write`和`approval_policy="never"`，只请求读取`pyproject.toml`前8行：

1. 在课程打磨仓库中：[事件](events.jsonl)、[stderr](stderr.txt)、[真实回复](reply.md)。
2. 在仅复制一份`pyproject.toml`的临时目录中：[事件](clean-events.jsonl)、[stderr](clean-stderr.txt)、[真实回复](clean-reply.md)。

两次模型回复均实际返回，但本地命令都在创建进程时失败：`helper_unknown_error: apply deny-read ACLs`。两次CLI进程退出码都是0，再次说明“有最终回复/退出码0”不能证明本地课堂操作成功。未运行训练，未进行完整Skill教学。

临时目录为`C:\Users\16571\AppData\Local\Temp\nanoscigpt-sandbox-probe-47c6fc42b29341cd944ac955f3faebdb`。两个新会话分别是`01a06ff6-ffa9-78e2-9aa5-de5b867d271b`和`01a06ff8-1bf6-7193-879c-99d60e47403b`。

## 根因证据

本机`C:\Users\16571\.codex\.sandbox\sandbox.2026-09-05.log`在13:08:57记录的错误链为：

```text
setup error: apply deny-read ACLs
Caused by:
    0: parse deny-read ACL state C:\Users\16571\.codex\.sandbox\deny_read_acl_state.json
    1: expected value at line 1 column 1
```

对该状态文件只读检查：长度22字节，最后修改时间2026-09-04 07:56:18，全部字节为`00`，不是合法JSON。仓库与干净目录得到同样错误，因此当前故障不由课程模型或训练显存引起。

[OpenAI源码](https://github.com/openai/codex/blob/main/codex-rs/windows-sandbox-rs/src/deny_read_state.rs)中的`load_state`会解析存在的文件，只有文件不存在时才建立默认状态；随后由沙箱机制重新保存状态。主分支源码用于解释和制定修复步骤，不当作本机二进制版本的逐字证明。当前根因判断以本机错误链和文件字节为准。

已向用户请求：先备份损坏文件，再让Codex按原机制重建，保持受限沙箱复测。尚未获得回复，未擅自重置这个全局文件，也未删除或修改ACL。当前无证据证明重建后所有其他问题都会消失。

## 修正旧记录的解释

旧会话`01a06f2b-96d4-73a1-82cf-530287c421a3`第二轮确实读取过本地文件。读取该会话原始`turn_context`，确认：

| 轮次 | sandbox_policy.type | approval_policy | model |
|---|---|---|---|
| 第一轮 | workspace-write | never | gpt-5.6-sol |
| 第二轮 | danger-full-access | never | gpt-5.6-sol |

来源：`C:\Users\16571\.codex\sessions\2026\09\05\rollout-2026-09-05T09-25-22-01a06f2b-96d4-73a1-82cf-530287c421a3.jsonl`。仅检查了指定会话的权限和模型字段，未改写任何原话。因此第二轮成功不能证明受限沙箱恢复了，也不能作为原条件下连续教学通过的证据。

## 记录脚本的实际修正

`capture_classroom_dialogue.py`原先默认硬编码`gpt-5.6-sol`，且只在首轮显式传沙箱参数。本轮改为：

- 不传`--model`时沿用CLI默认模型；显式传`scnet/GLM-5.3`时才进行该模型测试。
- 每轮，包括resume，都显式传`workspace-write`；不依赖用户全局权限默认值。
- 记录每轮完整启动参数；标准输入关闭，避免继承无关输入。
- 仍将对话完成与课堂执行通过分开，ACL错误仍返回记录脚本失败。

4项相关测试通过，见[测试报告](capture-tests.xml)。这些测试使用模拟CLI进程边界检查参数与记录行为，不是4次真实模型教学；本轮两次真实诊断均失败。没有将GLM兼容性或三个Skill的完整教学标为通过。
