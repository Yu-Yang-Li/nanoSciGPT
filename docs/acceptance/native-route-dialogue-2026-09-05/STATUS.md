# 本次不是通过的连续带练记录

发布说明（2026-09-05）：第二轮原始`turn-2.jsonl`含本机记忆读取内容，不公开上传。未改动原件，已移至本机`out/publish-20260905/private-records/native-route-turn-2.jsonl`，SHA256为`87e3a144536dca2ea682c81e10488e03540d8b94e9834353d04cd3dce745c94b`。公开目录保留实际学生输入、第一轮回复和失败说明；以下“保留”不表示该私有原件在Git中。

后续诊断补充：第二轮实际成功读取过本地文件，但原始会话`turn_context`显示它已从首轮`workspace-write`变成`danger-full-access`，不能作为受限沙箱恢复的证据。记录脚本现已修正为每轮显式保持受限沙箱；详见[根因与修正](../cli-sandbox-diagnosis-2026-09-05/README.md)。本目录原始对话和事件不改。

第一轮CLI已返回实际回复，但本机Windows sandbox创建命令进程失败：`helper_unknown_error: apply deny-read ACLs`。因此没有本地训练、没有读取本地新版Skill。助教转读了远程旧版Skill，不能据此验收本次修改。

第一轮的退出码0只代表对话返回，不代表课堂实验成功。第二轮启动后发现相同环境前置条件未恢复，已停止本次测试；未运行第三轮。第一轮原话见`dialogue.md`；各轮JSONL和stderr保留，不覆盖为成功。

另外出现了本机MCP连接和WebSocket Upgrade报错，不能仅据这些附带错误确定训练命令失败的全部原因。下一次应先让同一CLI沙箱完成无副作用的本地文件读取，再重新进行完整的连续带练；本轮没有调整沙箱权限或修改全局Codex配置。
