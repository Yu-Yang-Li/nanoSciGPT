读取失败。唯一一次本地 shell 命令在创建进程时被沙箱拒绝：`helper_unknown_error: apply deny-read ACLs`，未能读取 `pyproject.toml`。

未修改文件、未联网、未尝试放宽沙箱权限。