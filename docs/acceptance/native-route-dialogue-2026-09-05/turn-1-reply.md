第一轮先做只读预检，不启动训练：

```powershell
python -m nanoscigpt.classroom --list
```

请把输出中的 `protein` 那一行发给我；若命令失败，就贴完整报错。确认后我再带你运行蛋白质实验，输出将固定写入 `out/cli-native-check`。

本实验中：

- 读取单位：氨基酸；
- 保留关系：残基顺序和蛋白质序列边界；
- 预训练目标：预测下一个氨基酸；
- 课程数据：UniProtKB reviewed entries 的500条教学子集，属于真实来源子集，不是生成夹具。

当前本地命令执行器因 Windows ACL 初始化错误无法启动，因此我尚未创建产物，也没有修改任何课程源文件。仓库同名远程 Skill 已核对，但本地版本仍需在执行器恢复后复核。