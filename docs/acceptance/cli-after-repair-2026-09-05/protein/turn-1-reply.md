已实际跑完，结果保存在 `out/cli-after-repair-protein`。

- **配置**：CPU 单线程，2 层、2 头、64 维小模型。
- **预训练**：30 步，验证损失 `3.0951 → 2.9163`。
- **真实微调**：32 个训练样本、2 个 epoch；16 个验证样本的准确率 `56.25% → 43.75%`，本轮效果下降。
- 已验证预训练参数确实更新，原 checkpoint 保持不变。微调前使用新初始化任务头，不能视为训练充分的对照。

查看[实验结果说明](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-protein/实验结果.md)。

本次仅写入新输出目录。核验发现既有 `docs/upstream-adaptation.md` 在运行期间发生变化，本任务未写入或覆盖该文件。

