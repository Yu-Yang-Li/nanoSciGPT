# 预生成课堂结果

这里保存十类内置数据的真实 `smoke` 运行摘要。课堂现场运行受环境影响无法完成时，可以打开相应领域的 JSON 继续讲解结果格式。

这些文件只是备用展示材料，不能写成“本次学生运行成功”。每份结果都保留预训练指标、下游任务指标、数据来源入口和教学边界；模型 checkpoint 与本机绝对路径不进入仓库。

重新完成十领域 smoke 后，可统一更新：

```bash
python scripts/export_precomputed_results.py --recorded-on YYYY-MM-DD
```
