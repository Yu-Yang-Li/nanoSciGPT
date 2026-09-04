# 干净克隆复现验收

日期：2026-09-04  
来源：`origin/codex/classroom-hardening-20260904`  
克隆提交：`c9a88e4373ae69f696915f733a8a585ee2b80585`

## 操作

在与开发目录不同的新目录中执行：

```powershell
git clone --branch codex/classroom-hardening-20260904 --single-branch https://github.com/Yu-Yang-Li/nanoSciGPT.git <clean-clone>
python -m nanoscigpt.baseline --case lamost --out_root out/clean-proof/lamost
python -m nanoscigpt.classroom --domain protein --profile smoke --out_root out/clean-proof
```

## 结果

- 克隆后 `HEAD` 与远端提交一致，工作树干净。
- LAMOST 基线实际完成：2000 条课程光谱，测试 RMSE `83.4367` K、R² `0.9981`；生成 `baseline_summary.json`、`metrics.json`、`train_log.txt` 和 `rf_model.joblib`。
- Protein smoke 实际完成：预训练验证损失记录、采样 `MRATD`、下游 composition teaching classification 和 `run_report.json` 均生成；设备为 `cpu`。
- 运行没有读取开发目录的 `out/` 或 `.tmp/` 产物，所有输出都写入新克隆的 `out/clean-proof/`。

这条检查只证明 GitHub 分支可独立复现两条最低教学路径；它不把课程规模 smoke 结果写成真实科学结论。
