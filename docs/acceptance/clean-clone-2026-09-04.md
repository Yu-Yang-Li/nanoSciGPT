# 干净克隆复现验收

日期：2026-09-04  
来源：`origin/master`  
克隆提交：`88bd1e3005cf9e61698d4b20e16baadb220bb163`

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

## 最终 master 复测

在文档审查提交 `0af722dc51196b4cb3e90854d926ce9f22d2f136` 的干净克隆中再次执行同样两条命令。`RUN_CWD` 明确为新克隆目录，LAMOST 的 `baseline_summary.json` 和 Protein 的 `run_report.json` 均从该目录生成；结果仍为 LAMOST `RMSE 83.4367` K、R² `0.9981`，Protein 预训练与下游任务 `completed`，设备 `cpu`。

随后在发布头 `b01abeaf7141140f7700849ac8adef008177c913` 的全新克隆中再次执行，`RUN_CWD`、LAMOST 结果文件和 Protein 结果文件均确认位于该克隆；两条路径仍分别得到 `RMSE 83.4367` K / `R² 0.9981` 和 `completed`。这次复测之后只增加验收文档，不改变运行代码。

## 当前发布头复测

在最新 `master` 发布头 `ef389c2d96e3c19fe00c9d13d078f7511f225a55` 的全新克隆
`nanoscigpt-clean-proof-20260904-5` 中重新执行两条命令。`RUN_CWD` 明确位于该
克隆目录，且克隆后工作树干净。LAMOST 输出的 `baseline_summary.json` 记录
`test_rmse=83.43673513294728` K、`test_r2=0.99809165883606`；Protein smoke
输出 `status=completed`、`downstream_task=completed`、`device=cpu`，预训练、采样和
下游教学分类结果均生成。该提交只包含讲师资料归档和验收文档更新，未改变运行代码。
