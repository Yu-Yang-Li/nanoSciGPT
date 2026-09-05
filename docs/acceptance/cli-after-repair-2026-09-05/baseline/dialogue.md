# 连续CLI教学实测

模型：CLI当前默认（本脚本未覆盖模型设置）

## 第1轮

学生原文：

请按当前仓库 skills/research-baseline-builder/SKILL.md 带我做实验。我想用你们的 LAMOST 光谱示例预测恒星温度，你帮我实际跑一下。结果另存到 out/cli-after-repair-baseline，不修改已有文件。

CLI实际回复：

已按仓库技能**实际跑完**随机森林温度预测基线，结果另存到 `out/cli-after-repair-baseline/LAMOST光谱有效温度回归/`，未修改原有文件。

使用 2000 条光谱、128 个流量特征，预测 `teff`；1600 条用于训练，400 条用于测试。

| 指标 | 实测结果 |
|---|---:|
| 测试集 RMSE（均方根误差） | **83.44 K** |
| 测试集 R² | **0.9981** |
| 训练集五折交叉验证 R² | **0.9477 ± 0.0946**（均值 ± 标准差） |

结果表明模型在这份课程数据的随机留出测试上表现很好；交叉验证存在波动，不能据此保证跨巡天、跨仪器或新温度范围的效果。下一步适合检查不同温度区间的误差。

已核验生成的文件：

- [metrics.json](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-baseline/LAMOST光谱有效温度回归/metrics.json)
- [baseline_summary.json](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-baseline/LAMOST光谱有效温度回归/baseline_summary.json)
- [train_log.txt](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-baseline/LAMOST光谱有效温度回归/train_log.txt)

训练好的模型也已保存为同目录下的 `rf_model.joblib`。

原始日志：turn-1.jsonl；退出码：0

## 第2轮

学生原文：

我想自己再试一遍。请给我这次实际使用的输入、目标、运行命令和实际结果文件位置，再解释这个分数说明了什么。

CLI实际回复：

这次实验是：**用一条光谱的 128 个流量特征，预测这颗恒星的有效温度。**

- **输入文件**：`data/course/lamost_atlas_a_teff_2000.csv`
- **输入特征**：`flux_bin_000` 至 `flux_bin_127`，共 128 列。
- **预测目标**：`teff`，单位 K；根据仓库元数据，标签来自 `Teff_sed`。
- **数据划分**：2000 条光谱中，1600 条训练、400 条测试；随机种子为 `42`。
- **模型**：200 棵树的随机森林回归；训练集内另做五折交叉验证。

这次实际使用的是 `S2` 环境的 Python。在仓库根目录打开 PowerShell，运行命令为：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'

& C:/Users/16571/.conda/envs/S2/python.exe -B -m nanoscigpt.baseline --case lamost --out_root out/cli-after-repair-baseline
```

**你重跑时，请把最后的输出目录换成尚不存在的新目录**，例如 `out/my-lamost-baseline-01`，避免覆盖本次结果。其余参数保持一致即可复现实验。

本次结果保存在：

```text
C:\Users\16571\Documents\Codex\2026-09-04\nanoscigpt-classroom-hardening\
out\cli-after-repair-baseline\LAMOST光谱有效温度回归\
```

其中可直接查看：

- [metrics.json](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-baseline/LAMOST光谱有效温度回归/metrics.json)：测试指标。
- [baseline_summary.json](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-baseline/LAMOST光谱有效温度回归/baseline_summary.json)：数据来源、划分、交叉验证和模型摘要。
- [train_log.txt](C:/Users/16571/Documents/Codex/2026-09-04/nanoscigpt-classroom-hardening/out/cli-after-repair-baseline/LAMOST光谱有效温度回归/train_log.txt)：实际训练日志。

分数应当这样理解：

| 实际结果 | 含义 |
|---|---|
| **测试 RMSE = 83.4367 K** | 400 条测试光谱的温度预测，均方根误差约为 83 K。它对大误差更敏感，不代表每颗恒星都只差 83 K。 |
| **测试 R² = 0.99809** | 模型解释了该测试集约 99.81% 的温度变异；这不是“99.81% 的恒星预测正确”。 |
| **五折 R² = 0.94771 ± 0.09455** | 换用不同验证折时，成绩存在波动；这里的 ± 是折间标准差，不是置信区间。 |

这说明：**在同一课程数据范围内，光谱特征能够很好地预测参考温度标签。** 它还不能证明对其他巡天、仪器或不同数据分布同样有效，也不能把相对于参考标签的误差直接当成相对于恒星真实温度的误差。

原始日志：turn-2.jsonl；退出码：0
