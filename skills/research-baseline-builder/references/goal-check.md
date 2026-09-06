# Goal Check

After analysis or modeling, check the result against the original scientific goal before writing a confident conclusion.

## Questions

- Does the output match the requested scientific output?
- Did the model use only information available at the intended decision time?
- Does the split match the scientific generalization target?
- Are the metrics aligned with the scientific cost of errors?
- Are negative, weak, or inconclusive results stated plainly?

## Common Mismatches

| Result | Does not prove |
|---|---|
| High prediction accuracy | causal effect or mechanism |
| Significant correlation | causation |
| Feature importance | biological/physical mechanism |
| Clean clusters | real scientific classes |
| Good random-split score | cross-site, cross-patient, cross-batch, or future performance |
| Nice visualization | quantitative support |
| Low error on held-out rows | robustness to new instruments, labs, materials, or cohorts |

## Explaining a first result

For a beginner, start with the actual input, prediction target, and one interpretable error with its unit. Provide the saved result location or a reproducible command. Expand the diagnosis when the student asks for it; respect a requested short answer throughout the reply.

- Distinguish RMSE (root mean squared error, more sensitive to large errors) from MAE (mean absolute error). Neither by itself gives an error bound for every prediction.
  In a Chinese explanation, use “平均绝对误差为…（MAE）” or “均方根误差为…（RMSE）”. Claims about “多数样本” or a typical range need the actual error distribution or quantiles, rather than renaming either aggregate as typical error.
- State whether the target is a measurement, catalog estimate, or teaching label. Agreement with that label is not automatically accuracy against physical truth.
- A difference between validation and test scores is an observation. Explanations involving outliers, leakage, or distribution shift are hypotheses until the corresponding checks support them. High R² alone does not establish either excellent generalization or an invalid score.
- A feature index needs an actual coordinate/identity mapping before making domain claims. Overall wavelength range and feature count alone do not establish bin edges or spectral-line identity. If a source cannot be verified, separate that uncertainty from the measured model result.
- Classification accuracy or confusion does not establish what caused the decisions. A model may use physical signals, instrument effects, preprocessing artifacts, or a mixture. Misclassification does not prove it only relies on superficial similarity; a physically informed model can still be wrong. Say the current experiment does not establish mechanism understanding, not that physical understanding has been disproved.

## Closing the explanation

Use the student's language. End with the answer to their question and, if useful, one next check. The scope of the result belongs beside the score it qualifies; a separate formulaic conclusion would only repeat it. For a short-answer request, keep the entire reply within that request, including headings and closing remarks.
# 解释一次分类结果时

验证集和测试集分数接近，是这两个划分上的观察，不足以单独证明结果稳定、没有过拟合或没有记忆样本。需要另查独立性、重复运行及相应诊断。普通教学追问不必展开所有检查，但不能把尚未检查的原因说成已经确定。
