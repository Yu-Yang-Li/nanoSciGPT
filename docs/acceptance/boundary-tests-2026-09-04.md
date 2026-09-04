# 课程边界测试

日期：2026-09-04

| 边界 | 可复核证据 | 结果 |
|---|---|---|
| 没有标签 | `test_baseline_cli_does_not_train_a_student_csv_without_a_target`；CLI记录`science-private.jsonl` | 监督入口缺少目标列时退出且不创建结果；无标签光谱只设计遮盖预训练，不生成监督分数 |
| 数据不能上传 | `baseline-private.jsonl`、`science-private.jsonl`、`ai-private.jsonl` | 三个入口均停在设计或只补一个问题，没有声称读取数据或完成实验 |
| 只有单一明确任务 | `test_route_decision`中的`task_sharing=false`分支 | 路线为`use_specialized_model`，不强行升级为领域基座 |
| 没有实验或观测反馈 | `test_v1_blocks_writing_when_evaluated_comparison_is_missing` | 缺少同口径比较时写出`blocked_no_evaluated_evidence`，不生成`draft.md` |
| 新版本没有超过基线 | `out/acceptance-student-journey/autoresearch/protein/comparison.json` | 增益0.0000，记录`stop_branch`；v2最终保留baseline |
| v2中途停止后恢复 | `test_v2_run_next_resumes_the_same_route_after_an_interrupted_attempt` | 同一`tree_state.json`从`running`恢复；`attempts`从1增至2，并记录`resumed_after_interruption=true` |

相关命令的新鲜结果：

```text
tests/test_baseline_cli.py::test_baseline_cli_does_not_train_a_student_csv_without_a_target  1 passed
tests/test_core.py::test_route_decision                                                    1 passed
tests/test_ai_scientist_v1.py::test_v1_blocks_writing_when_evaluated_comparison_is_missing 1 passed
tests/test_ai_scientist_v2.py                                                              6 passed
```

完整CLI记录位于`docs/acceptance/cli-dialogues/`，统一课程贯通记录见`deep-path-2026-09-04.md`。
