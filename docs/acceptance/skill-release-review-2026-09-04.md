# 三个学生 Skill 上线审查

日期：2026-09-04  
审查范围：仓库 `skills/` 下公开给学生的三个入口  
审查依据：Skill 文本契约、真实 Codex CLI 对话、边界测试和仓库测试

## 结果

| Skill | 文本校验 | 真实压力场景 | 结论 |
|---|---|---|---|
| `research-baseline-builder` | `quick_validate.py` 通过 | LAMOST 课程数据、学生 CSV、数据不能上传 | 无 Skill 级阻塞 |
| `nanoscigpt-scientific-language` | `quick_validate.py` 通过 | protein 课程数据、weather 自有 NPZ、无标签光谱且不能上传 | 无 Skill 级阻塞 |
| `ai-scientist-research-loop` | `quick_validate.py` 通过 | 已有模型、只有指标文件、模型和数据不能上传 | 无 Skill 级阻塞 |

## 复核项

- 学生不必先说页码、阶段名或 `domain` 参数；Skill 根据已经说出的对象和文件继续。
- 文件可读且信息足够时直接检查或运行；缺一个会改变下一步的信息时只补一个问题。
- 没有标签、不能上传或缺少完整运行报告时停在设计，不生成分数或科学结论。
- 每条真实对话的原始 JSONL、学生输入和最终助手回复均可由 `tests/test_cli_dialogue_evidence.py` 逐字核对。
- 旧的分步材料仍保存在三个入口的 `references/` 中，没有继续作为学生可触发的第四、第五个入口。

## 证据位置

- [输入—输出索引](cli-dialogues/scenarios.json)
- [Codex CLI 对话记录](dialogue-tests-2026-09-04.md)
- [边界测试](boundary-tests-2026-09-04.md)
- [全量测试记录](pytest-non-course-2026-09-04.txt)

## 剩余条件

本审查只覆盖三个 Skill、数据入口和运行证据；整门课程是否可正式授课仍取决于另一会话合并并验收已经逐页确认的 15 页课程材料。本仓库不把这个未合并条件隐藏为“全部完成”。
