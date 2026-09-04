# 发布资源审计

日期：2026-09-04

## 本轮正式交付

- 三个学生入口及其参考材料；
- 十类离线课堂数据与统一CPU入口；
- 运行、比较、停止和证据包所需代码；
- 验收记录与数据来源说明。

## 暂不纳入本轮提交

- `assets/`：共119个文件、约158.50 MB，混合14页、16页和17页PPT的成图与原型；
- `docs/assets/`：共16个文件、约24.36 MB，属于16页过程稿的页面图片；
- `.tmp/`、`out/`和根目录运行日志：均为构建或实跑过程产物。

这些文件当前保留在本地，不删除。最终15页课程由另一会话冻结后，再单独选择唯一正式版本、补来源清单并纳入课程材料提交；在此之前不把多套相互冲突的页面图推入学生仓库。

## 链接检查

首次检查在61个相对链接中发现4个断链，均来自`skills/README.md`仍指向已收束的旧Skill入口。导航已改为三个公开入口；旧详细内容保留在`ai-scientist-research-loop/references/`。

修正并加入备用结果导航后，重新检查README、docs、skills与data说明中的71个本地相对链接：断链0个；空Markdown链接0个。

## 数据检查

- `data/manifest.json`共12个条目，列出35个必需文件；当前缺失0个。
- `data/`共50个文件，约10.85 MB；其中新增十个领域结果、结果清单和说明共12个文件，约16 KB。
- LAMOST与Palmer Penguins课程文件的SHA-256均与清单一致。
- 十类数据是否能真正进入CPU流程另由`smoke-2026-09-03.md`和`test_classroom.py`验证，不以“文件存在”代替运行验收。
- `data/precomputed_results/`逐领域保存真实`smoke`运行的可移植摘要；不含checkpoint和本机绝对路径。对应结构测试为`tests/test_precomputed_results.py`，当前通过。

## 自动化复测

- 三个 Skill 的 `quick_validate.py` 均输出 `Skill is valid!`。
- 对话索引与十份 JSONL 的最终助手消息逐字一致，见 `tests/test_cli_dialogue_evidence.py`。
- 复测命令 `python -m pytest -q`：`120 passed in 323.29s`；初次冷克隆的环境超时已在对话验收记录中保留并复测通过。
