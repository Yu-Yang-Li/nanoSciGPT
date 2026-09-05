# 学生自己的蛋白质CSV

学生给出CSV时先查看列名与少量内容，沿用他已经说出的目标。序列列明确而目标还不明确时，只问希望预测哪个量；不是让学生填写下面的参数。路径和字段由助教代入。

```powershell
python -m nanoscigpt.student_protein --csv <学生CSV绝对路径> --sequence-column <序列列> --target-column <数值目标列> --data_root out/student-data/run1
python -m nanoscigpt.classroom --domain protein --profile classroom --data_root out/student-data/run1 --out_root out/student-model/run1
```

只收到序列、没有标签时省略`--target-column`。训练后报告会写`skipped_no_labels`，没有监督分数；这不妨碍展示序列预训练和生成。需要继续微调时沿用同一份自有数据，而不是默默换回仓库样例：

```powershell
python -m nanoscigpt.tasks.downstream_demo --domain protein --data_root out/student-data/run1 --ckpt out/student-model/run1/protein/model/ckpt.pt --adaptation finetune --epochs 2 --max_samples 32 --out_dir out/student-model/run1/protein/finetune
```

当前入口接收标准氨基酸字母组成的独立序列和可选的有限数值回归标签，不会自动解释分类名、突变记号、非标准残基或缺失值。这些内容需要先和学生核对，不能静默删行或填造标签。原文件保持不变，处理后的数据保存在新目录，元数据记录源路径、SHA256和字段名。

已有分组时可用`--split-column <列名>`，取值为`train`/`val`；否则按相同序列分组后固定随机留出约20%，两部分各需至少两条样本。同一序列不能跨两部分出现。这只是教学留出，不保证相似蛋白家族独立；科学泛化需要另行设计同源性划分。小模型任务头只读取上下文范围内的前缀，不据此宣称已经利用完整长蛋白结构或功能信息。

不能上传的数据不需要传来：可以帮助学生按本地列名执行上述命令，但没有收到运行证据时只说明方案，不报告分数。不要将学生原始数据、处理后数据或含隐私的日志自动加入Git。
