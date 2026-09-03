# nanoSciGPT 课堂 Skills

课程最终拆成六个职责独立的 Skill。目前已落地的部分如下：

1. [`nanoscigpt-research-baseline-builder`](nanoscigpt-research-baseline-builder/SKILL.md)：把科学问题变成第一份基线；仓库可直接运行 LAMOST `teff`、带标签 CSV 分类/回归和单条数值时序 CSV 预测，图像与 FITS 暂先完成接入设计。
2. [`nanogpt-pretraining`](nanogpt-pretraining/SKILL.md)：用 tiny Shakespeare 小样例分开看文本预训练和带标签任务头。
3. [`nanoscigpt-scientific-language`](nanoscigpt-scientific-language/SKILL.md)：把同一思路迁移到蛋白质、DNA、SMILES、天气、晶体、三维结构、图像、光谱和连续物理场；蛋白质与DNA可接入FASTA，SMILES可接入学生自己的CSV，其余六类可接入带回归标签的NPZ数组。
4. [`autoresearch-model-iteration`](autoresearch-model-iteration/SKILL.md)：读取已有模型 V0，先生成设置，再运行一个只改变训练预算的候选，并留下完整比较和下一步。
5. [`ai-scientist-v1-workflow`](ai-scientist-v1-workflow/SKILL.md)：读取已完成的 AutoResearch 比较，生成单一路线的相关工作、表、图、证据映射、短稿和规则审查。
6. [`ai-scientist-v2-tree-search`](ai-scientist-v2-tree-search/SKILL.md)：从 v1 的单一路线继续，建立两条可比较路线，逐条批准执行并按同一评价器作出取舍。

所有 Skill 都沿用学生已经提供的信息，每轮推进一项可检查的操作。课程数据与来源见 [`data/manifest.json`](../data/manifest.json)。

当前实现的是普通电脑可运行的基础教学能力。The AI Scientist v1/v2 分别用于解释单路线计算研究和多路线比较，均不等同于原项目的完整复现。

## 安装

从仓库根目录运行对应平台的脚本，并显式给出Codex Skill目录：

```powershell
pwsh -NoProfile -File scripts/install_skills.ps1 -Destination (Join-Path $env:USERPROFILE ".codex\skills")
```

```bash
bash scripts/install_skills.sh "$HOME/.codex/skills"
```

脚本会预先检查六个来源目录和六个目标目录。任何同名目标已经存在时，安装整体停止，不覆盖旧Skill。
