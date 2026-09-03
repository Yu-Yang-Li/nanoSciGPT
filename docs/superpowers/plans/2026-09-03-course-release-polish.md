# nanoSciGPT Course Release Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将六个已跑通的教学 Skill 收口为可安装、可运行、可复查的16页课程仓库。

**Architecture:** 保持 `skills/` 中六个原子 Skill；正式课程资料放入 `docs/current/`，旧版继续保留但不进入主导航。确定性执行由Python命令完成，教学判断与证据边界留在Skill文本中。

**Tech Stack:** Python 3.10+、PyTorch、pytest、PowerShell/Bash、GitHub Actions。

**Spec:** `docs/superpowers/specs/2026-09-03-course-release-polish.md`

## Global Constraints

- 六个原子 Skill 数量和职责不变。
- `smoke`只用于环境检查，`classroom`才承担课堂效果展示。
- 不覆盖已有运行结果，不把课程阈值写成科学发现。
- Windows和Ubuntu均需可安装；课堂默认CPU、离线数据。

---

### Task 1: 发布证据包与正式16页资料

**Files:**
- Create: `nanoscigpt/evidence_pack.py`
- Create: `tests/test_evidence_pack.py`
- Create: `docs/current/course-outline-16p.md`
- Create: `docs/current/slide-copy-16p.md`
- Create: `docs/current/speaker-script-16p.md`
- Create: `docs/README.md`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_course_release.py`

**Interfaces:**
- Consumes: `run_report.json`、可选的`comparison.json`和`research_state.json`。
- Produces: `nanoscigpt-evidence-pack`命令、正式资料导航和可复查Markdown证据包。

- [x] 编写失败测试，要求正式资料连续覆盖P1—P16、时间从0连续到90分钟，并要求证据包入口可安装。
- [x] 运行目标测试，确认因文件和入口缺失而失败。
- [x] 加入证据包模块、五项已有边界测试和控制台入口`nanoscigpt-evidence-pack = "nanoscigpt.evidence_pack:main"`。
- [x] 将已确认的16页大纲、逐页文字和讲稿放入`docs/current/`，修正P11/P12的一分钟重叠。
- [x] 重写首页，使其只陈述十类数据、六个Skill、真实命令和当前边界。
- [x] 运行`pytest tests/test_evidence_pack.py tests/test_course_release.py -q`并提交。

### Task 2: 安装、环境检查与持续集成

**Files:**
- Create: `nanoscigpt/doctor.py`
- Create: `scripts/install_skills.ps1`
- Create: `scripts/install_skills.sh`
- Create: `.github/workflows/test-course.yml`
- Create: `tests/test_doctor.py`
- Create: `tests/test_skill_installers.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: 当前Python解释器、六个`skills/*/SKILL.md`和仓库数据清单。
- Produces: `nanoscigpt-doctor`、六Skill安装脚本和双平台CI。

- [ ] 编写失败测试，要求doctor返回Python/依赖/数据状态且安装脚本只复制六个Skill。
- [ ] 实现只读doctor；缺依赖返回非零，完整环境列出十类`ready`。
- [ ] 实现PowerShell与Bash安装脚本，目标目录显式传入，已有目录默认拒绝覆盖。
- [ ] 新增Windows/Ubuntu安装与smoke CI，完整课堂测试单独运行。
- [ ] 运行doctor、安装器沙箱测试和CI命令等价测试并提交。

### Task 3: 强化v1/v2状态安全

**Files:**
- Modify: `autoresearch/v1.py`
- Modify: `autoresearch/v2.py`
- Modify: `skills/ai-scientist-v1-workflow/SKILL.md`
- Modify: `skills/ai-scientist-v2-tree-search/SKILL.md`
- Test: `tests/test_ai_scientist_v1.py`
- Test: `tests/test_ai_scientist_v2.py`

**Interfaces:**
- Consumes: 已完成AutoResearch或v1状态。
- Produces: 不静默覆盖的v1目录、可从`running`恢复或明确失败的v2树状态。

- [ ] 添加v1重复输出和v2进程中断的失败测试。
- [ ] v1检测已有产物并要求新目录或显式`--overwrite`。
- [ ] v2在子进程异常、缺指标或中断时原子写入`failed`及恢复说明。
- [ ] 运行v1/v2全套测试和Codex CLI授权边界测试并提交。

### Task 4: 让v2比较不同研究路线

**Files:**
- Modify: `autoresearch/v1.py`
- Modify: `autoresearch/v2.py`
- Modify: `skills/ai-scientist-v2-tree-search/SKILL.md`
- Test: `tests/test_ai_scientist_v2.py`

**Interfaces:**
- Consumes: 同一V0、同一评价器、两条单变量路线。
- Produces: 一条预算路线和一条非预算路线，或在当前模型不支持时明确停在设计。

- [ ] 添加测试，拒绝两条只改变同一预算字段的伪多路线。
- [ ] 为序列课堂模型增加可执行的上下文长度备选路线；结构化领域在尚无等价路线时标为`design_only`。
- [ ] 保证每条路线只改变一个研究变量，最终决策仍使用同一评价器。
- [ ] 运行text、protein、weather三条端到端路线并提交。

### Task 5: 统一领域注册与自定义数据入口

**Files:**
- Create: `nanoscigpt/domains/registry.py`
- Create: `docs/current/custom-domain-guide.md`
- Create: `examples/custom_domain/README.md`
- Modify: `nanoscigpt/classroom.py`
- Modify: `autoresearch/experiment.py`
- Modify: `autoresearch/evaluator.py`
- Test: `tests/test_domain_registry.py`

**Interfaces:**
- Consumes: `DomainSpec(name, family, representation, task_name, source_kind)`。
- Produces: 唯一领域清单和自定义领域接入说明。

- [ ] 添加测试，要求十类领域只从注册表导出，并检查重复注册。
- [ ] 将散落的领域常量改为注册表查询，不改变现有命令输出。
- [ ] 提供自定义领域模板，明确必须实现读取、表示、预训练目标、下游任务和评价。
- [ ] 运行十类smoke、十类classroom和AutoResearch回归测试并提交。

### Task 6: 最终发布

**Files:**
- Modify: `README.md`
- Create: `CHANGELOG.md`

**Interfaces:**
- Consumes: 前五项全部产物。
- Produces: 可打标签的课程版本。

- [ ] 在干净worktree执行六Skill校验、全量pytest、安装后命令检查和一条完整protein链。
- [ ] 检查Markdown链接、第三方声明、未跟踪文件和Git差异。
- [ ] 更新变更记录，提交并推送分支供最终合并。
