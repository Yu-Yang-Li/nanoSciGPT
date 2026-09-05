# 原版 v1 失败处理教学修补

修补对象：SakanaAI/AI-Scientist 固定版本 `1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb` 的 `ai_scientist/perform_experiments.py`。没有更改研究提示词、想法生成、改码或写作的模型流程。

## 已实现

- 非零退出和超时不删除结果目录：每次失败的部分文件、当次代码及完整 stderr 保存到 `failed_runs/run_N_<唯一后缀>/`，同一轮重试不会覆盖此前记录。
- 成功实验继续读取原来的 `run_N/final_info.json`。
- 没有完成新实验，或绘图重试耗尽后仍失败，返回 False，停止当前 idea，而不是据模型自报或一个成功布尔值进入写作。
- 配置入口 `python -m nanoscigpt.upstream configure-failures v1 --root out/upstream` 只应用补丁，不运行研究。已有学生修改或来源版本不匹配时拒绝覆盖。

[完整补丁](teaching_failure_changes.diff)与[配置收据](teaching_failure_setup.json)来自实际独立原版副本 `out/native-api-final-20260905/v1`；收据仍标为 `configured_not_run`，没有把配置成功当作研究完成。

## 怎么测的

先在未应用修补的原版执行器上运行相同子进程测试，观察到4失败、2通过：非零退出和超时丢失部分结果、绘图失败仍返回 True、未运行新实验仍返回 True。随后应用修补，失败处理、模型路由和源码准备三组共20项通过，见[首轮测试报告](tests.xml)。这些测试实际运行了退出码7、超时、正常结果和绘图子进程；仅远程 coder 的回复使用测试替身，不能称为20次真实 Agent 研究。

另将对话记录脚本的持久记忆覆盖加入首轮和 resume 命令：先观察到2失败、2通过，修改后[4项捕获测试](capture-tests.xml)通过。它们检查启动参数和记录，不代表模型交互质量已经通过。

随后增加配置CLI入口测试并完成全量回归：**185项通过，389.30秒，0失败、0跳过**，见[全量报告](full-tests.xml)。这是本仓库全部`tests/`的结果；其中模型回复替身不冒充真实模型，原项目完整研究仍未验收。

仍未实现整场 API/墙钟预算或 v2 停止恢复；单个实验默认超时也没有冒充整场限制。完整原版研究、学生连续教学、远端发布仍须分别验收。
