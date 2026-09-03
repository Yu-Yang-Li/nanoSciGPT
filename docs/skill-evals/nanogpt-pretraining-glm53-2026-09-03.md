# nanoGPT Pretraining Skill｜GLM-5.3 CLI实测

## 发现的问题

P3已经写成“预训练＋微调”，但原来的text课堂代码只冻结预训练模型、训练一个任务头。GLM-5.3读取原Skill后据此回答：

> 第二段不是微调。它叫“任务头训练”：预训练模型参数被冻结，只训练接在表示后面的小任务头。

这个回答忠实于旧代码，却与已经确认的P3讲法冲突。

## 修正

text课堂路径现在留下：

- `model/ckpt.pt`：微调前的预训练模型；
- `downstream/finetuned_ckpt.pt`：预训练参数和任务输出层共同更新后的模型；
- `downstream/downstream_result.json`：记录 `training_mode=full_fine_tune`、`encoder_frozen=false` 与 `pretrained_parameters_updated=true`。

测试同时比较两个检查点中的模型权重，不能只凭状态字段宣称发生了微调。protein、DNA、SMILES及六个结构化科学样例仍使用冻结表示的任务头，P4相应改称“预训练及下游任务”。

## GLM-5.3复测

学生问“我运行了text课堂样例，第二段到底是不是微调？”时，修改后的Skill回答：

> 第二段确实是微调。第一段只用无标签文本练“预测下一个字符”，是预训练；第二段用少量课堂标签，同时更新任务输出层和预训练参数。可看 `downstream/downstream_result.json` 中 `pretrained_parameters_updated=true`，并对比 `model/ckpt.pt` 与 `downstream/finetuned_ckpt.pt` 权重，确为微调而非冻结任务头。

本次CLI使用Codex 0.152.0、`scnet/GLM-5.3`、`read-only`、`ephemeral`与`minimal`设置。
