# Baseline Builder Skill｜GLM-5.3 CLI实测

## 测试设置

- Codex CLI：0.152.0
- 模型：`scnet/GLM-5.3`
- 模式：`read-only`、`ephemeral`、`minimal`
- 要求：读取仓库Skill后，只回复此刻会对学生说的话，不执行命令

## 修改前发现的问题

仓库原来使用`research-baseline-builder`，与本机已经安装的同名通用Skill冲突。一次FITS分类测试同时读取了仓库版和全局版，并把“仓库有图像模板”说成可以直接运行学生的FITS数据。实际课程CLI当时只能直接运行LAMOST样例和带标签CSV分类/回归。

因此将仓库Skill改名为`nanoscigpt-research-baseline-builder`，并把可运行范围写进Skill：时序、普通图像和FITS先做任务识别与接入设计，不在没有数据加载器时声称已经训练。

## 修改后复测

### 学生要求直接运行FITS分类

> 我理解：一批FITS图像，按labels.csv标签做分类。先说明边界：当前基线命令只能直接跑带标签CSV的表格任务，还不能直接读取FITS图像训练，这一步尚未运行。
> 只问一个问题：labels.csv里哪一列对应图像文件名？
> 下条消息带回该列名和几个文件名样例，我们先核对标签与图像能否对上。

### 学生使用内置LAMOST课程数据

> 我理解：你想用仓库自带的2000条LAMOST课程光谱估计恒星有效温度teff，一条光谱就是一个样本。目标明确，我先不继续追问。下一步我会检查当前Python能否导入numpy、pandas、sklearn，再从仓库根目录运行lamost基线；这一步还没有运行。等结果文件生成后，我们再看这份指标能说明什么、不能说明什么。

### 学生给出目标明确的材料CSV

第三个并发测试在SCNet本地代理从WebSocket回退后长时间没有返回，已人工终止；没有把它计入Skill行为通过项。仓库自身的CSV命令由自动化测试覆盖，模型回答仍需在服务稳定时补做一次。

### 学生给出带时间列的单变量时序

> 我理解为：sensor.csv 是一条按小时的温度序列，timestamp 是时间列，temperature 是数值列，目标是预测下一小时。第一步先跑时序基线：`python -m nanoscigpt.baseline --series-csv <sensor.csv绝对路径> --value-column temperature --time-column timestamp --out_root out/baseline`，这一步还没有运行。跑完把 metrics.json、baseline_summary.json 和 train_log.txt 带回来，我们再看下一小时误差能说明什么。

## 当前判断

命名冲突已经消除。GLM-5.3能够对可直接运行的LAMOST示例和暂不能直接运行的FITS数据给出不同回答，并且不会假装已经训练。随后已补入单条数值时序CSV入口，可指定时间列排序或明确沿用现有行顺序；普通图像和FITS仍需数据加载器。
