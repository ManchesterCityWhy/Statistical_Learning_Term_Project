# T1 实验与提交文件流程

这套代码于 2026-11-24 在整理后的项目目录中新增。它实现开题和中期报告中的主要表达外推方法，以及一个保留零值的对照；不是原始中期脚本的恢复件，也尚未复现中期报告的数值。

2026-11-24 已用三份官方数据完成真实开发验证，并从 E10.5 生成本地候选。最新方法、分数与复现命令见 [T1 竞赛实验进展](../experiments/README.md)。

## 时间与官方资料

- 最终截止已改为 **2026-11-25 23:59 AoE，即北京时间 11 月 26 日 19:59**；应在 25 日完成，勿沿用报告中的 12 月 2 日。
- [时间线](https://virtualembryo.ai/challenge/timeline)
- [T1 任务](https://virtualembryo.ai/challenge/tasks/temporal)
- [数据与校验值](https://virtualembryo.ai/challenge/data)
- [当前评分](https://virtualembryo.ai/challenge/evaluation?section=scoring&task=1)
- [基线定义](https://virtualembryo.ai/challenge/baselines)
- [提交与使用规则](https://virtualembryo.ai/challenge/rules)
- [veckit 官方本地评价代码](https://github.com/aristoteleo/veckit)，固定版本 `cc3859ab2a2305efc4544815d3f44f660efa77d4`。

## 环境与数据

在项目根目录执行；建议 Python 3.12：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r t1_pipeline/requirements.txt
.venv/bin/python -m t1_pipeline sync-specs
.venv/bin/python -m t1_pipeline doctor
```

原始文件放在以下位置，或修改 `configs/t1.json` 中的绝对/项目相对路径：

```text
data/T1/E8.5_RNA.h5ad
data/T1/E9.5_RNA.h5ad
data/T1/E10.5_RNA.h5ad
```

前两份需与官网公布的 SHA-256 一致。E10.5 必须来自团队获准的官方 P3 validation opt-in；代码不会申请访问、登录账户或自动下载受限数据。真实表达直接使用发布的 log1p 值，不重新归一化。只凭数值非负、有限不能证明表达尺度正确。

`specs/t1/` 仅包含公开基因名和文件规格，不包含表达数据。数据、运行结果、预测文件和虚拟环境已由原有 `.gitignore` 排除。未经允许，不把原始数据或验证真值上传仓库。

## 开发验证

```bash
.venv/bin/python -m t1_pipeline inspect
.venv/bin/python -m t1_pipeline benchmark --output runs/dev_20261124
```

所有模型只拟合 E8.5/E9.5，预测 E10.5，步长为 1。默认比较七个候选、三个抽样种子（7、19、42），每次输出 2,000 个细胞：

1. 抽样 `copy_last`；
2. 按同名细胞类型均值差外推的 `pseudobulk_shift`；无前驱标签的群体不动；
3. 全空间收缩，alpha=0.5；
4. PCA10，alpha=0.5；
5. PCA10，alpha=1；
6. PCA30，alpha=0.5；
7. PCA10，alpha=0.5，仅对原有正值位置添加增量，原有零值不激活。

第 7 项是待检验的稀疏性对照：它不能激活原先为零的基因，未必改善预测，不能因为非零率下降就宣布成功。此轮优先解决官方评价及稀疏性，未重新实现中期的四组比例消融；应拿回原始脚本后对照复现。

PCA 每阶段最多抽 2,500 个细胞，使用全部基因、固定拟合种子。所有候选共享相同种子的源细胞抽样；均值与细胞类型均值通过分块计算，避免把全部训练矩阵转为稠密矩阵。

评价调用固定版 veckit 的 `common/core_metrics.py` 中 DES、DCS、无偏多核 MMD 和 variogram/CSS。RMSE、变化余弦、非零率单列为诊断，不能替代四项评分。MMD 的目标拟合 PCA 只用于评价，不进入预测模型。

本地评价固定抽取参考阶段的 10%、目标阶段的 10%，将目标抽样分成不重叠的评价半集和 ceiling 半集。公开工具与官方服务的全部抽样细节不保证相同，因此**这里重新计算本地 floor/ceiling，输出 `local_score`，不是官方 leaderboard 分数**。各候选均使用同一批评价细胞、同一参考和同一指标种子。数据量/抽样不同，不能直接与中期表或官网校准值比较。

每个输出目录必须是新目录，防止覆盖已有结果。产物包括：

- `manifest.json`：配置、代码校验值、输入文件校验值、环境及完成状态；
- `evaluation_indices.npz`：固定的参考、评价与 ceiling 行号；
- `local_anchors.json`：本轮本地校准锚点；
- `metrics.csv`、`summary.csv`：逐种子数值和均值/样本标准差；
- 每组 JSON：候选配置、四项原始指标及稀疏性诊断；
- `pca_diagnostics.json`：解释方差和训练均值差保留比例。

默认不保存全部大矩阵；需要时加 `--keep-predictions`。抽样标准差只描述算法抽样波动，不是跨胚胎的生物学置信区间。某指标缺失按零 skill 处理；本地 floor/ceiling 无法区分时保留原始指标并标记不能校准，不伪造总分。

## 选模、两天外推与提交

阅读完整结果后，由组员决定候选。以下候选名只是命令示例，并不表示它已经胜出：

```bash
.venv/bin/python -m t1_pipeline freeze --run runs/dev_20261124 --candidate pca10_a05 --output configs/frozen_selection.json
.venv/bin/python -m t1_pipeline predict-final --output predictions/final_20261125
.venv/bin/python -m t1_pipeline validate predictions/final_20261125/E12.5_prediction.h5ad --board T1:test
```

`freeze` 只接受已完成的本地验证记录。最终模型重新用 E9.5/E10.5 拟合，步长为 2；不读取 E8.5 或任何 E12.5 真值。基因顺序来自目标 board 的公开清单，细胞标签不写入预测。代码或数据在验证后变化时要求重新验证并冻结。

`predict-final` 输出文件、校验信息和状态 `prediction_ready_not_submitted`。本地校验只检查格式，不能证明生物学质量，也不等于官方有效提交。随后在团队账户完成官方 format check，再正式提交并保存提交编号、回执与方法摘要。P3 每个 test board 全阶段最多两次正式提交；先查剩余次数。

## 下一步待办

1. 找回数据、原始中期代码和完整 13 方案结果，确认 E10.5 来源与团队账号提交状态。
2. 跑通输入检查和七候选官方指标验证，复核新结果与中期设置的差异。
3. 选择方法、冻结配置、生成 E12.5、完成官方检查与至少一次有效提交。
4. 补做比例消融、分群误差、稀疏性与基因变化图；据实记录失败结果。
5. 将代码、配置、汇总表和报告同步到 GitHub；原始数据留在授权本地环境。
6. 按课程要求准备汇报与 8–12 页最终正文，补齐成员贡献、Code and Data Availability Statement、AI Assistance Statement。

## 软件测试

```bash
.venv/bin/python -m pytest tests -q
```

模拟数据检查涵盖基因重排、稀疏分块读取、alpha=0 退回复制、两天倍率、新标签回退、保零限制、真实 veckit 指标调用、校准方向、文件导出和最终阶段选择。模拟文件均由测试写入临时目录，不能用于竞赛提交。
