# T1 实验归档与复现

先读[期末报告](../reports/final/T1_最终报告.pdf)和[第二次官方提交复盘](../reports/final/T1_第二次提交复盘.md)。这里按实验轮次索引代码和结果。候选文档里的推荐顺序及预估分数保留提交前口径，两次正式成绩见[仓库首页](../README.md#官方测试成绩)。

| 轮次 | 核心方法或分析 | 代码、评分与记录 |
| --- | --- | --- |
| 1 | 复制基线、均值平移、PCA、保零外推、bulk 趋势和单细胞迁移 | [评分与清单](archive/2026-11-24/README.md) · [复现命令](#第一轮2026-11-24) |
| 2 | bulk 趋势及带岭惩罚的非负最小二乘反卷积 | [说明与评分表](round2_seed42/README.md) · [第一次正式提交 h1 清单](round2_seed42/prediction_h1_manifest.json) |
| 3 | 细胞级熵平衡重加权、组成与表达的已知真值分析 | [熵平衡评分](round3_seed42/round3_scores.csv) · [已知真值分析评分](round3_seed42/oracle_scores.csv) · [复现命令](#第三轮与上限分解) |
| 4 | 方向回归与测序深度对齐 | [方向回归脚本](round4_direction.py) · [后续对照与方向分数](round5_seed42/README.md) |
| 5 | 同平台 E14.5 心脏细胞核作为发育参照 | [说明](round5_seed42/README.md) · [评分表](round5_seed42/scores/) |
| 6 | E10.5 心脏文库作为基底、校正 E14.5 参照 | [说明](round6_seed42/README.md) · [评分表](round6_seed42/scores/) · [运行日志](round6_seed42/logs/) |
| 7 | 比较 E14.5 重采样比例与不同表达位移 | [说明](round7_seed42/README.md) · [评分表](round7_seed42/scores/) · [运行日志](round7_seed42/logs/) |
| 8 | 用 E10.5 真实细胞作为替身，补入部分低表达基因信号 | [说明](round8_seed42/README.md) · [评分与日志归档](round7_seed42/) · [导出清单](round8_seed42/manifest.json) |
| 9 | 在第八轮方案上对源细胞的分子计数抽稀 | [说明](round9_seed42/README.md) · [评分与日志归档](round7_seed42/) · [第二次正式提交清单](round9_seed42/manifest.json) |

## 第一轮（2026-11-24）

[逐次实验评分与运行归档](archive/2026-11-24/README.md) · [中期报告公开版](../reports/T1_中期报告_公开版.pdf)

每轮用同一组参考细胞和同一组 E10.5 评价细胞比较候选，抽样种子为 0、31、101。生成开发预测时，竞赛数据的拟合阶段限于 E8.5 和 E9.5，E10.5 用于开发评价和候选选择。不同抽样种子复用同一阶段的数据，不能视为独立胚胎。

| 方法 | 种子 0 | 种子 31 | 种子 101 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| 分类型保零 + GEO bulk 发育趋势 | 52.269 | 53.201 | 52.996 | 52.822 |
| 分类型保零 + E14.5 单细胞输运 | 52.079 | 52.526 | 52.940 | 52.515 |
| 细胞状态重加权 + 分类型保零 | 51.083 | 53.344 | 52.423 | 52.283 |
| 分类型保零、归一化 | 50.451 | 51.268 | 51.077 | 50.932 |
| 保零 PCA10 | 49.811 | 50.250 | 50.196 | 50.086 |
| 复制最近阶段 | 50.000 | 50.000 | 50.000 | 50.000 |

以上为本地开发分数，经本轮抽样各自的 floor 和 ceiling 校准。bulk 组合的平均 MMD 为 0.07032，复制基线为 0.08500。单细胞输运组合的平均 CSS 为 0.00387，复制基线为 0.00571。三组领先模型的平均 DES 仍为负，差异表达恢复是主要缺口。

## 数据和三种候选

- 竞赛输入为官方下载的 T1 E8.5、E9.5 和 P3 已授权的 E10.5。竞赛数据由 UC San Diego 的 Neil Chi 团队生成，并由 Virtual Embryo Challenge 发布。最终预测可用 E10.5，测试 E12.5 真值始终保留在官方。 [数据](https://virtualembryo.ai/challenge/data) · [规则](https://virtualembryo.ai/challenge/rules)
- `bulk_type_075` 使用 GSE223237 的 3 个 E9.5 样本（GSM6943750–52）和 3 个 E14.5 样本（GSM6943761–63）。只下载上述 6 个单独的计数文件，另用 GSM6211349 的基因注释表做 Ensembl ID 转换。先按组进行保零平移，再迁移六个 bulk 重复估计出的基因级变化，并恢复每细胞 10,000 的归一化尺度。 [GEO 研究](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223237) · [基因注释来源](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6211349)
- `geo_type_075` 使用 GSE150817 的 CELLxGENE 版本。仅从原始计数中取 E14.5、正常、心脏的 8,889 个细胞。先做分类型保零平移，再按跨平台的细胞状态匹配结果做部分输运，联合调整非零项和表达幅度。 [原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13602375/) · [CELLxGENE 数据](https://cellxgene.cziscience.com/collections/613f5480-4957-4f80-b804-0e2b85ac454c)
- `type_keepzero_weighted_norm` 仅用竞赛数据。在共同变异基因的 PCA 表示中拟合早、晚阶段的分类器，用经截断的分类 log-odds 给晚阶段细胞重加权。对同名细胞群体用观测到的均值差做保零平移，最后重新归一化。

## 复现

先按 [T1 运行说明](../t1_pipeline/README.md) 初始化环境、同步公开基因清单，并把受限竞赛数据放到 `data/T1/`。

外部数据放到以下本地目录。GEO 文件用单个 GSM 样本条目里的补充文件名。CELLxGENE 选择该集合的 H5AD 格式，下载后重命名。

```text
data/external/GSE223237/GSM6943750_E9_Rep1_counts.txt.gz
data/external/GSE223237/GSM6943751_E9_Rep2_counts.txt.gz
data/external/GSE223237/GSM6943752_E9_Rep3_counts.txt.gz
data/external/GSE223237/GSM6943761_E14_Rep1_counts.txt.gz
data/external/GSE223237/GSM6943762_E14_Rep2_counts.txt.gz
data/external/GSE223237/GSM6943763_E14_Rep3_counts.txt.gz
data/external/GSM6211349/GSM6211349_E14_genes.tsv.gz
data/external/GSE150817/perinatal_heart.h5ad
```

```bash
.venv/bin/python -m t1_pipeline benchmark --config configs/t1_competitive.json --output runs/competitive_screen
.venv/bin/python -m t1_pipeline benchmark --config configs/t1_competitive_verify.json --output runs/competitive_verify_0
.venv/bin/python -m experiments.bulk_trend benchmark --output runs/competitive_bulk --seeds 0 31 101
.venv/bin/python -m experiments.geo_transport benchmark --output runs/competitive_geo_transport --seeds 0 31 101
```

重跑另外两轮内部对照时，将 `t1_competitive_verify.json` 分别复制到两个新配置，将 `seeds` 改为 `[31]` 或 `[101]`，将 `evaluation_seed` 改为相同的值，指定各自的输出目录。

当前两种主要候选的 E12.5 文件均由种子 0 生成。下面的命令会重建文件并执行本地格式检查。

```bash
.venv/bin/python -m experiments.bulk_trend export --candidate bulk_type_075 --seeds 0 --output predictions/e125_competitive_bulk
.venv/bin/python -m experiments.geo_transport export --candidate geo_type_075 --seeds 0 --output predictions/e125_competitive_geo
```

候选的训练/生成步骤、输入文件校验值及配置见输出目录内的 manifest。代码开发、调试和结果汇总使用了 Codex 辅助，实验记录供组员复核。原始矩阵、评价子集和预测文件存放在授权的本地环境。

## 第三轮与上限分解

现有记录中，熵平衡选中的配置 `s5_ess0.15_b3` 在选模集得 57.53 分，复核集得 56.50 分，低于第二轮 59.79 分，未用于正式提交。已知真值分析在复核集上的结果见[期末报告](../reports/final/T1_最终报告.pdf)和[原始评分表](round3_seed42/oracle_scores.csv)。

两个脚本都复用第二轮的细胞划分 `runs/expanded_seed42/indices.npz`（缺失时按第二轮的随机调用顺序重建），分数可以和 `round2_seed42/` 直接比较。若本地还有 `runs/composition_refined_seed42/sampled_rows.npz`，脚本会先重算第二轮最佳模型，应得到选模集 60.50、复核集 59.79，用来确认环境一致。

```bash
python -m pytest tests/test_round3.py -q
python -m experiments.oracle_decomposition --with-confirmation   # 输出 runs/oracle_decomposition_seed42/
python -m experiments.round3_balance benchmark                   # 输出 runs/round3_balance_seed42/
python -m experiments.round3_balance export --config-name <选模集第一名> --h 1
```

- `oracle_decomposition` 的每个条件都使用 E10.5 真值（取自不参与任何评价的细胞），只回答“如果知道真实组成或真实表达变化，最多能得多少分”，**不得作为预测提交**。
- `round3_balance` 在 E9.5 细胞上求熵平衡权重：权重是对 30 个细胞状态主成分的指数倾斜，岭系数由“相对有效样本量不低于 0.5、0.3 或 0.15”反解。只按选模集挑选，前三名在第一复核集上各评一次。事先约定：第一复核集超过第二轮的 59.79 至少 0.5 分才用于 E12.5。
